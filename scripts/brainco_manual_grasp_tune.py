"""RAVEN manual BrainCo Revo2 grasp-pose calibration.

The script first moves the LEFT G1 arm automatically into the current
human-style near-grasp pose.

It then switches to MANUAL mode.

Manual controls
---------------
W / S : move hand toward / away from handle
A / D : move hand along the handle
Q / E : move hand up / down

Z / X : local hand roll
T / G : local hand pitch
C / V : local hand yaw

P     : print exact calibrated pose and 7 arm joint values
R     : reset to the starting manual pose

Important
---------
- G1 base remains fixed.
- BrainCo fingers remain open.
- Tripod pan/tilt remain locked.
- This tool is ONLY for finding the exact human-looking grasp pose.
"""

# =============================================================================
# Standard library
# =============================================================================

import argparse
import copy
import math


# =============================================================================
# Isaac Sim launcher
# =============================================================================

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="RAVEN manual BrainCo grasp-pose calibration."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


# =============================================================================
# Isaac Lab imports
# =============================================================================

import torch

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation

from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

from isaaclab.devices.keyboard.se3_keyboard import (
    Se3Keyboard,
    Se3KeyboardCfg,
)

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_slerp,
    subtract_frame_transforms,
)

from raven_g1.assets import (
    G1_BRAINCO_MODE15_CFG,
    TRIPOD_CAMERA_CFG,
)


# =============================================================================
# Configuration
# =============================================================================

TRIPOD_X = 0.60


# -----------------------------------------------------------------------------
# Automatic setup
# -----------------------------------------------------------------------------

REACH_DURATION_S = 8.0

APPROACH_DURATION_S = 4.0

AUTO_DURATION_S = (
    REACH_DURATION_S
    + APPROACH_DURATION_S
)


# -----------------------------------------------------------------------------
# Tripod handle
# -----------------------------------------------------------------------------

HANDLE_HALF_LENGTH = 0.16

GRASP_ALONG_HANDLE_X = -0.10


# -----------------------------------------------------------------------------
# Existing near-grasp geometry
# -----------------------------------------------------------------------------

PREGRASP_PALM_DISTANCE = 0.10

FINAL_PALM_DISTANCE = 0.040

PALM_CENTER_FRACTION = 0.42


# -----------------------------------------------------------------------------
# IK
# -----------------------------------------------------------------------------

DLS_DAMPING = 0.15

MAX_ARM_JOINT_STEP_RAD = math.radians(
    0.10
)

MAX_WRIST_DEVIATION_RAD = math.radians(
    45.0
)


# -----------------------------------------------------------------------------
# Manual movement speed
#
# Se3Keyboard outputs a command while a key is held.
# We integrate it using sim_dt.
# -----------------------------------------------------------------------------

MANUAL_TRANSLATION_SPEED = 0.040

MANUAL_ROTATION_SPEED = 0.35


# =============================================================================
# Helpers
# =============================================================================


def resolve_body(
    articulation,
    body_name: str,
) -> int:

    body_ids, _ = articulation.find_bodies(
        [body_name]
    )

    if len(body_ids) == 0:
        raise RuntimeError(
            f"Body '{body_name}' not found."
        )

    body_id = int(
        body_ids[0]
    )

    print(
        f"[RAVEN] Resolved "
        f"{body_name} -> body ID {body_id}"
    )

    return body_id


def resolve_joint(
    articulation,
    joint_name: str,
) -> int:

    joint_ids, _ = articulation.find_joints(
        [joint_name]
    )

    if len(joint_ids) == 0:
        raise RuntimeError(
            f"Joint '{joint_name}' not found."
        )

    return int(
        joint_ids[0]
    )


def normalize_vector(
    vector,
    eps: float = 1.0e-8,
):

    magnitude = torch.linalg.vector_norm(
        vector
    )

    if magnitude.item() < eps:
        raise RuntimeError(
            "Attempted to normalize a near-zero vector."
        )

    return (
        vector
        / magnitude
    )


def smoothstep(
    value: float,
) -> float:

    value = max(
        0.0,
        min(
            1.0,
            value,
        ),
    )

    return (
        3.0 * value * value
        - 2.0 * value * value * value
    )


def vec_string(
    vector,
) -> str:

    return (
        "("
        f"{vector[0].item(): .6f}, "
        f"{vector[1].item(): .6f}, "
        f"{vector[2].item(): .6f}"
        ")"
    )


def quat_string(
    quaternion,
) -> str:

    return (
        "("
        f"{quaternion[0].item(): .8f}, "
        f"{quaternion[1].item(): .8f}, "
        f"{quaternion[2].item(): .8f}, "
        f"{quaternion[3].item(): .8f}"
        ")"
    )


def point_to_segment_distance(
    point,
    segment_a,
    segment_b,
):

    segment = (
        segment_b
        - segment_a
    )

    length_squared = torch.dot(
        segment,
        segment,
    )

    if length_squared.item() < 1.0e-12:
        return torch.linalg.vector_norm(
            point
            - segment_a
        )

    t = (
        torch.dot(
            point - segment_a,
            segment,
        )
        / length_squared
    )

    t = torch.clamp(
        t,
        0.0,
        1.0,
    )

    closest = (
        segment_a
        + t * segment
    )

    return torch.linalg.vector_norm(
        point
        - closest
    )


# =============================================================================
# Main
# =============================================================================


def main():

    # =========================================================================
    # Simulation
    # =========================================================================

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
    )

    sim = sim_utils.SimulationContext(
        sim_cfg
    )

    sim_dt = sim.get_physics_dt()


    # =========================================================================
    # Ground
    # =========================================================================

    ground_cfg = sim_utils.GroundPlaneCfg()

    ground_cfg.func(
        "/World/Ground",
        ground_cfg,
    )


    # =========================================================================
    # Light
    # =========================================================================

    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(
            0.75,
            0.75,
            0.75,
        ),
    )

    light_cfg.func(
        "/World/Light",
        light_cfg,
        translation=(
            1.0,
            0.0,
            10.0,
        ),
    )


    # =========================================================================
    # G1 + BrainCo
    # =========================================================================

    robot_cfg = copy.deepcopy(
        G1_BRAINCO_MODE15_CFG
    )

    robot_cfg.prim_path = (
        "/World/G1BrainCo"
    )

    robot = Articulation(
        robot_cfg
    )


    # =========================================================================
    # Tripod
    # =========================================================================

    tripod_cfg = copy.deepcopy(
        TRIPOD_CAMERA_CFG
    )

    tripod_cfg.prim_path = (
        "/World/TripodCamera"
    )

    tripod_cfg.init_state.pos = (
        TRIPOD_X,
        0.0,
        0.0,
    )

    tripod = Articulation(
        tripod_cfg
    )


    # =========================================================================
    # Viewer
    # =========================================================================

    sim.set_camera_view(
        eye=(
            2.3,
            2.2,
            1.65,
        ),
        target=(
            0.30,
            0.0,
            0.88,
        ),
    )


    # =========================================================================
    # Initialise
    # =========================================================================

    sim.reset()

    neutral_robot_joint_pos = (
        robot.data.default_joint_pos.clone()
    )

    zero_robot_joint_vel = torch.zeros_like(
        neutral_robot_joint_pos
    )

    neutral_tripod_joint_pos = (
        tripod.data.default_joint_pos.clone()
    )

    zero_tripod_joint_vel = torch.zeros_like(
        neutral_tripod_joint_pos
    )


    # =========================================================================
    # Settle
    # =========================================================================

    for _ in range(10):

        robot.write_joint_state_to_sim(
            neutral_robot_joint_pos,
            zero_robot_joint_vel,
        )

        tripod.write_joint_state_to_sim(
            neutral_tripod_joint_pos,
            zero_tripod_joint_vel,
        )

        robot.write_data_to_sim()

        tripod.write_data_to_sim()

        sim.step()

        robot.update(
            sim_dt
        )

        tripod.update(
            sim_dt
        )


    # =========================================================================
    # Left arm
    # =========================================================================

    arm_joint_names = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]

    arm_joint_ids = [
        resolve_joint(
            robot,
            joint_name,
        )
        for joint_name
        in arm_joint_names
    ]

    wrist_joint_ids = [
        arm_joint_ids[4],
        arm_joint_ids[5],
        arm_joint_ids[6],
    ]


    # =========================================================================
    # Hand bodies
    # =========================================================================

    hand_base_id = resolve_body(
        robot,
        "left_base_link",
    )

    index_tip_id = resolve_body(
        robot,
        "left_index_tip_Link",
    )

    middle_tip_id = resolve_body(
        robot,
        "left_middle_tip_Link",
    )

    ring_tip_id = resolve_body(
        robot,
        "left_ring_tip_Link",
    )

    pinky_tip_id = resolve_body(
        robot,
        "left_pinky_tip_Link",
    )


    # =========================================================================
    # Tripod
    # =========================================================================

    handle_id = resolve_body(
        tripod,
        "control_handle_link",
    )

    pan_joint_id = resolve_joint(
        tripod,
        "pan_joint",
    )

    tilt_joint_id = resolve_joint(
        tripod,
        "tilt_joint",
    )


    # =========================================================================
    # Jacobian
    # =========================================================================

    if not robot.is_fixed_base:
        raise RuntimeError(
            "Manual calibration requires the G1 base to be fixed."
        )

    hand_jacobian_index = (
        hand_base_id - 1
    )


    # =========================================================================
    # Initial hand geometry
    # =========================================================================

    initial_hand_pos_w = (
        robot.data.body_pos_w[
            0,
            hand_base_id,
        ].clone()
    )

    initial_hand_quat_w = (
        robot.data.body_quat_w[
            0,
            hand_base_id,
        ].clone()
    )

    index_initial = (
        robot.data.body_pos_w[
            0,
            index_tip_id,
        ].clone()
    )

    middle_initial = (
        robot.data.body_pos_w[
            0,
            middle_tip_id,
        ].clone()
    )

    ring_initial = (
        robot.data.body_pos_w[
            0,
            ring_tip_id,
        ].clone()
    )

    pinky_initial = (
        robot.data.body_pos_w[
            0,
            pinky_tip_id,
        ].clone()
    )

    finger_centroid_initial = (
        index_initial
        + middle_initial
        + ring_initial
        + pinky_initial
    ) / 4.0


    # =========================================================================
    # Natural semantic hand frame
    #
    # X = pinky -> index
    # Y = palm -> fingertips
    # Z = palm normal
    # =========================================================================

    current_across_w = normalize_vector(
        index_initial
        - pinky_initial
    )

    current_forward_raw_w = (
        finger_centroid_initial
        - initial_hand_pos_w
    )

    current_forward_w = (
        current_forward_raw_w
        - torch.dot(
            current_forward_raw_w,
            current_across_w,
        )
        * current_across_w
    )

    current_forward_w = normalize_vector(
        current_forward_w
    )

    current_normal_w = normalize_vector(
        torch.cross(
            current_across_w,
            current_forward_w,
            dim=0,
        )
    )

    current_forward_w = normalize_vector(
        torch.cross(
            current_normal_w,
            current_across_w,
            dim=0,
        )
    )

    current_hand_basis_w = torch.stack(
        (
            current_across_w,
            current_forward_w,
            current_normal_w,
        ),
        dim=1,
    )


    # =========================================================================
    # Palm centre
    # =========================================================================

    palm_center_initial_w = (
        initial_hand_pos_w
        + PALM_CENTER_FRACTION
        * (
            finger_centroid_initial
            - initial_hand_pos_w
        )
    )

    initial_hand_rot_w = matrix_from_quat(
        initial_hand_quat_w.unsqueeze(0)
    )[0]

    palm_center_offset_hand = (
        initial_hand_rot_w.transpose(
            0,
            1,
        )
        @ (
            palm_center_initial_w
            - initial_hand_pos_w
        )
    )


    # =========================================================================
    # Handle geometry
    # =========================================================================

    handle_pos_w = (
        tripod.data.body_pos_w[
            0,
            handle_id,
        ].clone()
    )

    handle_quat_w = (
        tripod.data.body_quat_w[
            0,
            handle_id,
        ].clone()
    )

    handle_rot_w = matrix_from_quat(
        handle_quat_w.unsqueeze(0)
    )[0]

    raw_handle_axis_w = normalize_vector(
        handle_rot_w
        @ torch.tensor(
            [
                1.0,
                0.0,
                0.0,
            ],
            device=robot.device,
            dtype=torch.float32,
        )
    )

    grasp_pos_w = (
        handle_pos_w
        + raw_handle_axis_w
        * GRASP_ALONG_HANDLE_X
    )


    # =========================================================================
    # Choose handle-axis direction
    # =========================================================================

    handle_axis_w = (
        raw_handle_axis_w.clone()
    )

    if torch.dot(
        handle_axis_w,
        current_across_w,
    ).item() < 0.0:

        handle_axis_w = (
            -handle_axis_w
        )


    # =========================================================================
    # Approach direction
    # =========================================================================

    handle_to_palm_w = (
        palm_center_initial_w
        - grasp_pos_w
    )

    approach_out_w = (
        handle_to_palm_w
        - torch.dot(
            handle_to_palm_w,
            handle_axis_w,
        )
        * handle_axis_w
    )

    approach_out_w = normalize_vector(
        approach_out_w
    )

    approach_in_w = (
        -approach_out_w
    )


    # =========================================================================
    # Natural handshake orientation
    # =========================================================================

    desired_across_w = (
        handle_axis_w
    )

    desired_forward_w = (
        approach_in_w
    )

    desired_normal_w = normalize_vector(
        torch.cross(
            desired_across_w,
            desired_forward_w,
            dim=0,
        )
    )

    desired_forward_w = normalize_vector(
        torch.cross(
            desired_normal_w,
            desired_across_w,
            dim=0,
        )
    )

    desired_hand_basis_w = torch.stack(
        (
            desired_across_w,
            desired_forward_w,
            desired_normal_w,
        ),
        dim=1,
    )

    semantic_rotation = (
        desired_hand_basis_w
        @ current_hand_basis_w.transpose(
            0,
            1,
        )
    )

    target_hand_rot_w = (
        semantic_rotation
        @ initial_hand_rot_w
    )

    target_hand_quat_w = quat_from_matrix(
        target_hand_rot_w.unsqueeze(0)
    )[0]


    # =========================================================================
    # Automatic palm targets
    # =========================================================================

    pregrasp_palm_w = (
        grasp_pos_w
        + PREGRASP_PALM_DISTANCE
        * approach_out_w
    )

    final_palm_w = (
        grasp_pos_w
        + FINAL_PALM_DISTANCE
        * approach_out_w
    )

    target_palm_offset_w = (
        target_hand_rot_w
        @ palm_center_offset_hand
    )

    pregrasp_hand_base_w = (
        pregrasp_palm_w
        - target_palm_offset_w
    )

    final_hand_base_w = (
        final_palm_w
        - target_palm_offset_w
    )


    # =========================================================================
    # Convert automatic poses to robot frame
    # =========================================================================

    root_pos_w = (
        robot.data.root_pos_w[
            0
        ].unsqueeze(0)
    )

    root_quat_w = (
        robot.data.root_quat_w[
            0
        ].unsqueeze(0)
    )

    (
        initial_hand_pos_b,
        initial_hand_quat_b,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        initial_hand_pos_w.unsqueeze(0),
        initial_hand_quat_w.unsqueeze(0),
    )

    (
        pregrasp_hand_pos_b,
        target_hand_quat_b,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        pregrasp_hand_base_w.unsqueeze(0),
        target_hand_quat_w.unsqueeze(0),
    )

    (
        final_hand_pos_b,
        _,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        final_hand_base_w.unsqueeze(0),
        target_hand_quat_w.unsqueeze(0),
    )


    # =========================================================================
    # IK
    # =========================================================================

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={
            "lambda_val": DLS_DAMPING,
        },
    )

    ik_controller = DifferentialIKController(
        ik_cfg,
        num_envs=1,
        device=robot.device,
    )

    ik_controller.reset()


    # =========================================================================
    # Keyboard
    # =========================================================================

    keyboard_cfg = Se3KeyboardCfg(
        sim_device=str(
            robot.device
        ),
        gripper_term=False,
        pos_sensitivity=MANUAL_TRANSLATION_SPEED,
        rot_sensitivity=MANUAL_ROTATION_SPEED,
    )

    keyboard = Se3Keyboard(
        keyboard_cfg
    )


    # =========================================================================
    # Callback flags
    # =========================================================================

    flags = {
        "print": False,
        "reset": False,
    }


    def request_print():

        flags["print"] = True


    def request_reset():

        flags["reset"] = True


    keyboard.add_callback(
        "P",
        request_print,
    )

    keyboard.add_callback(
        "R",
        request_reset,
    )


    # =========================================================================
    # Manual target state
    # =========================================================================

    manual_initialized = False

    manual_target_pos_w = None
    manual_target_quat_w = None

    manual_reset_pos_w = None
    manual_reset_quat_w = None


    # =========================================================================
    # Startup information
    # =========================================================================

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN MANUAL BRAINCO GRASP CALIBRATION"
    )

    print(
        "============================================================"
    )

    print()

    print(
        "[RAVEN] Robot will first move automatically "
        "to the current near-grasp pose."
    )

    print(
        "[RAVEN] Manual control becomes active after "
        f"{AUTO_DURATION_S:.1f} seconds."
    )

    print()


    # =========================================================================
    # Main loop
    # =========================================================================

    step = 0


    while simulation_app.is_running():

        with torch.inference_mode():

            time_s = (
                step
                * sim_dt
            )


            # =================================================================
            # Current poses
            # =================================================================

            root_pose_w_now = (
                robot.data.root_state_w[
                    :,
                    0:7,
                ]
            )

            hand_pose_w_now = (
                robot.data.body_state_w[
                    :,
                    hand_base_id,
                    0:7,
                ]
            )

            (
                hand_pos_b,
                hand_quat_b,
            ) = subtract_frame_transforms(
                root_pose_w_now[:, 0:3],
                root_pose_w_now[:, 3:7],
                hand_pose_w_now[:, 0:3],
                hand_pose_w_now[:, 3:7],
            )


            # =================================================================
            # Automatic reach
            # =================================================================

            if time_s < REACH_DURATION_S:

                phase = "AUTO_REACH"

                alpha = smoothstep(
                    time_s
                    / REACH_DURATION_S
                )

                desired_hand_pos_b = (
                    initial_hand_pos_b
                    * (
                        1.0
                        - alpha
                    )
                    + pregrasp_hand_pos_b
                    * alpha
                )

                desired_hand_quat_b = quat_slerp(
                    initial_hand_quat_b[0],
                    target_hand_quat_b[0],
                    alpha,
                ).unsqueeze(0)


            # =================================================================
            # Automatic final approach
            # =================================================================

            elif time_s < AUTO_DURATION_S:

                phase = "AUTO_APPROACH"

                approach_alpha = smoothstep(
                    (
                        time_s
                        - REACH_DURATION_S
                    )
                    / APPROACH_DURATION_S
                )

                desired_hand_pos_b = (
                    pregrasp_hand_pos_b
                    * (
                        1.0
                        - approach_alpha
                    )
                    + final_hand_pos_b
                    * approach_alpha
                )

                desired_hand_quat_b = (
                    target_hand_quat_b
                )


            # =================================================================
            # Manual calibration
            # =================================================================

            else:

                phase = "MANUAL"

                # -------------------------------------------------------------
                # Capture the ACTUAL pose when manual mode begins.
                # -------------------------------------------------------------

                if not manual_initialized:

                    manual_target_pos_w = (
                        robot.data.body_pos_w[
                            0,
                            hand_base_id,
                        ].clone()
                    )

                    manual_target_quat_w = (
                        robot.data.body_quat_w[
                            0,
                            hand_base_id,
                        ].clone()
                    )

                    manual_reset_pos_w = (
                        manual_target_pos_w.clone()
                    )

                    manual_reset_quat_w = (
                        manual_target_quat_w.clone()
                    )

                    keyboard.reset()

                    manual_initialized = True

                    print()
                    print(
                        "============================================================"
                    )

                    print(
                        " MANUAL CONTROL READY"
                    )

                    print(
                        "============================================================"
                    )

                    print()
                    print(
                        "Click once inside the Isaac Sim 3D viewport."
                    )

                    print()
                    print(
                        "Do NOT hold the right mouse button while tuning."
                    )

                    print()
                    print(
                        "W / S : toward / away from handle"
                    )

                    print(
                        "A / D : along handle"
                    )

                    print(
                        "Q / E : up / down"
                    )

                    print()

                    print(
                        "Z / X : hand roll"
                    )

                    print(
                        "T / G : hand pitch"
                    )

                    print(
                        "C / V : hand yaw"
                    )

                    print()

                    print(
                        "P     : PRINT exact pose"
                    )

                    print(
                        "R     : reset manual pose"
                    )

                    print()
                    print(
                        "============================================================"
                    )
                    print()


                # -------------------------------------------------------------
                # Handle callback requests
                # -------------------------------------------------------------

                if flags["reset"]:

                    manual_target_pos_w = (
                        manual_reset_pos_w.clone()
                    )

                    manual_target_quat_w = (
                        manual_reset_quat_w.clone()
                    )

                    keyboard.reset()

                    flags["reset"] = False

                    print()
                    print(
                        "[MANUAL] Reset to starting manual pose."
                    )
                    print()


                # -------------------------------------------------------------
                # Read keyboard
                #
                # Output:
                #
                # [x, y, z, rx, ry, rz]
                #
                # while keys are held.
                # -------------------------------------------------------------

                keyboard_command = (
                    keyboard.advance()
                )

                translation_input = (
                    keyboard_command[
                        0:3
                    ]
                )

                rotation_input = (
                    keyboard_command[
                        3:6
                    ]
                )


                # -------------------------------------------------------------
                # Translation mapping
                #
                # W/S = approach axis
                #
                # A/D = along 45-degree handle
                #
                # Q/E = world vertical
                # -------------------------------------------------------------

                world_up = torch.tensor(
                    [
                        0.0,
                        0.0,
                        1.0,
                    ],
                    device=robot.device,
                    dtype=torch.float32,
                )

                manual_velocity_w = (
                    translation_input[0]
                    * approach_in_w
                    + translation_input[1]
                    * handle_axis_w
                    + translation_input[2]
                    * world_up
                )

                manual_target_pos_w = (
                    manual_target_pos_w
                    + manual_velocity_w
                    * sim_dt
                )


                # -------------------------------------------------------------
                # Rotation mapping
                #
                # Applied in HAND-LOCAL XYZ axes.
                # -------------------------------------------------------------

                roll_step = (
                    rotation_input[0]
                    * sim_dt
                )

                pitch_step = (
                    rotation_input[1]
                    * sim_dt
                )

                yaw_step = (
                    rotation_input[2]
                    * sim_dt
                )

                if (
                    abs(
                        roll_step.item()
                    ) > 1.0e-12
                    or abs(
                        pitch_step.item()
                    ) > 1.0e-12
                    or abs(
                        yaw_step.item()
                    ) > 1.0e-12
                ):

                    delta_quat = quat_from_euler_xyz(
                        roll_step.reshape(1),
                        pitch_step.reshape(1),
                        yaw_step.reshape(1),
                    )[0]

                    manual_target_quat_w = quat_mul(
                        manual_target_quat_w.unsqueeze(0),
                        delta_quat.unsqueeze(0),
                    )[0]

                    manual_target_quat_w = (
                        manual_target_quat_w
                        / torch.linalg.vector_norm(
                            manual_target_quat_w
                        )
                    )


                # -------------------------------------------------------------
                # Convert manual WORLD target into G1 root frame
                # -------------------------------------------------------------

                (
                    desired_hand_pos_b,
                    desired_hand_quat_b,
                ) = subtract_frame_transforms(
                    root_pose_w_now[:, 0:3],
                    root_pose_w_now[:, 3:7],
                    manual_target_pos_w.unsqueeze(0),
                    manual_target_quat_w.unsqueeze(0),
                )


            # =================================================================
            # IK command
            # =================================================================

            ik_command = torch.cat(
                (
                    desired_hand_pos_b,
                    desired_hand_quat_b,
                ),
                dim=-1,
            )

            ik_controller.set_command(
                ik_command
            )


            # =================================================================
            # Jacobian
            # =================================================================

            jacobian = (
                robot.root_physx_view.get_jacobians()[
                    :,
                    hand_jacobian_index,
                    :,
                    arm_joint_ids,
                ]
            )


            # =================================================================
            # Current joint positions
            # =================================================================

            current_arm_joint_pos = (
                robot.data.joint_pos[
                    :,
                    arm_joint_ids,
                ]
            )


            # =================================================================
            # IK solve
            # =================================================================

            raw_arm_target = (
                ik_controller.compute(
                    hand_pos_b,
                    hand_quat_b,
                    jacobian,
                    current_arm_joint_pos,
                )
            )

            if not torch.isfinite(
                raw_arm_target
            ).all():

                raise RuntimeError(
                    "IK produced NaN or Inf."
                )


            # =================================================================
            # Bounded step
            # =================================================================

            arm_delta = (
                raw_arm_target
                - current_arm_joint_pos
            )

            arm_delta = torch.clamp(
                arm_delta,
                min=-MAX_ARM_JOINT_STEP_RAD,
                max=MAX_ARM_JOINT_STEP_RAD,
            )

            next_arm_joint_pos = (
                current_arm_joint_pos
                + arm_delta
            )


            # =================================================================
            # Joint limits
            # =================================================================

            arm_limits = (
                robot.data.soft_joint_pos_limits[
                    :,
                    arm_joint_ids,
                    :
                ]
            )

            next_arm_joint_pos = torch.maximum(
                next_arm_joint_pos,
                arm_limits[
                    :,
                    :,
                    0
                ],
            )

            next_arm_joint_pos = torch.minimum(
                next_arm_joint_pos,
                arm_limits[
                    :,
                    :,
                    1
                ],
            )


            # =================================================================
            # Extra wrist safety
            # =================================================================

            for arm_index in range(
                4,
                7,
            ):

                joint_id = (
                    arm_joint_ids[
                        arm_index
                    ]
                )

                neutral_value = (
                    neutral_robot_joint_pos[
                        0,
                        joint_id,
                    ]
                )

                wrist_min = (
                    neutral_value
                    - MAX_WRIST_DEVIATION_RAD
                )

                wrist_max = (
                    neutral_value
                    + MAX_WRIST_DEVIATION_RAD
                )

                next_arm_joint_pos[
                    0,
                    arm_index,
                ] = torch.clamp(
                    next_arm_joint_pos[
                        0,
                        arm_index,
                    ],
                    wrist_min,
                    wrist_max,
                )


            # =================================================================
            # Full robot pose
            #
            # Fingers remain OPEN.
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos


            # =================================================================
            # Apply robot
            # =================================================================

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
                zero_robot_joint_vel,
            )


            # =================================================================
            # Keep tripod locked
            # =================================================================

            tripod.write_joint_state_to_sim(
                neutral_tripod_joint_pos,
                zero_tripod_joint_vel,
            )


            robot.write_data_to_sim()

            tripod.write_data_to_sim()

            sim.step()

            robot.update(
                sim_dt
            )

            tripod.update(
                sim_dt
            )


            # =================================================================
            # Print exact calibrated pose
            # =================================================================

            if (
                manual_initialized
                and flags["print"]
            ):

                flags["print"] = False


                # -------------------------------------------------------------
                # Actual hand pose
                # -------------------------------------------------------------

                actual_hand_pos_w = (
                    robot.data.body_pos_w[
                        0,
                        hand_base_id,
                    ].clone()
                )

                actual_hand_quat_w = (
                    robot.data.body_quat_w[
                        0,
                        hand_base_id,
                    ].clone()
                )


                # -------------------------------------------------------------
                # Actual palm centre
                # -------------------------------------------------------------

                index_now = (
                    robot.data.body_pos_w[
                        0,
                        index_tip_id,
                    ]
                )

                middle_now = (
                    robot.data.body_pos_w[
                        0,
                        middle_tip_id,
                    ]
                )

                ring_now = (
                    robot.data.body_pos_w[
                        0,
                        ring_tip_id,
                    ]
                )

                pinky_now = (
                    robot.data.body_pos_w[
                        0,
                        pinky_tip_id,
                    ]
                )

                finger_centroid_now = (
                    index_now
                    + middle_now
                    + ring_now
                    + pinky_now
                ) / 4.0

                palm_center_now = (
                    actual_hand_pos_w
                    + PALM_CENTER_FRACTION
                    * (
                        finger_centroid_now
                        - actual_hand_pos_w
                    )
                )


                # -------------------------------------------------------------
                # Handle distance
                # -------------------------------------------------------------

                handle_pos_now = (
                    tripod.data.body_pos_w[
                        0,
                        handle_id,
                    ]
                )

                segment_start = (
                    handle_pos_now
                    - HANDLE_HALF_LENGTH
                    * raw_handle_axis_w
                )

                segment_end = (
                    handle_pos_now
                    + HANDLE_HALF_LENGTH
                    * raw_handle_axis_w
                )

                palm_handle_distance = (
                    point_to_segment_distance(
                        palm_center_now,
                        segment_start,
                        segment_end,
                    ).item()
                )


                # -------------------------------------------------------------
                # Manual movement relative to starting pose
                # -------------------------------------------------------------

                target_delta_w = (
                    manual_target_pos_w
                    - manual_reset_pos_w
                )

                toward_handle_delta = torch.dot(
                    target_delta_w,
                    approach_in_w,
                ).item()

                along_handle_delta = torch.dot(
                    target_delta_w,
                    handle_axis_w,
                ).item()

                vertical_delta = (
                    target_delta_w[2].item()
                )


                # -------------------------------------------------------------
                # Joint state
                # -------------------------------------------------------------

                arm_joint_pos_now = (
                    robot.data.joint_pos[
                        0,
                        arm_joint_ids,
                    ]
                )


                # -------------------------------------------------------------
                # Print
                # -------------------------------------------------------------

                print()
                print(
                    "============================================================"
                )

                print(
                    " RAVEN MANUAL GRASP CALIBRATION POSE"
                )

                print(
                    "============================================================"
                )

                print()

                print(
                    "[HAND BASE - ACTUAL WORLD]"
                )

                print(
                    f"pos  = "
                    f"{vec_string(actual_hand_pos_w)}"
                )

                print(
                    f"quat = "
                    f"{quat_string(actual_hand_quat_w)}"
                )

                print()

                print(
                    "[IK TARGET - WORLD]"
                )

                print(
                    f"pos  = "
                    f"{vec_string(manual_target_pos_w)}"
                )

                print(
                    f"quat = "
                    f"{quat_string(manual_target_quat_w)}"
                )

                print()

                print(
                    "[PALM CENTRE - WORLD]"
                )

                print(
                    f"pos = "
                    f"{vec_string(palm_center_now)}"
                )

                print(
                    f"distance_to_handle_axis = "
                    f"{palm_handle_distance:.6f} m"
                )

                print()

                print(
                    "[MANUAL OFFSET FROM START]"
                )

                print(
                    f"toward_handle = "
                    f"{toward_handle_delta:+.6f} m"
                )

                print(
                    f"along_handle  = "
                    f"{along_handle_delta:+.6f} m"
                )

                print(
                    f"world_z       = "
                    f"{vertical_delta:+.6f} m"
                )

                print()

                print(
                    "[LEFT ARM JOINTS]"
                )

                for joint_name, joint_value in zip(
                    arm_joint_names,
                    arm_joint_pos_now,
                ):

                    radians_value = (
                        joint_value.item()
                    )

                    degrees_value = math.degrees(
                        radians_value
                    )

                    print(
                        f"{joint_name:30s} "
                        f"= {radians_value:+.8f} rad "
                        f"({degrees_value:+8.3f} deg)"
                    )

                print()

                print(
                    "[HANDLE]"
                )

                print(
                    f"centre      = "
                    f"{vec_string(handle_pos_now)}"
                )

                print(
                    f"grasp_point = "
                    f"{vec_string(grasp_pos_w)}"
                )

                print()

                print(
                    "============================================================"
                )

                print(
                    " COPY THIS WHOLE BLOCK BACK TO CHAT"
                )

                print(
                    "============================================================"
                )

                print()


            # =================================================================
            # Counter
            # =================================================================

            step += 1


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()