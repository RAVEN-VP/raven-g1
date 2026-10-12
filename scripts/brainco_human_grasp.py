"""RAVEN natural human-style BrainCo Revo2 tripod grasp.

Motion:
    1. Reach toward the handle with a straight, handshake-style hand.
    2. Place the PALM beside the 45-degree handle.
    3. Move the palm straight toward the shaft.
    4. Close the Revo2 fingers around the handle.

The handle axis lies ACROSS the palm.

The fingers point toward the tripod/handle during the approach.

This deliberately avoids the previous thumb-to-finger "aperture frame"
that caused the back of the hand to face the shaft.

The tripod remains locked during this validation.
"""

# =============================================================================
# Standard library
# =============================================================================

import argparse
import copy
import math


# =============================================================================
# Isaac launcher
# =============================================================================

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="RAVEN natural BrainCo Revo2 handle grasp."
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

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_from_matrix,
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
# Timing
# -----------------------------------------------------------------------------

REACH_DURATION_S = 8.0

APPROACH_DURATION_S = 4.0

GRASP_DURATION_S = 4.0


# -----------------------------------------------------------------------------
# Handle geometry
# -----------------------------------------------------------------------------

HANDLE_HALF_LENGTH = 0.16

HANDLE_RADIUS = 0.018


# 10 cm toward the operator from the handle centre.
GRASP_ALONG_HANDLE_X = -0.10


# -----------------------------------------------------------------------------
# Palm positioning
#
# These distances are from the HANDLE CENTRE-LINE to the PALM CENTRE.
#
# We first stop 10 cm away.
#
# We then approach to ~4 cm.
#
# The handle radius is 1.8 cm, leaving the hand surface close to the shaft
# without trying to put the wrist itself through the handle.
# -----------------------------------------------------------------------------

PREGRASP_PALM_DISTANCE = 0.10

FINAL_PALM_DISTANCE = 0.040


# -----------------------------------------------------------------------------
# Approximate palm centre along:
#
#     hand base -> four-finger centroid
#
# 0.0 = wrist/base
# 1.0 = fingertips
#
# ~0.42 puts the reference roughly in the central palm.
# -----------------------------------------------------------------------------

PALM_CENTER_FRACTION = 0.42


# -----------------------------------------------------------------------------
# IK
# -----------------------------------------------------------------------------

DLS_DAMPING = 0.15

MAX_ARM_JOINT_STEP_RAD = math.radians(
    0.10
)


# -----------------------------------------------------------------------------
# Keep the wrist relatively straight.
#
# We still allow all three wrist DOFs to participate in IK, but they cannot
# bend more than +/-35 degrees from their neutral pose.
# -----------------------------------------------------------------------------

MAX_WRIST_DEVIATION_RAD = math.radians(
    35.0
)


# -----------------------------------------------------------------------------
# Revo2 closure
#
# Moderate cylindrical grasp.
# -----------------------------------------------------------------------------

THUMB_META_TARGET = 0.45

THUMB_PROX_TARGET = 0.55

INDEX_PROX_TARGET = 0.60

MIDDLE_PROX_TARGET = 0.60

RING_PROX_TARGET = 0.60

PINKY_PROX_TARGET = 0.60


# Distal mimic relationships from the combined model.
FINGER_DISTAL_MULTIPLIER = 1.155

THUMB_DISTAL_MULTIPLIER = 1.0


PRINT_EVERY_STEPS = 50


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
        f"{vector[0].item(): .3f}, "
        f"{vector[1].item(): .3f}, "
        f"{vector[2].item(): .3f}"
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


def angle_between_deg(
    vector_a,
    vector_b,
    directionless=False,
) -> float:

    vector_a = normalize_vector(
        vector_a
    )

    vector_b = normalize_vector(
        vector_b
    )

    dot = torch.dot(
        vector_a,
        vector_b,
    )

    if directionless:

        dot = torch.abs(
            dot
        )

    dot = torch.clamp(
        dot,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(
            dot.item()
        )
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
    # Lighting
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
    # Robot
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
    # Seven left-arm joints
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
    # BrainCo hand joints
    # =========================================================================

    hand_joint_ids = {
        "thumb_meta": resolve_joint(
            robot,
            "left_thumb_metacarpal_joint",
        ),

        "thumb_prox": resolve_joint(
            robot,
            "left_thumb_proximal_joint",
        ),

        "thumb_dist": resolve_joint(
            robot,
            "left_thumb_distal_joint",
        ),

        "index_prox": resolve_joint(
            robot,
            "left_index_proximal_joint",
        ),

        "index_dist": resolve_joint(
            robot,
            "left_index_distal_joint",
        ),

        "middle_prox": resolve_joint(
            robot,
            "left_middle_proximal_joint",
        ),

        "middle_dist": resolve_joint(
            robot,
            "left_middle_distal_joint",
        ),

        "ring_prox": resolve_joint(
            robot,
            "left_ring_proximal_joint",
        ),

        "ring_dist": resolve_joint(
            robot,
            "left_ring_distal_joint",
        ),

        "pinky_prox": resolve_joint(
            robot,
            "left_pinky_proximal_joint",
        ),

        "pinky_dist": resolve_joint(
            robot,
            "left_pinky_distal_joint",
        ),
    }


    # =========================================================================
    # Hand bodies
    # =========================================================================

    hand_base_id = resolve_body(
        robot,
        "left_base_link",
    )

    thumb_tip_id = resolve_body(
        robot,
        "left_thumb_tip_Link",
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

    fingertip_ids = [
        thumb_tip_id,
        index_tip_id,
        middle_tip_id,
        ring_tip_id,
        pinky_tip_id,
    ]


    # =========================================================================
    # Tripod
    # =========================================================================

    handle_id = resolve_body(
        tripod,
        "control_handle_link",
    )

    camera_id = resolve_body(
        tripod,
        "camera_link",
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
            "This validation requires a fixed G1 base."
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
    #
    # Y = wrist/palm -> fingertips
    #
    # Z = normal to palm
    #
    # THIS is the important correction.
    #
    # We are no longer using thumb -> fingers as an orientation axis.
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
    #
    # Approximate centre of the physical palm.
    # =========================================================================

    palm_center_initial_w = (
        initial_hand_pos_w
        + PALM_CENTER_FRACTION
        * (
            finger_centroid_initial
            - initial_hand_pos_w
        )
    )


    # =========================================================================
    # Palm centre expressed in hand-local coordinates
    # =========================================================================

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
    # Handle
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


    # =========================================================================
    # Physical grasp position on the shaft
    # =========================================================================

    grasp_pos_w = (
        handle_pos_w
        + raw_handle_axis_w
        * GRASP_ALONG_HANDLE_X
    )


    # =========================================================================
    # Choose handle direction closest to the current hand's across-palm axis.
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
    # Natural approach direction
    #
    # From the chosen handle point toward the robot's CURRENT palm.
    #
    # Remove the along-handle component.
    #
    # This makes the hand approach the SIDE of the shaft.
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
    # Desired HUMAN-LIKE handshake orientation
    #
    # Across palm:
    #     follows handle
    #
    # Fingers:
    #     point toward handle
    #
    # Palm:
    #     sits beside shaft
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


    # =========================================================================
    # Desired hand rotation
    # =========================================================================

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
    # Desired palm positions
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


    # =========================================================================
    # Convert palm target -> hand base target
    # =========================================================================

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
    # Convert poses into G1 root frame
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
    # Startup
    # =========================================================================

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN NATURAL HUMAN-STYLE REVO2 GRASP"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"[HANDLE] "
        f"{vec_string(handle_pos_w)}"
    )

    print(
        f"[GRASP POINT] "
        f"{vec_string(grasp_pos_w)}"
    )

    print()

    print(
        f"[PREGRASP PALM] "
        f"{vec_string(pregrasp_palm_w)}"
    )

    print(
        f"[FINAL PALM]    "
        f"{vec_string(final_palm_w)}"
    )

    print()

    print(
        "[HAND]"
    )

    print(
        "    fingers point toward handle"
    )

    print(
        "    handle lies across palm"
    )

    print(
        "    palm approaches side of shaft"
    )

    print(
        "    wrist limited to +/-35 degrees"
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
            # Current hand pose
            # =================================================================

            root_pose_now = (
                robot.data.root_state_w[
                    :,
                    0:7,
                ]
            )


            hand_pose_now = (
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
                root_pose_now[:, 0:3],
                root_pose_now[:, 3:7],
                hand_pose_now[:, 0:3],
                hand_pose_now[:, 3:7],
            )


            # =================================================================
            # Phase 1: natural handshake reach
            # =================================================================

            if time_s < REACH_DURATION_S:

                phase = "REACH"

                reach_alpha = smoothstep(
                    time_s
                    / REACH_DURATION_S
                )


                desired_hand_pos_b = (
                    initial_hand_pos_b
                    * (
                        1.0
                        - reach_alpha
                    )
                    + pregrasp_hand_pos_b
                    * reach_alpha
                )


                desired_hand_quat_b = quat_slerp(
                    initial_hand_quat_b[0],
                    target_hand_quat_b[0],
                    reach_alpha,
                ).unsqueeze(0)


                closure_alpha = 0.0


            # =================================================================
            # Phase 2: straight palm approach
            # =================================================================

            elif (
                time_s
                < REACH_DURATION_S
                + APPROACH_DURATION_S
            ):

                phase = "APPROACH"


                approach_time = (
                    time_s
                    - REACH_DURATION_S
                )


                approach_alpha = smoothstep(
                    approach_time
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


                closure_alpha = 0.0


            # =================================================================
            # Phase 3: hold arm, close fingers
            # =================================================================

            else:

                phase = "GRASP"


                desired_hand_pos_b = (
                    final_hand_pos_b
                )


                desired_hand_quat_b = (
                    target_hand_quat_b
                )


                grasp_time = (
                    time_s
                    - REACH_DURATION_S
                    - APPROACH_DURATION_S
                )


                closure_alpha = smoothstep(
                    grasp_time
                    / GRASP_DURATION_S
                )


            # =================================================================
            # Arm IK
            # =================================================================

            command = torch.cat(
                (
                    desired_hand_pos_b,
                    desired_hand_quat_b,
                ),
                dim=-1,
            )


            ik_controller.set_command(
                command
            )


            jacobian = (
                robot.root_physx_view.get_jacobians()[
                    :,
                    hand_jacobian_index,
                    :,
                    arm_joint_ids,
                ]
            )


            current_arm_joint_pos = (
                robot.data.joint_pos[
                    :,
                    arm_joint_ids,
                ]
            )


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


            delta = (
                raw_arm_target
                - current_arm_joint_pos
            )


            delta = torch.clamp(
                delta,
                min=-MAX_ARM_JOINT_STEP_RAD,
                max=MAX_ARM_JOINT_STEP_RAD,
            )


            next_arm_joint_pos = (
                current_arm_joint_pos
                + delta
            )


            # =================================================================
            # Normal G1 soft limits
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
            # Additional wrist-straight constraint
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
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )


            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos


            # =================================================================
            # Revo2 finger closure
            # =================================================================

            thumb_meta = (
                THUMB_META_TARGET
                * closure_alpha
            )


            thumb_prox = (
                THUMB_PROX_TARGET
                * closure_alpha
            )


            index_prox = (
                INDEX_PROX_TARGET
                * closure_alpha
            )


            middle_prox = (
                MIDDLE_PROX_TARGET
                * closure_alpha
            )


            ring_prox = (
                RING_PROX_TARGET
                * closure_alpha
            )


            pinky_prox = (
                PINKY_PROX_TARGET
                * closure_alpha
            )


            next_robot_joint_pos[
                0,
                hand_joint_ids["thumb_meta"],
            ] = thumb_meta


            next_robot_joint_pos[
                0,
                hand_joint_ids["thumb_prox"],
            ] = thumb_prox


            next_robot_joint_pos[
                0,
                hand_joint_ids["thumb_dist"],
            ] = (
                thumb_prox
                * THUMB_DISTAL_MULTIPLIER
            )


            next_robot_joint_pos[
                0,
                hand_joint_ids["index_prox"],
            ] = index_prox


            next_robot_joint_pos[
                0,
                hand_joint_ids["index_dist"],
            ] = (
                index_prox
                * FINGER_DISTAL_MULTIPLIER
            )


            next_robot_joint_pos[
                0,
                hand_joint_ids["middle_prox"],
            ] = middle_prox


            next_robot_joint_pos[
                0,
                hand_joint_ids["middle_dist"],
            ] = (
                middle_prox
                * FINGER_DISTAL_MULTIPLIER
            )


            next_robot_joint_pos[
                0,
                hand_joint_ids["ring_prox"],
            ] = ring_prox


            next_robot_joint_pos[
                0,
                hand_joint_ids["ring_dist"],
            ] = (
                ring_prox
                * FINGER_DISTAL_MULTIPLIER
            )


            next_robot_joint_pos[
                0,
                hand_joint_ids["pinky_prox"],
            ] = pinky_prox


            next_robot_joint_pos[
                0,
                hand_joint_ids["pinky_dist"],
            ] = (
                pinky_prox
                * FINGER_DISTAL_MULTIPLIER
            )


            # =================================================================
            # Apply
            # =================================================================

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
                zero_robot_joint_vel,
            )


            # Keep tripod locked for now.
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
            # Telemetry
            # =================================================================

            if (
                step
                % PRINT_EVERY_STEPS
                == 0
            ):

                # -------------------------------------------------------------
                # Current hand geometry
                # -------------------------------------------------------------

                hand_pos_now = (
                    robot.data.body_pos_w[
                        0,
                        hand_base_id,
                    ]
                )


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
                    hand_pos_now
                    + PALM_CENTER_FRACTION
                    * (
                        finger_centroid_now
                        - hand_pos_now
                    )
                )


                across_now = normalize_vector(
                    index_now
                    - pinky_now
                )


                forward_now_raw = (
                    finger_centroid_now
                    - hand_pos_now
                )


                forward_now = (
                    forward_now_raw
                    - torch.dot(
                        forward_now_raw,
                        across_now,
                    )
                    * across_now
                )


                forward_now = normalize_vector(
                    forward_now
                )


                # -------------------------------------------------------------
                # Handle segment
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


                palm_axis_distance = (
                    point_to_segment_distance(
                        palm_center_now,
                        segment_start,
                        segment_end,
                    ).item()
                )


                # -------------------------------------------------------------
                # Orientation diagnostics
                # -------------------------------------------------------------

                across_error = angle_between_deg(
                    across_now,
                    handle_axis_w,
                    directionless=True,
                )


                forward_error = angle_between_deg(
                    forward_now,
                    desired_forward_w,
                    directionless=False,
                )


                # -------------------------------------------------------------
                # Wrist
                # -------------------------------------------------------------

                wrist_values = [
                    math.degrees(
                        robot.data.joint_pos[
                            0,
                            joint_id,
                        ].item()
                    )
                    for joint_id
                    in wrist_joint_ids
                ]


                # -------------------------------------------------------------
                # Tripod
                # -------------------------------------------------------------

                pan_deg = math.degrees(
                    tripod.data.joint_pos[
                        0,
                        pan_joint_id,
                    ].item()
                )


                tilt_deg = math.degrees(
                    tripod.data.joint_pos[
                        0,
                        tilt_joint_id,
                    ].item()
                )


                camera_pos = (
                    tripod.data.body_pos_w[
                        0,
                        camera_id,
                    ]
                )


                # -------------------------------------------------------------
                # Print
                # -------------------------------------------------------------

                print(
                    f"[{phase}] "
                    f"time={time_s:5.2f}s"
                )


                print(
                    "[PALM] "
                    f"pos={vec_string(palm_center_now)} | "
                    f"handle_distance="
                    f"{palm_axis_distance:.4f} m"
                )


                print(
                    "[ORIENTATION] "
                    f"handle/across="
                    f"{across_error:.2f} deg | "
                    f"finger_direction="
                    f"{forward_error:.2f} deg"
                )


                print(
                    "[WRIST] "
                    f"roll={wrist_values[0]:+6.2f} | "
                    f"pitch={wrist_values[1]:+6.2f} | "
                    f"yaw={wrist_values[2]:+6.2f} deg"
                )


                print(
                    "[HAND] "
                    f"closure={closure_alpha:.3f}"
                )


                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+7.3f} | "
                    f"tilt={tilt_deg:+7.3f} deg | "
                    f"camera={vec_string(camera_pos)}"
                )


                print()


            step += 1


# =============================================================================
# Entry
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()