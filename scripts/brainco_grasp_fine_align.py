"""RAVEN BrainCo Revo2 closed-loop grasp-pocket alignment.

This stage starts from neutral and performs:

    1. 7-DOF handshake pre-grasp
    2. coarse approach toward the 45-degree tripod handle
    3. closed-loop fine alignment of the ACTUAL open-hand aperture
    4. hold when the shaft is correctly positioned between thumb/fingers

Important:
    - fingers remain OPEN
    - tripod remains locked
    - no grasp closure yet
    - no RL

The grasp aperture is measured every frame from the real Revo2 geometry.

This removes the remaining ~4 cm positioning error observed when only
the hand base pose was used as the IK reference.
"""

import argparse
import copy
import math

from isaaclab.app import AppLauncher


# =============================================================================
# Launch
# =============================================================================

parser = argparse.ArgumentParser(
    description="RAVEN BrainCo closed-loop grasp aperture alignment."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


# =============================================================================
# Isaac imports
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

PREGRASP_DURATION_S = 8.0
COARSE_APPROACH_DURATION_S = 5.0

HANDLE_HALF_LENGTH = 0.16
HANDLE_RADIUS = 0.018

GRASP_ALONG_HANDLE_X = -0.10

# Coarse approach target.
PREGRASP_APERTURE_STANDOFF = 0.10
COARSE_APERTURE_STANDOFF = 0.045

# Final desired distance of the EMPTY grasp-pocket centre from the
# handle centre-line.
#
# The handle radius is 18 mm, but this point is virtual empty space,
# so it is allowed to lie inside the cylinder cross-section.
FINAL_APERTURE_AXIS_DISTANCE = 0.005

# Fine servo translation speed.
#
# 0.25 mm per physics frame at 200 Hz is a maximum of ~5 cm/s.
MAX_APERTURE_CORRECTION_PER_STEP = 0.00025

DLS_DAMPING = 0.15

MAX_ARM_JOINT_STEP_RAD = math.radians(
    0.10
)

# Stop if physical fingertip geometry penetrates the shaft significantly.
MAX_FINGER_PENETRATION_M = 0.003

# Final alignment tolerances.
APERTURE_TOLERANCE_M = 0.007
AXIS_ALIGNMENT_TOLERANCE_DEG = 3.0
CLOSING_ALIGNMENT_TOLERANCE_DEG = 3.0

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

    body_id = int(body_ids[0])

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


def vector_angle_deg(
    a,
    b,
    directionless=False,
) -> float:

    a = normalize_vector(
        a
    )

    b = normalize_vector(
        b
    )

    dot = torch.dot(
        a,
        b,
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
            point - segment_a
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
        point - closest
    )


def clamp_vector_length(
    vector,
    maximum_length: float,
):

    length = torch.linalg.vector_norm(
        vector
    )

    if length.item() <= maximum_length:
        return vector

    return (
        vector
        / length
        * maximum_length
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
    # G1 + Revo2
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
    # 7-DOF left arm
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


    # =========================================================================
    # BrainCo bodies
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
    # Tripod bodies/joints
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
    # Jacobian body
    # =========================================================================

    if not robot.is_fixed_base:
        raise RuntimeError(
            "This test requires a fixed G1 root."
        )

    hand_jacobian_index = (
        hand_base_id - 1
    )


    # =========================================================================
    # Initial Revo2 geometry
    # =========================================================================

    initial_hand_base_pos_w = (
        robot.data.body_pos_w[
            0,
            hand_base_id,
        ].clone()
    )

    initial_hand_base_quat_w = (
        robot.data.body_quat_w[
            0,
            hand_base_id,
        ].clone()
    )

    thumb_initial = (
        robot.data.body_pos_w[
            0,
            thumb_tip_id,
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

    aperture_initial = (
        thumb_initial
        + finger_centroid_initial
    ) * 0.5


    # =========================================================================
    # Initial semantic grasp frame
    # =========================================================================

    current_across_w = normalize_vector(
        index_initial
        - pinky_initial
    )

    current_closing_raw_w = (
        finger_centroid_initial
        - thumb_initial
    )

    current_closing_w = (
        current_closing_raw_w
        - torch.dot(
            current_closing_raw_w,
            current_across_w,
        )
        * current_across_w
    )

    current_closing_w = normalize_vector(
        current_closing_w
    )

    current_normal_w = normalize_vector(
        torch.cross(
            current_across_w,
            current_closing_w,
            dim=0,
        )
    )

    current_closing_w = normalize_vector(
        torch.cross(
            current_normal_w,
            current_across_w,
            dim=0,
        )
    )

    current_grasp_basis_w = torch.stack(
        (
            current_across_w,
            current_closing_w,
            current_normal_w,
        ),
        dim=1,
    )


    # =========================================================================
    # Aperture offset in hand-local coordinates
    # =========================================================================

    initial_hand_rot_w = matrix_from_quat(
        initial_hand_base_quat_w.unsqueeze(0)
    )[0]

    aperture_offset_hand = (
        initial_hand_rot_w.transpose(
            0,
            1,
        )
        @ (
            aperture_initial
            - initial_hand_base_pos_w
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
    # Choose handle-axis sign closest to current hand
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
    # Approach direction derived from actual hand position
    # =========================================================================

    from_handle_to_hand_w = (
        aperture_initial
        - grasp_pos_w
    )

    approach_out_w = (
        from_handle_to_hand_w
        - torch.dot(
            from_handle_to_hand_w,
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
    # Desired semantic hand orientation
    # =========================================================================

    desired_across_w = (
        handle_axis_w
    )

    desired_closing_w = (
        approach_in_w
    )

    desired_normal_w = normalize_vector(
        torch.cross(
            desired_across_w,
            desired_closing_w,
            dim=0,
        )
    )

    desired_closing_w = normalize_vector(
        torch.cross(
            desired_normal_w,
            desired_across_w,
            dim=0,
        )
    )

    desired_grasp_basis_w = torch.stack(
        (
            desired_across_w,
            desired_closing_w,
            desired_normal_w,
        ),
        dim=1,
    )

    semantic_rotation = (
        desired_grasp_basis_w
        @ current_grasp_basis_w.transpose(
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
    # Aperture targets
    # =========================================================================

    pregrasp_aperture_w = (
        grasp_pos_w
        + PREGRASP_APERTURE_STANDOFF
        * approach_out_w
    )

    coarse_aperture_w = (
        grasp_pos_w
        + COARSE_APERTURE_STANDOFF
        * approach_out_w
    )

    final_aperture_w = (
        grasp_pos_w
        + FINAL_APERTURE_AXIS_DISTANCE
        * approach_out_w
    )


    # =========================================================================
    # Base targets corresponding to pregrasp/coarse aperture positions
    # =========================================================================

    target_aperture_offset_w = (
        target_hand_rot_w
        @ aperture_offset_hand
    )

    pregrasp_hand_base_w = (
        pregrasp_aperture_w
        - target_aperture_offset_w
    )

    coarse_hand_base_w = (
        coarse_aperture_w
        - target_aperture_offset_w
    )


    # =========================================================================
    # Convert initial / coarse poses into robot-root frame
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
        initial_hand_base_pos_w.unsqueeze(0),
        initial_hand_base_quat_w.unsqueeze(0),
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
        coarse_hand_pos_b,
        _,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        coarse_hand_base_w.unsqueeze(0),
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
        " RAVEN CLOSED-LOOP GRASP POCKET ALIGNMENT"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"[GRASP POINT]       "
        f"{vec_string(grasp_pos_w)}"
    )

    print(
        f"[PREGRASP APERTURE] "
        f"{vec_string(pregrasp_aperture_w)}"
    )

    print(
        f"[COARSE APERTURE]   "
        f"{vec_string(coarse_aperture_w)}"
    )

    print(
        f"[FINAL APERTURE]    "
        f"{vec_string(final_aperture_w)}"
    )

    print()

    print(
        "[RAVEN] Final phase servos the ACTUAL "
        "thumb/finger aperture, not the wrist/base."
    )

    print()


    # =========================================================================
    # Runtime
    # =========================================================================

    step = 0

    aligned = False
    safety_stop = False


    while simulation_app.is_running():

        with torch.inference_mode():

            time_s = (
                step * sim_dt
            )


            # =================================================================
            # Actual hand geometry
            # =================================================================

            hand_base_pos_now = (
                robot.data.body_pos_w[
                    0,
                    hand_base_id,
                ].clone()
            )

            hand_base_quat_now = (
                robot.data.body_quat_w[
                    0,
                    hand_base_id,
                ].clone()
            )

            thumb_now = (
                robot.data.body_pos_w[
                    0,
                    thumb_tip_id,
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

            aperture_now = (
                thumb_now
                + finger_centroid_now
            ) * 0.5


            # =================================================================
            # Current semantic axes
            # =================================================================

            across_now = normalize_vector(
                index_now
                - pinky_now
            )

            closing_raw_now = (
                finger_centroid_now
                - thumb_now
            )

            closing_now = (
                closing_raw_now
                - torch.dot(
                    closing_raw_now,
                    across_now,
                )
                * across_now
            )

            closing_now = normalize_vector(
                closing_now
            )


            # =================================================================
            # Current root/hand frame
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
            # Determine phase
            # =================================================================

            fine_start_time = (
                PREGRASP_DURATION_S
                + COARSE_APPROACH_DURATION_S
            )

            if time_s < PREGRASP_DURATION_S:

                phase = "PREGRASP"

                alpha = smoothstep(
                    time_s
                    / PREGRASP_DURATION_S
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

            elif time_s < fine_start_time:

                phase = "COARSE_APPROACH"

                alpha = smoothstep(
                    (
                        time_s
                        - PREGRASP_DURATION_S
                    )
                    / COARSE_APPROACH_DURATION_S
                )

                desired_hand_pos_b = (
                    pregrasp_hand_pos_b
                    * (
                        1.0
                        - alpha
                    )
                    + coarse_hand_pos_b
                    * alpha
                )

                desired_hand_quat_b = (
                    target_hand_quat_b
                )

            else:

                phase = (
                    "ALIGNED"
                    if aligned
                    else "FINE_ALIGN"
                )

                # -------------------------------------------------------------
                # Closed-loop aperture correction
                # -------------------------------------------------------------

                aperture_error_w = (
                    final_aperture_w
                    - aperture_now
                )

                aperture_error_step_w = clamp_vector_length(
                    aperture_error_w,
                    MAX_APERTURE_CORRECTION_PER_STEP,
                )

                desired_base_pos_w = (
                    hand_base_pos_now
                    + aperture_error_step_w
                )

                (
                    desired_hand_pos_b,
                    desired_hand_quat_b,
                ) = subtract_frame_transforms(
                    root_pose_w_now[:, 0:3],
                    root_pose_w_now[:, 3:7],
                    desired_base_pos_w.unsqueeze(0),
                    target_hand_quat_w.unsqueeze(0),
                )


            # =================================================================
            # IK solve
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

            raw_target = (
                ik_controller.compute(
                    hand_pos_b,
                    hand_quat_b,
                    jacobian,
                    current_arm_joint_pos,
                )
            )

            if not torch.isfinite(
                raw_target
            ).all():

                raise RuntimeError(
                    "IK produced NaN or Inf."
                )

            delta = (
                raw_target
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

            limits = (
                robot.data.soft_joint_pos_limits[
                    :,
                    arm_joint_ids,
                    :
                ]
            )

            next_arm_joint_pos = torch.maximum(
                next_arm_joint_pos,
                limits[
                    :,
                    :,
                    0
                ],
            )

            next_arm_joint_pos = torch.minimum(
                next_arm_joint_pos,
                limits[
                    :,
                    :,
                    1
                ],
            )


            # =================================================================
            # Full G1 state
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
                zero_robot_joint_vel,
            )


            # =========================================================================
            # Keep tripod locked
            # =========================================================================

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
            # Measurements after step
            # =================================================================

            thumb_after = (
                robot.data.body_pos_w[
                    0,
                    thumb_tip_id,
                ]
            )

            index_after = (
                robot.data.body_pos_w[
                    0,
                    index_tip_id,
                ]
            )

            middle_after = (
                robot.data.body_pos_w[
                    0,
                    middle_tip_id,
                ]
            )

            ring_after = (
                robot.data.body_pos_w[
                    0,
                    ring_tip_id,
                ]
            )

            pinky_after = (
                robot.data.body_pos_w[
                    0,
                    pinky_tip_id,
                ]
            )

            finger_centroid_after = (
                index_after
                + middle_after
                + ring_after
                + pinky_after
            ) / 4.0

            aperture_after = (
                thumb_after
                + finger_centroid_after
            ) * 0.5


            # =================================================================
            # Handle segment
            # =================================================================

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


            # =================================================================
            # Aperture distance
            # =================================================================

            aperture_axis_distance = (
                point_to_segment_distance(
                    aperture_after,
                    segment_start,
                    segment_end,
                ).item()
            )


            # =================================================================
            # Finger clearance
            # =================================================================

            minimum_finger_clearance = float(
                "inf"
            )

            for fingertip_id in fingertip_ids:

                fingertip_pos = (
                    robot.data.body_pos_w[
                        0,
                        fingertip_id,
                    ]
                )

                distance_to_axis = (
                    point_to_segment_distance(
                        fingertip_pos,
                        segment_start,
                        segment_end,
                    ).item()
                )

                clearance = (
                    distance_to_axis
                    - HANDLE_RADIUS
                )

                minimum_finger_clearance = min(
                    minimum_finger_clearance,
                    clearance,
                )


            # =================================================================
            # Alignment angles
            # =================================================================

            across_after = normalize_vector(
                index_after
                - pinky_after
            )

            closing_raw_after = (
                finger_centroid_after
                - thumb_after
            )

            closing_after = (
                closing_raw_after
                - torch.dot(
                    closing_raw_after,
                    across_after,
                )
                * across_after
            )

            closing_after = normalize_vector(
                closing_after
            )

            across_error_deg = vector_angle_deg(
                across_after,
                handle_axis_w,
                directionless=True,
            )

            closing_error_deg = vector_angle_deg(
                closing_after,
                desired_closing_w,
            )


            # =================================================================
            # Is the shaft bracketed by thumb and fingers?
            #
            # Project both sides onto the hand's closing direction around the
            # chosen grasp point.
            # =================================================================

            thumb_projection = torch.dot(
                thumb_after
                - grasp_pos_w,
                desired_closing_w,
            ).item()

            finger_projection = torch.dot(
                finger_centroid_after
                - grasp_pos_w,
                desired_closing_w,
            ).item()

            shaft_bracketed = (
                thumb_projection
                * finger_projection
                <= 0.0
            )


            # =================================================================
            # Safety / success
            # =================================================================

            if (
                minimum_finger_clearance
                < -MAX_FINGER_PENETRATION_M
            ):

                safety_stop = True


            if (
                time_s >= fine_start_time
                and aperture_axis_distance
                <= APERTURE_TOLERANCE_M
                and across_error_deg
                <= AXIS_ALIGNMENT_TOLERANCE_DEG
                and closing_error_deg
                <= CLOSING_ALIGNMENT_TOLERANCE_DEG
                and shaft_bracketed
            ):

                aligned = True


            # =================================================================
            # Tripod
            # =================================================================

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


            # =================================================================
            # Telemetry
            # =================================================================

            if (
                step
                % PRINT_EVERY_STEPS
                == 0
            ):

                print(
                    f"[{phase}] "
                    f"time={time_s:5.2f}s"
                )

                print(
                    "[APERTURE] "
                    f"pos={vec_string(aperture_after)} | "
                    f"axis_distance="
                    f"{aperture_axis_distance:.4f} m"
                )

                print(
                    "[ALIGNMENT] "
                    f"handle/across="
                    f"{across_error_deg:.2f} deg | "
                    f"closing="
                    f"{closing_error_deg:.2f} deg"
                )

                print(
                    "[GRASP POCKET] "
                    f"thumb_side="
                    f"{thumb_projection:+.4f} | "
                    f"finger_side="
                    f"{finger_projection:+.4f} | "
                    f"shaft_bracketed="
                    f"{shaft_bracketed}"
                )

                print(
                    "[FINGERS] "
                    f"minimum_clearance="
                    f"{minimum_finger_clearance:+.4f} m"
                )

                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+7.3f} deg | "
                    f"tilt={tilt_deg:+7.3f} deg | "
                    f"camera={vec_string(camera_pos)}"
                )

                if aligned:

                    print(
                        "[RAVEN] GRASP POCKET ALIGNED."
                    )

                    print(
                        "[RAVEN] Handle is positioned between "
                        "thumb and fingers."
                    )

                if safety_stop:

                    print(
                        "[RAVEN] WARNING: fingertip penetration "
                        "exceeded safety threshold."
                    )

                print()


            step += 1


# =============================================================================
# Entry
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()