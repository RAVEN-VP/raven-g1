"""RAVEN G1 + BrainCo Revo2 controlled tripod-handle grasp validation.

Sequence
--------
1. Move the full 7-DOF LEFT arm into the validated handshake-style pre-grasp.
2. Move slowly toward the 45-degree tripod handle.
3. Detect when the fingertips reach the handle region.
4. Stop advancing the arm.
5. Close the six commanded Revo2 hand coordinates.
6. Explicitly reproduce the URDF mimic relationships for distal joints.
7. Hold the grasp pose.

The tripod pan and tilt remain locked during this stage.

This is a grasp-pose / contact-geometry validation.
The following stage will introduce proper actuator dynamics and release
the tripod so the G1 can physically pan and tilt the camera.
"""

import argparse
import copy
import math

from isaaclab.app import AppLauncher


# =============================================================================
# Launch
# =============================================================================

parser = argparse.ArgumentParser(
    description="RAVEN BrainCo Revo2 tripod handle grasp validation."
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
    quat_apply,
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

# Stage 1: neutral -> handshake pre-grasp.
PREGRASP_DURATION_S = 10.0

# Stage 2: creep toward the shaft.
APPROACH_DURATION_S = 4.0

# Stage 3: close the hand.
GRASP_DURATION_S = 4.0

# -------------------------------------------------------------------------
# Handle geometry
# -------------------------------------------------------------------------

HANDLE_HALF_LENGTH = 0.16
HANDLE_RADIUS = 0.018

# Point on the shaft, measured along handle-local X.
#
# -0.16 = very rear end.
# -0.10 = 6 cm inward from that end.
GRASP_ALONG_HANDLE_X = -0.10

# Safe handshake pre-grasp distance perpendicular to shaft.
PREGRASP_DISTANCE = 0.12

# Final approach target.
#
# We only move 2 cm farther than the validated pre-grasp.
CONTACT_TARGET_DISTANCE = 0.10

# When fingertip-point clearance becomes this small, stop advancing
# the arm and begin finger closure.
CONTACT_CLEARANCE_M = 0.003

# If our geometric fingertip estimate penetrates too far, freeze closure.
MAX_ALLOWED_PENETRATION_M = 0.005


# =============================================================================
# IK safety
# =============================================================================

DLS_DAMPING = 0.15

MAX_ARM_JOINT_STEP_RAD = math.radians(
    0.10
)

PRINT_EVERY_STEPS = 50


# =============================================================================
# Revo2 grasp target
#
# Moderate cylindrical grasp, deliberately below maximum joint ranges.
#
# These are NOT maximum-fist values.
# =============================================================================

THUMB_METACARPAL_TARGET = 0.65
THUMB_PROXIMAL_TARGET = 0.70

INDEX_PROXIMAL_TARGET = 0.85
MIDDLE_PROXIMAL_TARGET = 0.85
RING_PROXIMAL_TARGET = 0.85
PINKY_PROXIMAL_TARGET = 0.85

# Unitree combined BrainCo URDF mimic relationships.
FINGER_DISTAL_MULTIPLIER = 1.155
THUMB_DISTAL_MULTIPLIER = 1.0


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

    return int(joint_ids[0])


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

    return vector / magnitude


def smoothstep(
    alpha: float,
) -> float:

    alpha = max(
        0.0,
        min(
            1.0,
            alpha,
        ),
    )

    return (
        3.0 * alpha * alpha
        - 2.0 * alpha * alpha * alpha
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
            name,
        )
        for name in arm_joint_names
    ]


    # =========================================================================
    # Revo2 hand joints
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
    # Revo2 bodies
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
    # Tripod bodies / joints
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
    # Fixed-base Jacobian
    # =========================================================================

    if not robot.is_fixed_base:
        raise RuntimeError(
            "This test requires the G1 root to remain fixed."
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

    index_pos_w = (
        robot.data.body_pos_w[
            0,
            index_tip_id,
        ].clone()
    )

    middle_pos_w = (
        robot.data.body_pos_w[
            0,
            middle_tip_id,
        ].clone()
    )

    ring_pos_w = (
        robot.data.body_pos_w[
            0,
            ring_tip_id,
        ].clone()
    )

    pinky_pos_w = (
        robot.data.body_pos_w[
            0,
            pinky_tip_id,
        ].clone()
    )

    fingertip_centroid_w = (
        index_pos_w
        + middle_pos_w
        + ring_pos_w
        + pinky_pos_w
    ) / 4.0


    # =========================================================================
    # Current semantic hand frame
    # =========================================================================

    current_across_w = normalize_vector(
        index_pos_w
        - pinky_pos_w
    )

    current_finger_raw_w = (
        fingertip_centroid_w
        - initial_hand_pos_w
    )

    current_finger_w = (
        current_finger_raw_w
        - torch.dot(
            current_finger_raw_w,
            current_across_w,
        )
        * current_across_w
    )

    current_finger_w = normalize_vector(
        current_finger_w
    )

    current_normal_w = normalize_vector(
        torch.cross(
            current_across_w,
            current_finger_w,
            dim=0,
        )
    )

    current_finger_w = normalize_vector(
        torch.cross(
            current_normal_w,
            current_across_w,
            dim=0,
        )
    )

    current_semantic_basis_w = torch.stack(
        (
            current_across_w,
            current_finger_w,
            current_normal_w,
        ),
        dim=1,
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

    local_handle_x = torch.tensor(
        [
            1.0,
            0.0,
            0.0,
        ],
        device=robot.device,
        dtype=torch.float32,
    )

    handle_axis_w = normalize_vector(
        quat_apply(
            handle_quat_w.unsqueeze(0),
            local_handle_x.unsqueeze(0),
        )[0]
    )


    # =========================================================================
    # Grasp point
    # =========================================================================

    grasp_local = torch.tensor(
        [
            GRASP_ALONG_HANDLE_X,
            0.0,
            0.0,
        ],
        device=robot.device,
        dtype=torch.float32,
    )

    grasp_pos_w = (
        handle_pos_w
        + quat_apply(
            handle_quat_w.unsqueeze(0),
            grasp_local.unsqueeze(0),
        )[0]
    )


    # =========================================================================
    # Perpendicular approach vector
    # =========================================================================

    toward_robot_w = torch.tensor(
        [
            -1.0,
            0.0,
            0.0,
        ],
        device=robot.device,
        dtype=torch.float32,
    )

    approach_out_w = (
        toward_robot_w
        - torch.dot(
            toward_robot_w,
            handle_axis_w,
        )
        * handle_axis_w
    )

    approach_out_w = normalize_vector(
        approach_out_w
    )


    # =========================================================================
    # Pre-grasp and contact-search target
    # =========================================================================

    pregrasp_pos_w = (
        grasp_pos_w
        + PREGRASP_DISTANCE
        * approach_out_w
    )

    contact_target_pos_w = (
        grasp_pos_w
        + CONTACT_TARGET_DISTANCE
        * approach_out_w
    )


    # =========================================================================
    # Handshake orientation
    # =========================================================================

    desired_across_w = (
        handle_axis_w.clone()
    )

    desired_finger_w = normalize_vector(
        -approach_out_w
    )

    desired_normal_w = normalize_vector(
        torch.cross(
            desired_across_w,
            desired_finger_w,
            dim=0,
        )
    )

    desired_finger_w = normalize_vector(
        torch.cross(
            desired_normal_w,
            desired_across_w,
            dim=0,
        )
    )

    desired_semantic_basis_w = torch.stack(
        (
            desired_across_w,
            desired_finger_w,
            desired_normal_w,
        ),
        dim=1,
    )

    initial_hand_rot_w = matrix_from_quat(
        initial_hand_quat_w.unsqueeze(0)
    )[0]

    semantic_delta_rot = (
        desired_semantic_basis_w
        @ current_semantic_basis_w.transpose(
            0,
            1,
        )
    )

    target_hand_rot_w = (
        semantic_delta_rot
        @ initial_hand_rot_w
    )

    target_hand_quat_w = quat_from_matrix(
        target_hand_rot_w.unsqueeze(0)
    )[0]


    # =========================================================================
    # Convert poses to G1 root frame
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
        pregrasp_pos_b,
        target_hand_quat_b,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        pregrasp_pos_w.unsqueeze(0),
        target_hand_quat_w.unsqueeze(0),
    )

    (
        contact_target_pos_b,
        _,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        contact_target_pos_w.unsqueeze(0),
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
    # Diagnostics
    # =========================================================================

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN REVO2 CONTROLLED HANDLE GRASP"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"[PREGRASP]      "
        f"{vec_string(pregrasp_pos_w)}"
    )

    print(
        f"[CONTACT TARGET] "
        f"{vec_string(contact_target_pos_w)}"
    )

    print(
        f"[GRASP POINT]    "
        f"{vec_string(grasp_pos_w)}"
    )

    print()

    print(
        "[SEQUENCE]"
    )

    print(
        "    1. Reach handshake pre-grasp"
    )

    print(
        "    2. Approach shaft"
    )

    print(
        "    3. Detect near-contact"
    )

    print(
        "    4. Freeze arm"
    )

    print(
        "    5. Close Revo2 fingers"
    )

    print()


    # =========================================================================
    # Runtime state
    # =========================================================================

    step = 0

    contact_latched = False

    contact_time_s = 0.0

    latched_arm_joint_pos = None

    closure_hold_alpha = None

    closure_stopped = False

    grasp_complete = False


    # =========================================================================
    # Main loop
    # =========================================================================

    while simulation_app.is_running():

        with torch.inference_mode():

            time_s = (
                step
                * sim_dt
            )


            # =================================================================
            # Current hand pose
            # =================================================================

            root_pose_w = (
                robot.data.root_state_w[
                    :,
                    0:7,
                ]
            )

            hand_pose_w = (
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
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                hand_pose_w[:, 0:3],
                hand_pose_w[:, 3:7],
            )


            # =================================================================
            # Choose arm target
            # =================================================================

            phase_name = "PREGRASP"

            if not contact_latched:

                if time_s < PREGRASP_DURATION_S:

                    phase_name = "PREGRASP"

                    alpha = smoothstep(
                        time_s
                        / PREGRASP_DURATION_S
                    )

                    desired_pos_b = (
                        initial_hand_pos_b
                        * (
                            1.0
                            - alpha
                        )
                        + pregrasp_pos_b
                        * alpha
                    )

                    desired_quat_b = quat_slerp(
                        initial_hand_quat_b[0],
                        target_hand_quat_b[0],
                        alpha,
                    ).unsqueeze(0)

                else:

                    phase_name = "APPROACH"

                    approach_time = (
                        time_s
                        - PREGRASP_DURATION_S
                    )

                    approach_alpha = smoothstep(
                        approach_time
                        / APPROACH_DURATION_S
                    )

                    desired_pos_b = (
                        pregrasp_pos_b
                        * (
                            1.0
                            - approach_alpha
                        )
                        + contact_target_pos_b
                        * approach_alpha
                    )

                    desired_quat_b = (
                        target_hand_quat_b
                    )

                command = torch.cat(
                    (
                        desired_pos_b,
                        desired_quat_b,
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

                raw_delta = (
                    raw_arm_target
                    - current_arm_joint_pos
                )

                bounded_delta = torch.clamp(
                    raw_delta,
                    min=-MAX_ARM_JOINT_STEP_RAD,
                    max=MAX_ARM_JOINT_STEP_RAD,
                )

                next_arm_joint_pos = (
                    current_arm_joint_pos
                    + bounded_delta
                )

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

            else:

                phase_name = "GRASP"

                next_arm_joint_pos = (
                    latched_arm_joint_pos.clone()
                )


            # =================================================================
            # Build complete robot state
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos


            # =================================================================
            # Finger closure AFTER contact latch
            # =================================================================

            closure_alpha = 0.0

            if contact_latched:

                grasp_elapsed = (
                    time_s
                    - contact_time_s
                )

                closure_alpha = smoothstep(
                    grasp_elapsed
                    / GRASP_DURATION_S
                )

                if closure_hold_alpha is not None:
                    closure_alpha = min(
                        closure_alpha,
                        closure_hold_alpha,
                    )

                thumb_meta = (
                    THUMB_METACARPAL_TARGET
                    * closure_alpha
                )

                thumb_prox = (
                    THUMB_PROXIMAL_TARGET
                    * closure_alpha
                )

                index_prox = (
                    INDEX_PROXIMAL_TARGET
                    * closure_alpha
                )

                middle_prox = (
                    MIDDLE_PROXIMAL_TARGET
                    * closure_alpha
                )

                ring_prox = (
                    RING_PROXIMAL_TARGET
                    * closure_alpha
                )

                pinky_prox = (
                    PINKY_PROXIMAL_TARGET
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
                    THUMB_DISTAL_MULTIPLIER
                    * thumb_prox
                )

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["index_prox"],
                ] = index_prox

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["index_dist"],
                ] = (
                    FINGER_DISTAL_MULTIPLIER
                    * index_prox
                )

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["middle_prox"],
                ] = middle_prox

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["middle_dist"],
                ] = (
                    FINGER_DISTAL_MULTIPLIER
                    * middle_prox
                )

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["ring_prox"],
                ] = ring_prox

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["ring_dist"],
                ] = (
                    FINGER_DISTAL_MULTIPLIER
                    * ring_prox
                )

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["pinky_prox"],
                ] = pinky_prox

                next_robot_joint_pos[
                    0,
                    hand_joint_ids["pinky_dist"],
                ] = (
                    FINGER_DISTAL_MULTIPLIER
                    * pinky_prox
                )


            # =================================================================
            # Apply state
            # =================================================================

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
                zero_robot_joint_vel,
            )

            # Tripod remains locked during grasp validation.
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
            # Handle segment
            # =================================================================

            handle_pos_now = (
                tripod.data.body_pos_w[
                    0,
                    handle_id,
                ]
            )

            handle_quat_now = (
                tripod.data.body_quat_w[
                    0,
                    handle_id,
                ].unsqueeze(0)
            )

            handle_axis_now = normalize_vector(
                quat_apply(
                    handle_quat_now,
                    local_handle_x.unsqueeze(0),
                )[0]
            )

            segment_start = (
                handle_pos_now
                - HANDLE_HALF_LENGTH
                * handle_axis_now
            )

            segment_end = (
                handle_pos_now
                + HANDLE_HALF_LENGTH
                * handle_axis_now
            )


            # =================================================================
            # Finger clearance
            # =================================================================

            minimum_clearance = float(
                "inf"
            )

            for fingertip_id in fingertip_ids:

                fingertip_pos = (
                    robot.data.body_pos_w[
                        0,
                        fingertip_id,
                    ]
                )

                axis_distance = (
                    point_to_segment_distance(
                        fingertip_pos,
                        segment_start,
                        segment_end,
                    )
                )

                clearance = (
                    axis_distance.item()
                    - HANDLE_RADIUS
                )

                minimum_clearance = min(
                    minimum_clearance,
                    clearance,
                )


            # =================================================================
            # Detect first deliberate contact during APPROACH
            # =================================================================

            if (
                not contact_latched
                and time_s >= PREGRASP_DURATION_S
                and minimum_clearance
                <= CONTACT_CLEARANCE_M
            ):

                contact_latched = True

                contact_time_s = (
                    time_s
                )

                latched_arm_joint_pos = (
                    robot.data.joint_pos[
                        :,
                        arm_joint_ids,
                    ].clone()
                )

                print()
                print(
                    "**************** CONTACT LATCHED ****************"
                )

                print(
                    f"[CONTACT] clearance="
                    f"{minimum_clearance:.4f} m"
                )

                print(
                    "[CONTACT] Arm approach stopped."
                )

                print(
                    "[CONTACT] Beginning Revo2 finger closure."
                )

                print(
                    "*************************************************"
                )

                print()


            # =================================================================
            # Prevent excessive kinematic penetration while closing
            # =================================================================

            if (
                contact_latched
                and not closure_stopped
                and minimum_clearance
                < -MAX_ALLOWED_PENETRATION_M
            ):

                closure_stopped = True

                closure_hold_alpha = (
                    closure_alpha
                )

                print()
                print(
                    "*************** CLOSURE SAFETY STOP ***************"
                )

                print(
                    f"[GRASP] fingertip estimate="
                    f"{minimum_clearance:.4f} m"
                )

                print(
                    f"[GRASP] closure held at "
                    f"{closure_hold_alpha:.3f}"
                )

                print(
                    "***************************************************"
                )

                print()


            # =================================================================
            # Grasp complete
            # =================================================================

            if (
                contact_latched
                and not grasp_complete
                and closure_alpha >= 0.999
            ):

                grasp_complete = True

                print()
                print(
                    "**************** GRASP COMPLETE ****************"
                )

                print(
                    "[GRASP] Revo2 closure target reached."
                )

                print(
                    "[GRASP] Holding grasp pose."
                )

                print(
                    "***********************************************"
                )

                print()


            # =================================================================
            # Telemetry
            # =================================================================

            if (
                step
                % PRINT_EVERY_STEPS
                == 0
            ):

                hand_pos_now = (
                    robot.data.body_pos_w[
                        0,
                        hand_base_id,
                    ]
                )

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

                print(
                    f"[{phase_name}] "
                    f"time={time_s:5.2f}s | "
                    f"hand={vec_string(hand_pos_now)}"
                )

                print(
                    "[CONTACT] "
                    f"clearance={minimum_clearance:+.4f} m | "
                    f"latched={contact_latched}"
                )

                print(
                    "[HAND] "
                    f"closure={closure_alpha:.3f} | "
                    f"safety_stop={closure_stopped}"
                )

                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+7.3f} deg | "
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