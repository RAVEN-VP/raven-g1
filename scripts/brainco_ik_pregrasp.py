"""RAVEN G1 Mode 15 + BrainCo Revo2 handle-aligned pre-grasp.

This test uses the full 7-DOF LEFT G1 arm:

    shoulder pitch
    shoulder roll
    shoulder yaw
    elbow
    wrist roll
    wrist pitch
    wrist yaw

The BrainCo hand is oriented deliberately for a tripod-handle grasp.

Grasp convention
----------------
The real tripod handle is approximately 45 degrees downward toward
the operator.

Instead of presenting the OPEN PALM directly toward the handle, the
hand is oriented in a handshake-like configuration:

    - pinky -> index direction follows the handle shaft
    - fingers point toward the shaft from the side
    - palm plane is approximately vertical / edge-on

This is PRE-GRASP only.

No finger closing.
No physical contact.
No tripod movement.
No RL.
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
    description="RAVEN Revo2 handle-aligned 7-DOF pre-grasp."
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

# Bring the tripod slightly closer than the previous 0.65 m experiment.
#
# We want the G1 operating comfortably inside its arm workspace rather
# than extending toward a singular configuration.
TRIPOD_X = 0.60


# Slow, deliberate movement.
REACH_DURATION_S = 10.0


# -------------------------------------------------------------------------
# Real tripod handle
# -------------------------------------------------------------------------

HANDLE_HALF_LENGTH = 0.16

HANDLE_RADIUS = 0.018


# -------------------------------------------------------------------------
# Grasp location
#
# The handle centre is local X=0.
#
# Rear end toward G1:
#     X = -0.16
#
# We choose:
#     X = -0.10
#
# This is approximately 6 cm inward from the rear end of the shaft.
# -------------------------------------------------------------------------

GRASP_ALONG_HANDLE_X = -0.10


# -------------------------------------------------------------------------
# Pre-grasp approach distance
#
# The hand remains 12 cm away from the actual shaft along an axis
# perpendicular to the handle.
# -------------------------------------------------------------------------

PREGRASP_DISTANCE = 0.12


# -------------------------------------------------------------------------
# IK stability
# -------------------------------------------------------------------------

DLS_DAMPING = 0.15


# Maximum joint movement PER 5 ms simulation step.
#
# This makes violent IK jumps impossible.
MAX_JOINT_STEP_RAD = math.radians(
    0.10
)


# Telemetry.
PRINT_EVERY_STEPS = 50


# =============================================================================
# Helpers
# =============================================================================


def resolve_body(
    articulation,
    body_name: str,
) -> int:
    """Resolve an articulation body by exact name."""

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
    """Resolve an articulation joint by exact name."""

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
    """Safely normalize a vector."""

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


def vec_string(
    vector,
) -> str:
    """Pretty-print XYZ."""

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
    """Distance from a point to a finite line segment."""

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
    # G1 + BrainCo Revo2
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
    # Start physics
    # =========================================================================

    sim.reset()


    # =========================================================================
    # Neutral states
    # =========================================================================

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
    # Deterministic settling
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
    # Full real 7-DOF left arm
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
    # Revo2 geometry
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
    # Fixed-base Jacobian indexing
    # =========================================================================

    if not robot.is_fixed_base:
        raise RuntimeError(
            "This test requires the G1 root to be fixed."
        )

    hand_jacobian_index = (
        hand_base_id
        - 1
    )


    # =========================================================================
    # Initial hand and fingertip geometry
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
    # Construct CURRENT semantic hand frame
    #
    # Semantic X:
    #     pinky -> index
    #
    # Semantic Y:
    #     palm base -> fingertips
    #
    # Semantic Z:
    #     normal to palm
    #
    # We derive this from the ACTUAL Revo2 geometry.
    # =========================================================================

    current_across_w = normalize_vector(
        index_pos_w
        - pinky_pos_w
    )

    current_finger_raw_w = (
        fingertip_centroid_w
        - initial_hand_pos_w
    )

    # Remove any component parallel to the across-palm axis.
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

    # Rebuild perfectly orthogonal finger direction.
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
    # Grasp point ON the shaft
    #
    # This lies 10 cm toward the rear from the handle centre.
    # =========================================================================

    grasp_offset_handle = torch.tensor(
        [
            GRASP_ALONG_HANDLE_X,
            0.0,
            0.0,
        ],
        device=robot.device,
        dtype=torch.float32,
    )

    grasp_offset_w = quat_apply(
        handle_quat_w.unsqueeze(0),
        grasp_offset_handle.unsqueeze(0),
    )[0]

    grasp_pos_w = (
        handle_pos_w
        + grasp_offset_w
    )


    # =========================================================================
    # Determine a perpendicular approach direction
    #
    # Start with world -X, i.e. from the tripod toward the G1.
    #
    # Remove the component parallel to the handle.
    #
    # The resulting vector is perpendicular to the 45-degree shaft.
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
    # Pre-grasp point
    #
    # 12 cm outward from the shaft.
    # =========================================================================

    pregrasp_pos_w = (
        grasp_pos_w
        + PREGRASP_DISTANCE
        * approach_out_w
    )


    # =========================================================================
    # Construct DESIRED handshake-like hand frame
    #
    # Across-palm direction:
    #     pinky -> index
    #     aligned WITH handle axis
    #
    # Finger direction:
    #     from pre-grasp toward handle
    #
    # Palm normal:
    #     automatically becomes sideways.
    #
    # This is what removes the "open palm facing the handle" posture.
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


    # =========================================================================
    # Convert semantic hand alignment into the desired hand quaternion
    #
    # Current semantic basis:
    #
    #     S_current = R_hand_current * S_local
    #
    # Therefore:
    #
    #     R_target
    #       = S_target
    #       * S_current^T
    #       * R_hand_current
    # =========================================================================

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
    # Convert initial and target poses to G1 root frame
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
        target_hand_pos_b,
        target_hand_quat_b,
    ) = subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        pregrasp_pos_w.unsqueeze(0),
        target_hand_quat_w.unsqueeze(0),
    )


    # =========================================================================
    # IK controller
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
        " RAVEN 7-DOF HANDSHAKE PRE-GRASP"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"[TRIPOD] X = "
        f"{TRIPOD_X:.2f} m"
    )

    print(
        "[HANDLE] 45-degree shaft"
    )

    print()

    print(
        f"[HANDLE CENTRE] "
        f"{vec_string(handle_pos_w)}"
    )

    print(
        f"[GRASP POINT]   "
        f"{vec_string(grasp_pos_w)}"
    )

    print(
        f"[PREGRASP]      "
        f"{vec_string(pregrasp_pos_w)}"
    )

    print()

    print(
        "[HAND ORIENTATION]"
    )

    print(
        "    pinky -> index  = handle axis"
    )

    print(
        "    fingers         = toward handle"
    )

    print(
        "    palm            = edge-on / handshake-style"
    )

    print()

    print(
        f"[HANDLE AXIS] "
        f"{vec_string(handle_axis_w)}"
    )

    print(
        f"[APPROACH]    "
        f"{vec_string(-approach_out_w)}"
    )

    print(
        f"[PALM NORMAL] "
        f"{vec_string(desired_normal_w)}"
    )

    print()

    print(
        "[CONTROLLED JOINTS]"
    )

    for joint_id in arm_joint_ids:

        print(
            f"    "
            f"{robot.joint_names[joint_id]}"
        )

    print()

    print(
        f"[IK] damping = "
        f"{DLS_DAMPING:.3f}"
    )

    print(
        f"[IK] max joint step = "
        f"{math.degrees(MAX_JOINT_STEP_RAD):.3f} deg"
    )

    print()


    # =========================================================================
    # Simulation
    # =========================================================================

    step = 0

    safety_stopped = False

    safety_reason = ""

    last_safe_arm_joint_pos = (
        robot.data.joint_pos[
            :,
            arm_joint_ids,
        ].clone()
    )


    while simulation_app.is_running():

        with torch.inference_mode():

            # =================================================================
            # Current hand/root state
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
            # Smooth trajectory
            # =================================================================

            elapsed_s = (
                step
                * sim_dt
            )

            alpha = min(
                elapsed_s
                / REACH_DURATION_S,
                1.0,
            )

            smooth_alpha = (
                3.0
                * alpha
                * alpha
                - 2.0
                * alpha
                * alpha
                * alpha
            )


            # =================================================================
            # Position interpolation
            # =================================================================

            desired_pos_b = (
                initial_hand_pos_b
                * (
                    1.0
                    - smooth_alpha
                )
                + target_hand_pos_b
                * smooth_alpha
            )


            # =================================================================
            # Orientation interpolation
            # =================================================================

            desired_quat_b = quat_slerp(
                initial_hand_quat_b[0],
                target_hand_quat_b[0],
                float(
                    smooth_alpha
                ),
            ).unsqueeze(0)


            # =================================================================
            # Absolute pose command
            # =================================================================

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


            # =================================================================
            # Full 6x7 left-arm Jacobian
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
            # Jacobian health
            # =================================================================

            singular_values = torch.linalg.svdvals(
                jacobian
            )

            minimum_singular_value = (
                singular_values.min().item()
            )


            # =================================================================
            # Current arm joints
            # =================================================================

            current_arm_joint_pos = (
                robot.data.joint_pos[
                    :,
                    arm_joint_ids,
                ]
            )


            # =================================================================
            # Solve IK
            # =================================================================

            raw_desired_arm_joint_pos = (
                current_arm_joint_pos.clone()
            )

            raw_delta_max_deg = 0.0

            applied_delta_max_deg = 0.0


            if not safety_stopped:

                raw_desired_arm_joint_pos = (
                    ik_controller.compute(
                        hand_pos_b,
                        hand_quat_b,
                        jacobian,
                        current_arm_joint_pos,
                    )
                )

                if not torch.isfinite(
                    raw_desired_arm_joint_pos
                ).all():

                    safety_stopped = True

                    safety_reason = (
                        "IK produced NaN or Inf."
                    )


            # =================================================================
            # Bounded joint update
            # =================================================================

            if not safety_stopped:

                raw_delta = (
                    raw_desired_arm_joint_pos
                    - current_arm_joint_pos
                )

                raw_delta_max_deg = math.degrees(
                    torch.max(
                        torch.abs(
                            raw_delta
                        )
                    ).item()
                )

                bounded_delta = torch.clamp(
                    raw_delta,
                    min=-MAX_JOINT_STEP_RAD,
                    max=MAX_JOINT_STEP_RAD,
                )

                applied_delta_max_deg = math.degrees(
                    torch.max(
                        torch.abs(
                            bounded_delta
                        )
                    ).item()
                )

                next_arm_joint_pos = (
                    current_arm_joint_pos
                    + bounded_delta
                )


                # =============================================================
                # Joint limits
                # =============================================================

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

                last_safe_arm_joint_pos = (
                    next_arm_joint_pos.clone()
                )

            else:

                next_arm_joint_pos = (
                    last_safe_arm_joint_pos.clone()
                )


            # =================================================================
            # Full 51-joint robot pose
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos


            # =================================================================
            # Apply robot state
            # =================================================================

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
                zero_robot_joint_vel,
            )


            # =================================================================
            # Tripod remains LOCKED during pre-grasp development
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
            # Telemetry
            # =================================================================

            if (
                step
                % PRINT_EVERY_STEPS
                == 0
            ):

                current_hand_pos_w = (
                    robot.data.body_pos_w[
                        0,
                        hand_base_id,
                    ]
                )

                current_hand_quat_w = (
                    robot.data.body_quat_w[
                        0,
                        hand_base_id,
                    ]
                )


                # =============================================================
                # Position error
                # =============================================================

                position_error = (
                    torch.linalg.vector_norm(
                        current_hand_pos_w
                        - pregrasp_pos_w
                    ).item()
                )


                # =============================================================
                # Orientation error
                #
                # q and -q represent the same rotation.
                # =============================================================

                quaternion_dot = torch.abs(
                    torch.dot(
                        current_hand_quat_w,
                        target_hand_quat_w,
                    )
                )

                quaternion_dot = torch.clamp(
                    quaternion_dot,
                    0.0,
                    1.0,
                )

                orientation_error_deg = math.degrees(
                    2.0
                    * math.acos(
                        quaternion_dot.item()
                    )
                )


                # =============================================================
                # Actual handle segment
                # =============================================================

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

                handle_axis_now = quat_apply(
                    handle_quat_now,
                    local_handle_x.unsqueeze(0),
                )[0]

                handle_axis_now = normalize_vector(
                    handle_axis_now
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


                # =============================================================
                # Minimum fingertip clearance
                # =============================================================

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


                # =============================================================
                # Tripod state
                # =============================================================

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


                # =============================================================
                # Print
                # =============================================================

                print(
                    "[PREGRASP] "
                    f"alpha={alpha:4.2f} | "
                    f"hand={vec_string(current_hand_pos_w)}"
                )

                print(
                    "[ERROR] "
                    f"position={position_error:.4f} m | "
                    f"orientation="
                    f"{orientation_error_deg:.2f} deg | "
                    f"finger_clearance="
                    f"{minimum_clearance:.4f} m"
                )

                print(
                    "[IK HEALTH] "
                    f"sigma_min="
                    f"{minimum_singular_value:.6f} | "
                    f"raw_dq_max="
                    f"{raw_delta_max_deg:.2f} deg | "
                    f"applied_dq_max="
                    f"{applied_delta_max_deg:.2f} deg"
                )

                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+7.3f} deg | "
                    f"tilt={tilt_deg:+7.3f} deg | "
                    f"camera={vec_string(camera_pos)}"
                )

                if safety_stopped:

                    print(
                        "[SAFETY STOP] "
                        f"{safety_reason}"
                    )

                print()


            # =================================================================
            # Next frame
            # =================================================================

            step += 1


# =============================================================================
# Entry
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()