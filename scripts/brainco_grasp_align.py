"""RAVEN G1 + BrainCo Revo2 grasp-aperture alignment test.

Purpose
-------
Correctly place the open BrainCo Revo2 hand around the tripod handle
before finger closure.

The important change from the previous grasp test is that we do NOT
position the hand using left_base_link alone.

Instead we construct a virtual GRASP APERTURE CENTRE:

    aperture centre =
        midpoint(
            thumb tip,
            centroid(index, middle, ring, pinky tips)
        )

The 45-degree tripod shaft is then positioned inside that aperture.

Semantic grasp frame
--------------------
X = pinky -> index
    aligned with the tripod handle axis

Y = thumb -> finger centroid
    aligned toward the handle

Z = palm normal

Sequence
--------
1. Reach a safe aperture-centred pre-grasp.
2. Approach the shaft with fingers OPEN.
3. Place the shaft inside the open thumb/finger aperture.
4. Hold there for inspection.

No finger closure in this test.
No tripod motion.
No RL.
"""

import argparse
import copy
import math

from isaaclab.app import AppLauncher


# =============================================================================
# Launch
# =============================================================================

parser = argparse.ArgumentParser(
    description="RAVEN BrainCo grasp-aperture alignment validation."
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

APPROACH_DURATION_S = 5.0


# -----------------------------------------------------------------------------
# Handle
# -----------------------------------------------------------------------------

HANDLE_HALF_LENGTH = 0.16

HANDLE_RADIUS = 0.018


# Grasp approximately 6 cm inward from the operator-side end of the handle.
GRASP_ALONG_HANDLE_X = -0.10


# -----------------------------------------------------------------------------
# Aperture positioning
#
# These values describe where the CENTRE OF THE OPEN HAND APERTURE sits
# relative to the handle shaft.
#
# Pre-grasp:
#     10 cm away from shaft.
#
# Final:
#     aperture centre almost coincident with shaft.
# -----------------------------------------------------------------------------

PREGRASP_APERTURE_STANDOFF = 0.10

FINAL_APERTURE_STANDOFF = 0.005


# -----------------------------------------------------------------------------
# IK
# -----------------------------------------------------------------------------

DLS_DAMPING = 0.15

MAX_ARM_JOINT_STEP_RAD = math.radians(
    0.10
)

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


def vector_angle_deg(
    a,
    b,
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
    # Full 7-DOF left arm
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
            "This test requires the G1 root to remain fixed."
        )

    hand_jacobian_index = (
        hand_base_id
        - 1
    )


    # =========================================================================
    # Initial hand geometry
    # =========================================================================

    hand_base_pos_w = (
        robot.data.body_pos_w[
            0,
            hand_base_id,
        ].clone()
    )

    hand_base_quat_w = (
        robot.data.body_quat_w[
            0,
            hand_base_id,
        ].clone()
    )

    thumb_tip_w = (
        robot.data.body_pos_w[
            0,
            thumb_tip_id,
        ].clone()
    )

    index_tip_w = (
        robot.data.body_pos_w[
            0,
            index_tip_id,
        ].clone()
    )

    middle_tip_w = (
        robot.data.body_pos_w[
            0,
            middle_tip_id,
        ].clone()
    )

    ring_tip_w = (
        robot.data.body_pos_w[
            0,
            ring_tip_id,
        ].clone()
    )

    pinky_tip_w = (
        robot.data.body_pos_w[
            0,
            pinky_tip_id,
        ].clone()
    )


    # =========================================================================
    # Finger centroid
    # =========================================================================

    finger_centroid_w = (
        index_tip_w
        + middle_tip_w
        + ring_tip_w
        + pinky_tip_w
    ) / 4.0


    # =========================================================================
    # OPEN GRASP APERTURE CENTRE
    #
    # Halfway between the thumb and the four fingers.
    # =========================================================================

    aperture_center_w = (
        thumb_tip_w
        + finger_centroid_w
    ) * 0.5


    # =========================================================================
    # Current semantic grasp axes
    #
    # X = pinky -> index
    # Y = thumb -> fingers
    # Z = palm normal
    # =========================================================================

    current_across_w = normalize_vector(
        index_tip_w
        - pinky_tip_w
    )

    current_closing_raw_w = (
        finger_centroid_w
        - thumb_tip_w
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
    # Aperture-centre offset expressed in HAND-LOCAL coordinates
    #
    # This allows us to position the actual opening of the hand, rather than
    # guessing where left_base_link should be.
    # =========================================================================

    initial_hand_rot_w = matrix_from_quat(
        hand_base_quat_w.unsqueeze(0)
    )[0]

    aperture_offset_w = (
        aperture_center_w
        - hand_base_pos_w
    )

    aperture_offset_hand = (
        initial_hand_rot_w.transpose(
            0,
            1,
        )
        @ aperture_offset_w
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


    # =========================================================================
    # Grasp point
    # =========================================================================

    grasp_pos_w = (
        handle_pos_w
        + raw_handle_axis_w
        * GRASP_ALONG_HANDLE_X
    )


    # =========================================================================
    # Choose handle-axis sign that requires the least hand flip
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
    # APPROACH DIRECTION
    #
    # Derive it from where the hand actually starts relative to the shaft.
    #
    # Remove any component along the shaft so the approach is guaranteed to be
    # perpendicular to the handle.
    # =========================================================================

    from_handle_to_hand_w = (
        aperture_center_w
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
    # Desired grasp axes
    #
    # X = handle axis
    # Y = thumb -> fingers, directed INTO the shaft
    # Z = palm normal
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


    # =========================================================================
    # Desired hand rotation
    # =========================================================================

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
    # Target APERTURE positions
    # =========================================================================

    pregrasp_aperture_w = (
        grasp_pos_w
        + PREGRASP_APERTURE_STANDOFF
        * approach_out_w
    )

    final_aperture_w = (
        grasp_pos_w
        + FINAL_APERTURE_STANDOFF
        * approach_out_w
    )


    # =========================================================================
    # Convert aperture targets into HAND BASE targets
    #
    # hand_base =
    #
    #     desired_aperture
    #     - R_target * aperture_offset_local
    #
    # This is the key correction.
    # =========================================================================

    target_aperture_offset_w = (
        target_hand_rot_w
        @ aperture_offset_hand
    )

    pregrasp_hand_base_w = (
        pregrasp_aperture_w
        - target_aperture_offset_w
    )

    final_hand_base_w = (
        final_aperture_w
        - target_aperture_offset_w
    )


    # =========================================================================
    # Convert poses into robot-root frame
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
        hand_base_pos_w.unsqueeze(0),
        hand_base_quat_w.unsqueeze(0),
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
    # Startup diagnostics
    # =========================================================================

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN REVO2 GRASP-APERTURE ALIGNMENT"
    )

    print(
        "============================================================"
    )

    print()

    print(
        f"[HANDLE] "
        f"centre={vec_string(handle_pos_w)}"
    )

    print(
        f"[GRASP POINT] "
        f"{vec_string(grasp_pos_w)}"
    )

    print()

    print(
        f"[INITIAL APERTURE] "
        f"{vec_string(aperture_center_w)}"
    )

    print(
        f"[PREGRASP APERTURE] "
        f"{vec_string(pregrasp_aperture_w)}"
    )

    print(
        f"[FINAL APERTURE] "
        f"{vec_string(final_aperture_w)}"
    )

    print()

    print(
        f"[HANDLE AXIS] "
        f"{vec_string(handle_axis_w)}"
    )

    print(
        f"[APPROACH IN] "
        f"{vec_string(approach_in_w)}"
    )

    print(
        f"[PALM NORMAL] "
        f"{vec_string(desired_normal_w)}"
    )

    print()

    print(
        "[RAVEN] Fingers remain OPEN."
    )

    print(
        "[RAVEN] This run only aligns the real grasp aperture."
    )

    print()


    # =========================================================================
    # Runtime
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
            # Phase
            # =================================================================

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

            elif (
                time_s
                < PREGRASP_DURATION_S
                + APPROACH_DURATION_S
            ):

                phase = "APPROACH"

                approach_time = (
                    time_s
                    - PREGRASP_DURATION_S
                )

                alpha = smoothstep(
                    approach_time
                    / APPROACH_DURATION_S
                )

                desired_hand_pos_b = (
                    pregrasp_hand_pos_b
                    * (
                        1.0
                        - alpha
                    )
                    + final_hand_pos_b
                    * alpha
                )

                desired_hand_quat_b = (
                    target_hand_quat_b
                )

            else:

                phase = "GRASP_READY"

                desired_hand_pos_b = (
                    final_hand_pos_b
                )

                desired_hand_quat_b = (
                    target_hand_quat_b
                )


            # =================================================================
            # IK
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
            # Full robot pose
            #
            # BrainCo fingers remain neutral/open.
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = next_arm_joint_pos


            # =================================================================
            # Apply
            # =================================================================

            robot.write_joint_state_to_sim(
                next_robot_joint_pos,
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


            # =================================================================
            # Telemetry
            # =================================================================

            if (
                step
                % PRINT_EVERY_STEPS
                == 0
            ):

                # -------------------------------------------------------------
                # Actual hand geometry
                # -------------------------------------------------------------

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


                # -------------------------------------------------------------
                # Current semantic axes
                # -------------------------------------------------------------

                across_now = normalize_vector(
                    index_now
                    - pinky_now
                )

                closing_now_raw = (
                    finger_centroid_now
                    - thumb_now
                )

                closing_now = (
                    closing_now_raw
                    - torch.dot(
                        closing_now_raw,
                        across_now,
                    )
                    * across_now
                )

                closing_now = normalize_vector(
                    closing_now
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


                # -------------------------------------------------------------
                # Aperture distance to shaft axis
                # -------------------------------------------------------------

                aperture_axis_distance = (
                    point_to_segment_distance(
                        aperture_now,
                        segment_start,
                        segment_end,
                    ).item()
                )


                # -------------------------------------------------------------
                # Minimum fingertip surface clearance
                #
                # Diagnostic only.
                #
                # We do NOT stop movement just because one open fingertip
                # reaches the shaft first.
                # -------------------------------------------------------------

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


                # -------------------------------------------------------------
                # Alignment
                #
                # Handle axis is directionless, so use absolute dot for the
                # across-hand comparison.
                # -------------------------------------------------------------

                across_dot = torch.abs(
                    torch.dot(
                        normalize_vector(
                            across_now
                        ),
                        normalize_vector(
                            handle_axis_w
                        ),
                    )
                )

                across_dot = torch.clamp(
                    across_dot,
                    0.0,
                    1.0,
                )

                across_error_deg = math.degrees(
                    math.acos(
                        across_dot.item()
                    )
                )

                closing_error_deg = (
                    vector_angle_deg(
                        closing_now,
                        desired_closing_w,
                    )
                )


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
                    "[APERTURE] "
                    f"pos={vec_string(aperture_now)} | "
                    f"axis_distance="
                    f"{aperture_axis_distance:.4f} m"
                )

                print(
                    "[ALIGNMENT] "
                    f"handle/across="
                    f"{across_error_deg:.2f} deg | "
                    f"closing_axis="
                    f"{closing_error_deg:.2f} deg"
                )

                print(
                    "[FINGERS] "
                    f"minimum_surface_clearance="
                    f"{minimum_finger_clearance:+.4f} m"
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