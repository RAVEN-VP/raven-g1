"""RAVEN calibrated BrainCo Revo2 tripod-handle grasp.

This test uses the manually calibrated LEFT-arm pose obtained in Isaac Sim.

The arm pose is no longer solved with IK during the grasp.
Instead:

    1. Smoothly move the seven G1 arm joints to the calibrated pose.
    2. Hold the arm exactly there.
    3. Slowly close the BrainCo Revo2 hand around the 45-degree handle.
    4. Keep the tripod pan/tilt locked.

This isolates finger/grasp geometry from arm IK.

No RL.
No tripod manipulation yet.
"""

import argparse
import copy
import math

from isaaclab.app import AppLauncher


# =============================================================================
# Launch
# =============================================================================

parser = argparse.ArgumentParser(
    description="RAVEN calibrated BrainCo Revo2 grasp."
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

from raven_g1.assets import (
    G1_BRAINCO_MODE15_CFG,
    TRIPOD_CAMERA_CFG,
)


# =============================================================================
# Configuration
# =============================================================================

TRIPOD_X = 0.63

ARM_MOVE_DURATION_S = 8.0

ARM_HOLD_BEFORE_GRASP_S = 2.0

FINGER_CLOSE_DURATION_S = 6.0

PRINT_EVERY_STEPS = 50


# =============================================================================
# Exact manually calibrated arm pose
#
# Captured from:
# brainco_manual_grasp_tune.py
# =============================================================================

CALIBRATED_ARM_JOINTS = {
    "left_shoulder_pitch_joint": -1.10463631,
    "left_shoulder_roll_joint": +0.45274511,
    "left_shoulder_yaw_joint": +0.89249939,
    "left_elbow_joint": +2.09439969,
    "left_wrist_roll_joint": +0.67981434,
    "left_wrist_pitch_joint": -0.22342572,
    "left_wrist_yaw_joint": -0.18944184,
}


# =============================================================================
# Calibrated pose reference
#
# Used only for telemetry / sanity checking.
# =============================================================================

CALIBRATED_HAND_POS_W = (
    0.348712,
    0.182130,
    0.876655,
)

CALIBRATED_HAND_QUAT_W = (
    0.25772622,
    -0.58921874,
    0.65283203,
    -0.40026075,
)


# =============================================================================
# Revo2 grasp targets
#
# These are deliberately moderate values, comfortably inside the official
# BrainCo / Unitree combined-model limits.
#
# We will visually inspect this grasp before increasing closure further.
# =============================================================================

THUMB_METACARPAL_TARGET = 0.40

THUMB_PROXIMAL_TARGET = 0.48

INDEX_PROXIMAL_TARGET = 0.60

MIDDLE_PROXIMAL_TARGET = 0.60

RING_PROXIMAL_TARGET = 0.60

PINKY_PROXIMAL_TARGET = 0.60


# Distal mimic relationships in the combined BrainCo model.
FINGER_DISTAL_MULTIPLIER = 1.155

THUMB_DISTAL_MULTIPLIER = 1.0


# =============================================================================
# Handle geometry
# =============================================================================

HANDLE_HALF_LENGTH = 0.16

HANDLE_RADIUS = 0.018


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
        f"{vector[0].item(): .4f}, "
        f"{vector[1].item(): .4f}, "
        f"{vector[2].item(): .4f}"
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
            2.2,
            2.0,
            1.55,
        ),
        target=(
            0.36,
            0.14,
            0.86,
        ),
    )


    # =========================================================================
    # Start
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
    # Resolve calibrated arm
    # =========================================================================

    arm_joint_names = list(
        CALIBRATED_ARM_JOINTS.keys()
    )

    arm_joint_ids = [
        resolve_joint(
            robot,
            name,
        )
        for name in arm_joint_names
    ]


    calibrated_arm_pos = torch.tensor(
        [
            [
                CALIBRATED_ARM_JOINTS[
                    name
                ]
                for name
                in arm_joint_names
            ]
        ],
        device=robot.device,
        dtype=torch.float32,
    )


    # =========================================================================
    # Resolve hand joints
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
    # Resolve bodies
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

    pan_joint_id = resolve_joint(
        tripod,
        "pan_joint",
    )

    tilt_joint_id = resolve_joint(
        tripod,
        "tilt_joint",
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


    # Rotate local X by quaternion manually using a helper matrix.
    #
    # We only need the shaft direction for telemetry.
    w = handle_quat_w[0]
    x = handle_quat_w[1]
    y = handle_quat_w[2]
    z = handle_quat_w[3]


    handle_rot = torch.stack(
        [
            torch.stack(
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - z * w),
                    2.0 * (x * z + y * w),
                ]
            ),

            torch.stack(
                [
                    2.0 * (x * y + z * w),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - x * w),
                ]
            ),

            torch.stack(
                [
                    2.0 * (x * z - y * w),
                    2.0 * (y * z + x * w),
                    1.0 - 2.0 * (x * x + y * y),
                ]
            ),
        ]
    )


    handle_axis_w = (
        handle_rot
        @ torch.tensor(
            [
                1.0,
                0.0,
                0.0,
            ],
            device=robot.device,
        )
    )


    handle_axis_w = (
        handle_axis_w
        / torch.linalg.vector_norm(
            handle_axis_w
        )
    )


    segment_start = (
        handle_pos_w
        - HANDLE_HALF_LENGTH
        * handle_axis_w
    )

    segment_end = (
        handle_pos_w
        + HANDLE_HALF_LENGTH
        * handle_axis_w
    )


    # =========================================================================
    # Startup
    # =========================================================================

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN CALIBRATED REVO2 GRASP"
    )

    print(
        "============================================================"
    )

    print()

    print(
        "[RAVEN] Using exact manually calibrated arm pose."
    )

    print(
        "[RAVEN] No IK is used during the grasp."
    )

    print()

    for name in arm_joint_names:

        print(
            f"{name:30s} "
            f"= {CALIBRATED_ARM_JOINTS[name]:+.8f} rad"
        )

    print()


    # =========================================================================
    # Loop
    # =========================================================================

    step = 0


    while simulation_app.is_running():

        with torch.inference_mode():

            time_s = (
                step
                * sim_dt
            )


            # =================================================================
            # Arm trajectory
            # =================================================================

            if time_s < ARM_MOVE_DURATION_S:

                phase = "MOVE_ARM"

                arm_alpha = smoothstep(
                    time_s
                    / ARM_MOVE_DURATION_S
                )

                current_arm_target = (
                    neutral_robot_joint_pos[
                        :,
                        arm_joint_ids,
                    ]
                    * (
                        1.0
                        - arm_alpha
                    )
                    + calibrated_arm_pos
                    * arm_alpha
                )

                closure_alpha = 0.0


            elif (
                time_s
                < ARM_MOVE_DURATION_S
                + ARM_HOLD_BEFORE_GRASP_S
            ):

                phase = "HOLD"

                current_arm_target = (
                    calibrated_arm_pos
                )

                closure_alpha = 0.0


            else:

                phase = "CLOSE_HAND"

                current_arm_target = (
                    calibrated_arm_pos
                )

                grasp_elapsed = (
                    time_s
                    - ARM_MOVE_DURATION_S
                    - ARM_HOLD_BEFORE_GRASP_S
                )

                closure_alpha = smoothstep(
                    grasp_elapsed
                    / FINGER_CLOSE_DURATION_S
                )


            # =================================================================
            # Full 51-DOF state
            # =================================================================

            next_robot_joint_pos = (
                neutral_robot_joint_pos.clone()
            )

            next_robot_joint_pos[
                :,
                arm_joint_ids,
            ] = current_arm_target


            # =================================================================
            # Finger targets
            # =================================================================

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


            # Tripod still locked.
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

                hand_pos_now = (
                    robot.data.body_pos_w[
                        0,
                        hand_base_id,
                    ]
                )


                position_error = (
                    torch.linalg.vector_norm(
                        hand_pos_now
                        - torch.tensor(
                            CALIBRATED_HAND_POS_W,
                            device=robot.device,
                        )
                    ).item()
                )


                min_finger_distance = float(
                    "inf"
                )


                for body_id in fingertip_ids:

                    fingertip_pos = (
                        robot.data.body_pos_w[
                            0,
                            body_id,
                        ]
                    )


                    distance = (
                        point_to_segment_distance(
                            fingertip_pos,
                            segment_start,
                            segment_end,
                        ).item()
                    )


                    min_finger_distance = min(
                        min_finger_distance,
                        distance
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


                print(
                    f"[{phase}] "
                    f"time={time_s:5.2f}s | "
                    f"closure={closure_alpha:.3f}"
                )


                print(
                    "[ARM] "
                    f"hand={vec_string(hand_pos_now)} | "
                    f"calibration_error="
                    f"{position_error:.4f} m"
                )


                print(
                    "[HANDLE] "
                    f"minimum_tip_axis_distance="
                    f"{min_finger_distance:.4f} m"
                )


                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+7.3f} deg | "
                    f"tilt={tilt_deg:+7.3f} deg"
                )


                print()


            step += 1


# =============================================================================
# Entry
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()