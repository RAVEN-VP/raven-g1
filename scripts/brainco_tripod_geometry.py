"""RAVEN BrainCo Revo2 / tripod geometry validation.

This stage measures the neutral geometry between:

- Unitree G1 29-DOF Mode 15
- BrainCo Revo2 hands
- RAVEN articulated tripod

IMPORTANT:
    The G1 is deliberately frozen at its initial pose for this test.

    The tripod remains physically passive.

    The tripod is temporarily moved farther forward so the Revo2 hand
    begins with a clean, collision-free gap.

No IK.
No gripping.
No learning.
"""

import argparse
import copy
import math

from isaaclab.app import AppLauncher


# =============================================================================
# Launch Isaac Sim
# =============================================================================

parser = argparse.ArgumentParser(
    description="RAVEN BrainCo Revo2 tripod geometry validation."
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
# Test setup
# =============================================================================

# The normal tripod configuration places its base at X=0.50.
#
# That made the Revo2 fingers intersect the handle immediately.
#
# For this neutral geometry test we temporarily move the tripod
# 30 cm farther forward.
TEST_TRIPOD_X = 0.80


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

        print()
        print(
            f"[RAVEN] Could not find body: {body_name}"
        )

        print()
        print("[RAVEN] Available bodies:")

        for name in articulation.body_names:
            print(f"    {name}")

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


def vec_string(vector) -> str:

    return (
        "("
        f"{vector[0].item(): .3f}, "
        f"{vector[1].item(): .3f}, "
        f"{vector[2].item(): .3f}"
        ")"
    )


# =============================================================================
# Main
# =============================================================================


def main():

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args_cli.device,
    )

    sim = sim_utils.SimulationContext(
        sim_cfg
    )

    # -------------------------------------------------------------------------
    # Ground
    # -------------------------------------------------------------------------

    ground_cfg = sim_utils.GroundPlaneCfg()

    ground_cfg.func(
        "/World/Ground",
        ground_cfg,
    )

    # -------------------------------------------------------------------------
    # Light
    # -------------------------------------------------------------------------

    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )

    light_cfg.func(
        "/World/Light",
        light_cfg,
        translation=(1.0, 0.0, 10.0),
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

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Move tripod farther away for this test.
    #
    # Original:
    #     X = 0.50
    #
    # Geometry test:
    #     X = 0.80
    #
    # This is temporary and does NOT modify TRIPOD_CAMERA_CFG itself.
    # -------------------------------------------------------------------------

    tripod_cfg.init_state.pos = (
        TEST_TRIPOD_X,
        0.0,
        0.0,
    )

    tripod = Articulation(
        tripod_cfg
    )

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------

    sim.set_camera_view(
        eye=(2.6, 2.4, 1.7),
        target=(0.40, 0.0, 0.9),
    )

    # =========================================================================
    # Initialize
    # =========================================================================

    sim.reset()

    sim_dt = sim.get_physics_dt()

    # -------------------------------------------------------------------------
    # Capture the intended neutral G1 configuration.
    #
    # We will hard-write this state every frame because this is a geometry
    # measurement, NOT a dynamics/controller test.
    # -------------------------------------------------------------------------

    frozen_robot_joint_pos = (
        robot.data.default_joint_pos.clone()
    )

    frozen_robot_joint_vel = (
        torch.zeros_like(
            frozen_robot_joint_pos
        )
    )

    # -------------------------------------------------------------------------
    # Hold G1 neutral for a few frames while PhysX/Fabric buffers initialize.
    #
    # The tripod is NOT frozen. Therefore if it moves, we know there is still
    # physical contact.
    # -------------------------------------------------------------------------

    for _ in range(10):

        robot.write_joint_state_to_sim(
            frozen_robot_joint_pos,
            frozen_robot_joint_vel,
        )

        robot.write_data_to_sim()

        tripod.write_data_to_sim()

        sim.step()

        robot.update(sim_dt)
        tripod.update(sim_dt)

    # =========================================================================
    # Resolve actual Revo2 bodies
    # =========================================================================

    left_base_id = resolve_body(
        robot,
        "left_base_link",
    )

    left_thumb_tip_id = resolve_body(
        robot,
        "left_thumb_tip_Link",
    )

    left_index_tip_id = resolve_body(
        robot,
        "left_index_tip_Link",
    )

    left_middle_tip_id = resolve_body(
        robot,
        "left_middle_tip_Link",
    )

    left_ring_tip_id = resolve_body(
        robot,
        "left_ring_tip_Link",
    )

    left_pinky_tip_id = resolve_body(
        robot,
        "left_pinky_tip_Link",
    )

    # =========================================================================
    # Resolve tripod
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
    # Measurement function
    # =========================================================================

    def print_geometry():

        robot_root_pos = (
            robot.data.root_pos_w[0]
        )

        left_base_pos = (
            robot.data.body_pos_w[
                0,
                left_base_id,
            ]
        )

        handle_pos = (
            tripod.data.body_pos_w[
                0,
                handle_id,
            ]
        )

        camera_pos = (
            tripod.data.body_pos_w[
                0,
                camera_id,
            ]
        )

        tip_names = [
            "thumb",
            "index",
            "middle",
            "ring",
            "pinky",
        ]

        tip_ids = [
            left_thumb_tip_id,
            left_index_tip_id,
            left_middle_tip_id,
            left_ring_tip_id,
            left_pinky_tip_id,
        ]

        tip_positions = torch.stack(
            [
                robot.data.body_pos_w[
                    0,
                    body_id,
                ]
                for body_id in tip_ids
            ],
            dim=0,
        )

        fingertip_centroid = (
            tip_positions.mean(dim=0)
        )

        base_delta = (
            handle_pos
            - left_base_pos
        )

        base_distance = (
            torch.linalg.vector_norm(
                base_delta
            ).item()
        )

        tip_delta = (
            handle_pos
            - fingertip_centroid
        )

        tip_distance = (
            torch.linalg.vector_norm(
                tip_delta
            ).item()
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

        print()
        print(
            "============================================================"
        )

        print(
            " RAVEN BRAINCO REVO2 / TRIPOD GEOMETRY"
        )

        print(
            "============================================================"
        )
        print()

        print(
            f"[TEST] tripod base X = "
            f"{TEST_TRIPOD_X:.2f} m"
        )

        print(
            "[TEST] G1 joints hard-frozen at neutral pose."
        )

        print(
            "[TEST] Tripod pan/tilt remain physically passive."
        )

        print()

        print(
            f"[G1 ROOT] "
            f"pos={vec_string(robot_root_pos)}"
        )

        print()

        print(
            f"[LEFT HAND BASE] "
            f"pos={vec_string(left_base_pos)}"
        )

        print(
            f"[FINGERTIP CENTRE] "
            f"pos={vec_string(fingertip_centroid)}"
        )

        print()

        print(
            f"[HANDLE] "
            f"pos={vec_string(handle_pos)}"
        )

        print()

        print(
            "[BASE -> HANDLE] "
            f"distance={base_distance:.3f} m | "
            f"delta={vec_string(base_delta)}"
        )

        print(
            "[TIPS -> HANDLE] "
            f"distance={tip_distance:.3f} m | "
            f"delta={vec_string(tip_delta)}"
        )

        print()
        print(
            "------------------------------------------------------------"
        )

        print(
            " LEFT REVO2 FINGERTIPS"
        )

        print(
            "------------------------------------------------------------"
        )

        for name, tip_position in zip(
            tip_names,
            tip_positions,
        ):

            delta = (
                handle_pos
                - tip_position
            )

            distance = (
                torch.linalg.vector_norm(
                    delta
                ).item()
            )

            print(
                f"[{name.upper():6s}] "
                f"pos={vec_string(tip_position)} | "
                f"handle_dist={distance:.3f} m | "
                f"delta={vec_string(delta)}"
            )

        print()

        print(
            "------------------------------------------------------------"
        )

        print(
            f"[TRIPOD] "
            f"camera={vec_string(camera_pos)} | "
            f"pan={pan_deg:+.3f} deg | "
            f"tilt={tilt_deg:+.3f} deg"
        )

        print()

        # -------------------------------------------------------------
        # Collision sanity test
        # -------------------------------------------------------------

        if (
            abs(pan_deg) > 0.1
            or abs(tilt_deg) > 0.1
        ):

            print(
                "[RAVEN] WARNING: Tripod moved."
            )

            print(
                "[RAVEN] There may still be physical "
                "contact with the G1."
            )

        else:

            print(
                "[RAVEN] PASS: Tripod remained neutral."
            )

            print(
                "[RAVEN] Initial hand/handle geometry "
                "appears collision-free."
            )

        print()

    # -------------------------------------------------------------------------
    # Print clean initial measurement.
    # -------------------------------------------------------------------------

    print_geometry()

    # =========================================================================
    # Keep window alive
    # =========================================================================

    while simulation_app.is_running():

        # ---------------------------------------------------------------------
        # Freeze the entire G1 configuration.
        #
        # This prevents the actuator-free articulation from drifting or being
        # pushed around during this measurement-only test.
        # ---------------------------------------------------------------------

        robot.write_joint_state_to_sim(
            frozen_robot_joint_pos,
            frozen_robot_joint_vel,
        )

        robot.write_data_to_sim()

        # Tripod remains physically passive.
        tripod.write_data_to_sim()

        sim.step()

        robot.update(sim_dt)
        tripod.update(sim_dt)


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()