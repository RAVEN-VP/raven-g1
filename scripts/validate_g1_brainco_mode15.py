"""Validate the official Unitree G1 29-DOF Mode 15 + BrainCo hand URDF.

This does NOT modify the RAVEN environment.

Purpose:
    1. Verify Isaac Sim can import the combined official URDF.
    2. Verify all mesh paths resolve.
    3. Print the exact articulation body and joint names.
    4. Identify the BrainCo hand base/finger links.
    5. Confirm the 29-DOF Unitree wrist chain is present.

The robot is fixed and gravity-disabled for this validation only.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


# =============================================================================
# Launch Isaac Sim
# =============================================================================

parser = argparse.ArgumentParser(
    description="Validate official G1 Mode 15 + BrainCo Revo2 asset."
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# =============================================================================
# Isaac Lab imports
# =============================================================================

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation, ArticulationCfg


# =============================================================================
# Asset path
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

G1_BRAINCO_URDF = (
    PROJECT_ROOT
    / "external"
    / "unitree_ros"
    / "robots"
    / "g1_with_brainco_hand"
    / "g1_29dof_mode_15_brainco_hand_isaac.urdf"
)


# =============================================================================
# Robot configuration
# =============================================================================

G1_BRAINCO_CFG = ArticulationCfg(

    prim_path="/World/G1BrainCo",

    spawn=sim_utils.UrdfFileCfg(

        asset_path=str(G1_BRAINCO_URDF),

        # Fixed only for this asset-validation stage.
        fix_base=True,

        # Keep fixed links because we want to inspect the actual
        # BrainCo mounting structure.
        merge_fixed_joints=False,

        force_usd_conversion=True,

        self_collision=False,

        # No automatic drives during this validation.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(

            drive_type="force",

            target_type="none",

            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),

    # Pelvis height close to the official Unitree standing configuration.
    init_state=ArticulationCfg.InitialStateCfg(

        pos=(0.0, 0.0, 0.80),

        rot=(1.0, 0.0, 0.0, 0.0),

        joint_pos={

            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,

            "left_knee_joint": 0.30,
            "right_knee_joint": 0.30,

            "left_ankle_pitch_joint": -0.20,
            "right_ankle_pitch_joint": -0.20,

            "left_shoulder_pitch_joint": 0.30,
            "right_shoulder_pitch_joint": 0.30,

            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,

            "left_elbow_joint": 0.97,
            "right_elbow_joint": 0.97,

            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },

        joint_vel={
            ".*": 0.0,
        },
    ),

    # Passive for the validation test.
    actuators={},
)


# =============================================================================
# Main
# =============================================================================

def main():

    if not G1_BRAINCO_URDF.exists():

        raise FileNotFoundError(
            "Could not find official G1 + BrainCo URDF:\n"
            f"{G1_BRAINCO_URDF}"
        )

    print()
    print("============================================================")
    print(" RAVEN - G1 MODE 15 + BRAINCO ASSET VALIDATION")
    print("============================================================")
    print()
    print(f"[RAVEN] URDF:")
    print(f"        {G1_BRAINCO_URDF}")
    print()

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
    # Ground + lighting
    # -------------------------------------------------------------------------

    sim_utils.spawn_ground_plane(
        "/World/Ground",
        sim_utils.GroundPlaneCfg(),
    )

    light_cfg = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )

    light_cfg.func(
        "/World/Light",
        light_cfg,
        translation=(1.0, 0.0, 10.0),
    )

    # -------------------------------------------------------------------------
    # Spawn robot
    # -------------------------------------------------------------------------

    robot = Articulation(
        G1_BRAINCO_CFG
    )

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------

    sim.set_camera_view(
        eye=(2.8, 2.8, 2.1),
        target=(0.0, 0.0, 0.9),
    )

    # -------------------------------------------------------------------------
    # Start simulation
    # -------------------------------------------------------------------------

    sim.reset()

    # -------------------------------------------------------------------------
    # Articulation summary
    # -------------------------------------------------------------------------

    print()
    print("============================================================")
    print(" ARTICULATION SUMMARY")
    print("============================================================")
    print()

    print(
        f"[RAVEN] Fixed base : {robot.is_fixed_base}"
    )

    print(
        f"[RAVEN] Joint count: {robot.num_joints}"
    )

    print(
        f"[RAVEN] Body count : {robot.num_bodies}"
    )

    # -------------------------------------------------------------------------
    # Unitree joints
    # -------------------------------------------------------------------------

    print()
    print("------------------------------------------------------------")
    print(" ALL JOINTS")
    print("------------------------------------------------------------")

    for index, joint_name in enumerate(
        robot.joint_names
    ):

        print(
            f"{index:3d} : {joint_name}"
        )

    # -------------------------------------------------------------------------
    # Bodies
    # -------------------------------------------------------------------------

    print()
    print("------------------------------------------------------------")
    print(" ALL BODIES")
    print("------------------------------------------------------------")

    for index, body_name in enumerate(
        robot.body_names
    ):

        print(
            f"{index:3d} : {body_name}"
        )

    # -------------------------------------------------------------------------
    # Specifically inspect the 29-DOF wrist chains.
    # -------------------------------------------------------------------------

    print()
    print("------------------------------------------------------------")
    print(" WRIST JOINTS")
    print("------------------------------------------------------------")

    wrist_joint_names = [

        name

        for name in robot.joint_names

        if "wrist" in name.lower()
    ]

    for name in wrist_joint_names:

        print(
            f"[WRIST] {name}"
        )

    # -------------------------------------------------------------------------
    # BrainCo joints
    # -------------------------------------------------------------------------

    print()
    print("------------------------------------------------------------")
    print(" BRAINCO / HAND JOINTS")
    print("------------------------------------------------------------")

    hand_joint_keywords = (
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    )

    hand_joint_names = [

        name

        for name in robot.joint_names

        if any(
            keyword in name.lower()
            for keyword in hand_joint_keywords
        )
    ]

    for name in hand_joint_names:

        print(
            f"[HAND JOINT] {name}"
        )

    # -------------------------------------------------------------------------
    # BrainCo links
    # -------------------------------------------------------------------------

    print()
    print("------------------------------------------------------------")
    print(" BRAINCO / HAND BODIES")
    print("------------------------------------------------------------")

    hand_body_keywords = (
        "base",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
    )

    hand_body_names = [

        name

        for name in robot.body_names

        if any(
            keyword in name.lower()
            for keyword in hand_body_keywords
        )
    ]

    for name in hand_body_names:

        print(
            f"[HAND BODY] {name}"
        )

    # -------------------------------------------------------------------------
    # Explicit checks important to RAVEN.
    # -------------------------------------------------------------------------

    expected_joints = [

        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",

        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    expected_bodies = [

        "left_wrist_yaw_link",
        "right_wrist_yaw_link",

        "left_base_link",
        "right_base_link",
    ]

    print()
    print("------------------------------------------------------------")
    print(" RAVEN REQUIRED LINK/JOINТ CHECK")
    print("------------------------------------------------------------")

    success = True

    for name in expected_joints:

        exists = name in robot.joint_names

        success &= exists

        print(
            f"[{'OK' if exists else 'MISSING'}] "
            f"joint: {name}"
        )

    for name in expected_bodies:

        exists = name in robot.body_names

        success &= exists

        print(
            f"[{'OK' if exists else 'MISSING'}] "
            f"body : {name}"
        )

    print()

    if success:

        print(
            "[RAVEN] PASS: Mode 15 wrist + BrainCo "
            "mounting structure detected."
        )

    else:

        print(
            "[RAVEN] WARNING: One or more expected "
            "Mode 15 / BrainCo elements were not imported."
        )

    print()
    print(
        "[RAVEN] Leave the window running so the "
        "robot geometry can be inspected visually."
    )
    print()

    # -------------------------------------------------------------------------
    # Simulation loop
    # -------------------------------------------------------------------------

    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():

        robot.write_data_to_sim()

        sim.step()

        robot.update(
            sim_dt
        )


if __name__ == "__main__":

    main()

    simulation_app.close()