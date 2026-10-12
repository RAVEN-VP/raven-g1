"""RAVEN Unitree G1 29-DOF Mode 15 + BrainCo Revo2 asset configuration."""

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg


# =============================================================================
# Asset path
# =============================================================================

# This file lives at:
#
# raven_g1/
#   source/
#     raven_g1/
#       raven_g1/
#         assets/
#           g1_brainco_mode15_cfg.py
#
# parents[4] therefore resolves to the RAVEN project root:
#
# D:\RAVEN-Isaac\raven_g1

_PROJECT_ROOT = Path(__file__).resolve().parents[4]

G1_BRAINCO_MODE15_URDF = (
    _PROJECT_ROOT
    / "external"
    / "unitree_ros"
    / "robots"
    / "g1_with_brainco_hand"
    / "g1_29dof_mode_15_brainco_hand_isaac.urdf"
)


# =============================================================================
# G1 29-DOF Mode 15 + BrainCo Revo2
# =============================================================================

G1_BRAINCO_MODE15_CFG = ArticulationCfg(

    prim_path="{ENV_REGEX_NS}/Robot",

    # -------------------------------------------------------------------------
    # URDF import
    # -------------------------------------------------------------------------

    spawn=sim_utils.UrdfFileCfg(

        asset_path=str(
            G1_BRAINCO_MODE15_URDF
        ),

        # Temporarily fixed while we develop tripod manipulation.
        #
        # Later we will restore the floating base when whole-body balance
        # becomes part of the task.
        fix_base=True,

        root_link_name="pelvis",

        # Preserve the fixed BrainCo mounting structure and sensor links.
        merge_fixed_joints=False,

        # Always regenerate USD from our Isaac-compatible derived URDF.
        force_usd_conversion=True,

        self_collision=False,

        # ---------------------------------------------------------------------
        # Do not create automatic position drives from the URDF.
        #
        # We will explicitly configure the G1 and Revo2 actuators once the
        # geometry/manipulation chain is validated.
        # ---------------------------------------------------------------------

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(

            drive_type="force",

            target_type="none",

            gains=(
                sim_utils.UrdfConverterCfg
                .JointDriveCfg
                .PDGainsCfg(
                    stiffness=0.0,
                    damping=0.0,
                )
            ),
        ),

        rigid_props=sim_utils.RigidBodyPropertiesCfg(

            rigid_body_enabled=True,

            # Root is temporarily fixed and this stage is concerned with
            # kinematic/manipulation geometry.
            disable_gravity=True,

            max_depenetration_velocity=1.0,

            enable_gyroscopic_forces=True,
        ),

        articulation_props=(
            sim_utils.ArticulationRootPropertiesCfg(

                enabled_self_collisions=False,

                solver_position_iteration_count=8,

                solver_velocity_iteration_count=4,

                sleep_threshold=0.005,

                stabilization_threshold=0.001,
            )
        ),
    ),

    # -------------------------------------------------------------------------
    # Initial pose
    # -------------------------------------------------------------------------

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(
            0.0,
            0.0,
            0.80,
        ),

        rot=(
            1.0,
            0.0,
            0.0,
            0.0,
        ),

        # Any joint not listed here defaults to zero.
        #
        # That includes:
        #   - waist joints
        #   - wrist pitch/yaw
        #   - BrainCo Revo2 finger joints
        #
        # The locked fingertip joints no longer appear here because we already
        # converted those zero-range revolute joints into fixed joints in the
        # derived "_isaac.urdf".
        joint_pos={

            # -------------------------------------------------------------
            # Legs
            # -------------------------------------------------------------

            "left_hip_pitch_joint": -0.10,
            "right_hip_pitch_joint": -0.10,

            "left_knee_joint": 0.30,
            "right_knee_joint": 0.30,

            "left_ankle_pitch_joint": -0.20,
            "right_ankle_pitch_joint": -0.20,

            # -------------------------------------------------------------
            # Arms
            # -------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Actuators
    # -------------------------------------------------------------------------
    #
    # Intentionally empty at this stage.
    #
    # We have validated that the imported articulation contains:
    #
    #   29 G1 joints
    # + 11 left Revo2 joints
    # + 11 right Revo2 joints
    # = 51 total joints
    #
    # Proper actuator groups will be added after the next geometry test.
    # -------------------------------------------------------------------------

    actuators={},
)