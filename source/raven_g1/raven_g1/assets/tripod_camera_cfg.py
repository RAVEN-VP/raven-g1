from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_ASSET_DIR = Path(__file__).resolve().parent

TRIPOD_CAMERA_URDF = (
    _ASSET_DIR
    / "tripod_camera"
    / "tripod_camera.urdf"
)


# -----------------------------------------------------------------------------
# RAVEN tripod / camera
# -----------------------------------------------------------------------------

TRIPOD_CAMERA_CFG = ArticulationCfg(

    prim_path="{ENV_REGEX_NS}/TripodCamera",

    spawn=sim_utils.UrdfFileCfg(

        asset_path=str(TRIPOD_CAMERA_URDF),

        # -------------------------------------------------------------
        # Tripod base is fixed to the world.
        # -------------------------------------------------------------

        fix_base=True,

        root_link_name="base_link",

        # Keep the camera and handle as separate rigid bodies.
        merge_fixed_joints=False,

        # Rebuild USD whenever we modify the URDF.
        force_usd_conversion=True,

        self_collision=False,

        # -------------------------------------------------------------
        # URDF IMPORTER JOINT DRIVE
        #
        # IMPORTANT:
        #
        # The URDF converter requires explicit PD gains.
        #
        # We set both to zero here because the tripod pan/tilt joints
        # should not contain an automatic position servo.
        #
        # The Articulation actuator configuration below will provide
        # the small passive damping we actually want.
        # -------------------------------------------------------------

        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(

            drive_type="force",

            target_type="position",

            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(

                stiffness=0.0,

                damping=0.0,
            ),
        ),

        # -------------------------------------------------------------
        # Rigid-body properties
        # -------------------------------------------------------------

        rigid_props=sim_utils.RigidBodyPropertiesCfg(

            rigid_body_enabled=True,

            disable_gravity=True,

            max_depenetration_velocity=1.0,

            enable_gyroscopic_forces=True,
        ),

        # -------------------------------------------------------------
        # Articulation properties
        # -------------------------------------------------------------

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(

            enabled_self_collisions=False,

            solver_position_iteration_count=8,

            solver_velocity_iteration_count=4,

            sleep_threshold=0.005,

            stabilization_threshold=0.001,
        ),
    ),

    # -----------------------------------------------------------------
    # Starting transform
    #
    # G1 root:
    #     X = 0
    #     Y = 0
    #
    # Tripod:
    #     0.50 m in front of G1
    # -----------------------------------------------------------------

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(
            0.50,
            0.0,
            0.0,
        ),

        rot=(
            1.0,
            0.0,
            0.0,
            0.0,
        ),

        joint_pos={

            "pan_joint": 0.0,

            "tilt_joint": 0.0,
        },

        joint_vel={

            "pan_joint": 0.0,

            "tilt_joint": 0.0,
        },
    ),

    # -----------------------------------------------------------------
    # Passive tripod head
    #
    # Pan and tilt ARE FREE TO ROTATE.
    #
    # stiffness = 0:
    #     no position servo attempts to hold a particular angle.
    #
    # damping = 0.25:
    #     small resistance, similar to a damped tripod head.
    #
    # Eventually G1's hand/contact force will physically move these
    # joints.
    # -----------------------------------------------------------------

    actuators={

        "tripod_head": ImplicitActuatorCfg(

            joint_names_expr=[

                "pan_joint",

                "tilt_joint",
            ],

            effort_limit_sim=20.0,

            velocity_limit_sim=1.5,

            stiffness=0.0,

            damping=0.25,

            friction=0.10,
        ),
    },
)