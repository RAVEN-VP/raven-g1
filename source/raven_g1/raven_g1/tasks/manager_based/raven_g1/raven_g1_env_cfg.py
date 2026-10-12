import isaaclab.sim as sim_utils

from isaaclab.assets import (
    ArticulationCfg,
    RigidObjectCfg,
)

from isaaclab.managers import (
    EventTermCfg as EventTerm,
)

from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)

from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)

from isaaclab.managers import (
    SceneEntityCfg,
)

from isaaclab.utils import (
    configclass,
)

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from raven_g1.assets import (
    TRIPOD_CAMERA_CFG,
)

from . import mdp


# =============================================================================
# Scene
# =============================================================================


@configclass
class RavenSceneCfg(MySceneCfg):
    """Stationary G1 + fixed tripod + rotating camera head."""

    # -----------------------------------------------------------------
    # Moving target / actor
    # -----------------------------------------------------------------

    target = RigidObjectCfg(

        prim_path="{ENV_REGEX_NS}/Target",

        spawn=sim_utils.SphereCfg(

            radius=0.18,

            rigid_props=sim_utils.RigidBodyPropertiesCfg(

                kinematic_enabled=True,

                disable_gravity=True,
            ),

            visual_material=sim_utils.PreviewSurfaceCfg(

                diffuse_color=(
                    1.0,
                    0.15,
                    0.05,
                ),
            ),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(

            pos=(
                2.5,
                0.0,
                1.2,
            ),
        ),
    )

    # -----------------------------------------------------------------
    # Fixed tripod with pan + tilt articulation
    # -----------------------------------------------------------------

    tripod_camera: ArticulationCfg = (
        TRIPOD_CAMERA_CFG
    )


# =============================================================================
# Observations
# =============================================================================


@configclass
class RavenObservationsCfg(
    ObservationsCfg
):

    @configclass
    class RavenCfg(ObsGroup):

        # -------------------------------------------------------------
        # Target XYZ relative to current CAMERA orientation.
        #
        # Shape: 3
        # -------------------------------------------------------------

        target_position_camera = ObsTerm(

            func=mdp.target_position_camera,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),
            },
        )

        # -------------------------------------------------------------
        # Target velocity XYZ relative to camera.
        #
        # Shape: 3
        # -------------------------------------------------------------

        target_relative_velocity_camera = ObsTerm(

            func=mdp.target_relative_velocity_camera,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),
            },
        )

        # -------------------------------------------------------------
        # 3D distance.
        #
        # Shape: 1
        # -------------------------------------------------------------

        target_distance_camera = ObsTerm(

            func=mdp.target_distance_camera,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),
            },
        )

        # -------------------------------------------------------------
        # Distance metric.
        #
        # Shape: 1
        #
        # Useful as experimental context although the fixed tripod
        # cannot correct distance by translating.
        # -------------------------------------------------------------

        target_distance_error_camera = ObsTerm(

            func=mdp.target_distance_error_camera,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),

                "desired_distance": 2.0,
            },
        )

        # -------------------------------------------------------------
        # Horizontal framing error.
        #
        # Shape: 2
        # -------------------------------------------------------------

        target_azimuth_sin_cos = ObsTerm(

            func=mdp.target_azimuth_sin_cos,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),
            },
        )

        # -------------------------------------------------------------
        # Vertical framing error.
        #
        # Shape: 2
        # -------------------------------------------------------------

        target_elevation_sin_cos = ObsTerm(

            func=mdp.target_elevation_sin_cos,

            params={

                "camera_cfg": SceneEntityCfg(

                    "tripod_camera",

                    body_names=[
                        "camera_link"
                    ],
                ),

                "target_cfg": SceneEntityCfg(
                    "target"
                ),
            },
        )

        def __post_init__(self):

            self.enable_corruption = False

            self.concatenate_terms = True


    raven: RavenCfg = RavenCfg()


# =============================================================================
# Target motion
# =============================================================================


@configclass
class RavenEventCfg(EventCfg):

    move_target = EventTerm(

        func=mdp.move_target_on_3d_path,

        mode="interval",

        interval_range_s=(
            0.02,
            0.02,
        ),

        is_global_time=True,

        params={

            "asset_cfg": SceneEntityCfg(
                "target"
            ),

            "radius": 2.5,

            "angular_speed": 0.4,

            "base_height": 1.2,

            "vertical_amplitude": 0.5,

            "vertical_angular_speed": 0.6,

            "step_dt": 0.02,
        },
    )


# =============================================================================
# Training environment
# =============================================================================


@configclass
class RavenG1EnvCfg(
    G1FlatEnvCfg
):

    scene: RavenSceneCfg = RavenSceneCfg(

        num_envs=256,

        env_spacing=6.0,
    )

    observations: RavenObservationsCfg = (
        RavenObservationsCfg()
    )

    events: RavenEventCfg = (
        RavenEventCfg()
    )

    def __post_init__(self):

        super().__post_init__()


# =============================================================================
# Play environment
# =============================================================================


@configclass
class RavenG1EnvCfg_PLAY(
    G1FlatEnvCfg_PLAY
):

    scene: RavenSceneCfg = RavenSceneCfg(

        num_envs=1,

        env_spacing=6.0,
    )

    observations: RavenObservationsCfg = (
        RavenObservationsCfg()
    )

    events: RavenEventCfg = (
        RavenEventCfg()
    )

    def __post_init__(self):

        super().__post_init__()

        self.scene.num_envs = 1

        self.scene.env_spacing = 6.0