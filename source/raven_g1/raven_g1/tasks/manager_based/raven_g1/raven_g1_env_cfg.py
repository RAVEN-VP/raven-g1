import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from . import mdp


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


@configclass
class RavenSceneCfg(MySceneCfg):
    """RAVEN scene containing G1, target and camera/tripod proxy."""

    # ------------------------------------------------------------------
    # Actor / target
    # ------------------------------------------------------------------

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.18,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.15, 0.05),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(2.5, 0.0, 1.2),
        ),
    )

    # ------------------------------------------------------------------
    # Camera / tripod-head proxy
    #
    # This is deliberately simple for now.
    #
    # Its root transform represents the camera optical tracking frame:
    #
    # +X = forward
    # +Y = lateral
    # +Z = vertical
    #
    # We will replace this with the articulated tripod/camera system
    # before training the final manipulation policy.
    # ------------------------------------------------------------------

    camera_rig = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig",
        spawn=sim_utils.CuboidCfg(
            size=(0.30, 0.20, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.18),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -1.0, 1.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Camera-centric RAVEN observations
# -----------------------------------------------------------------------------


@configclass
class RavenObservationsCfg(ObservationsCfg):
    """Low-level G1 observations plus high-level camera-tracking observations."""

    @configclass
    class RavenCfg(ObsGroup):
        """Observations used by the future RAVEN camera policy."""

        # Full XYZ target position in camera coordinates.
        # Shape: 3
        target_position_camera = ObsTerm(
            func=mdp.target_position_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Full XYZ target velocity relative to the camera.
        # Shape: 3
        target_relative_velocity_camera = ObsTerm(
            func=mdp.target_relative_velocity_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Full 3D camera-target distance.
        # Shape: 1
        target_distance_camera = ObsTerm(
            func=mdp.target_distance_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Difference from desired shot distance.
        # Shape: 1
        target_distance_error_camera = ObsTerm(
            func=mdp.target_distance_error_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
                "desired_distance": 2.0,
            },
        )

        # Horizontal target angle.
        # Shape: 2
        target_azimuth_sin_cos = ObsTerm(
            func=mdp.target_azimuth_sin_cos,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Vertical target angle.
        # Shape: 2
        target_elevation_sin_cos = ObsTerm(
            func=mdp.target_elevation_sin_cos,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    raven: RavenCfg = RavenCfg()


# -----------------------------------------------------------------------------
# Events
# -----------------------------------------------------------------------------


@configclass
class RavenEventCfg(EventCfg):
    """RAVEN environment events."""

    move_target = EventTerm(
        func=mdp.move_target_on_3d_path,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("target"),

            "radius": 2.5,
            "angular_speed": 0.4,

            "base_height": 1.2,
            "vertical_amplitude": 0.5,
            "vertical_angular_speed": 0.6,

            "step_dt": 0.02,
        },
    )


# -----------------------------------------------------------------------------
# Training environment
# -----------------------------------------------------------------------------


@configclass
class RavenG1EnvCfg(G1FlatEnvCfg):
    """RAVEN G1 training environment."""

    scene: RavenSceneCfg = RavenSceneCfg(
        num_envs=256,
        env_spacing=6.0,
    )

    observations: RavenObservationsCfg = RavenObservationsCfg()

    events: RavenEventCfg = RavenEventCfg()

    def __post_init__(self):
        super().__post_init__()


# -----------------------------------------------------------------------------
# Play environment
# -----------------------------------------------------------------------------


@configclass
class RavenG1EnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Single-environment RAVEN validation environment."""

    scene: RavenSceneCfg = RavenSceneCfg(
        num_envs=1,
        env_spacing=6.0,
    )

    observations: RavenObservationsCfg = RavenObservationsCfg()

    events: RavenEventCfg = RavenEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 6.0