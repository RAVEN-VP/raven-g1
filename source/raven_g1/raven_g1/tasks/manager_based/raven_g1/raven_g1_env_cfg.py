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
    """RAVEN G1 cinematography scene."""

    # ------------------------------------------------------------------
    # Moving actor / subject
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
    # Tripod base
    #
    # 34 cm footprint. This is compact enough to sit in front of the
    # 45 cm-wide G1 without being absurdly large compared with it.
    # ------------------------------------------------------------------

    tripod_base = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TripodBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.34, 0.34, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.08, 0.10),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.38, 0.0, 0.03),
        ),
    )

    # ------------------------------------------------------------------
    # Tripod / pedestal column
    #
    # 82 cm column.
    #
    # Camera head is initially around 1.05 m, placing the controls and
    # camera around the G1's torso/arm working height.
    # ------------------------------------------------------------------

    tripod_column = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TripodColumn",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.82),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.12, 0.14),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.38, 0.0, 0.44),
        ),
    )

    # ------------------------------------------------------------------
    # Camera head / optical tracking frame
    #
    # The root transform of this object is currently treated as the
    # RAVEN camera tracking frame.
    #
    # Size:
    #   24 cm long
    #   16 cm wide
    #   14 cm high
    #
    # Initial height:
    #   1.05 m
    #
    # Initial horizontal offset from G1:
    #   0.38 m forward
    #
    # This keeps it comfortably inside the conservative G1 manipulation
    # envelope used by camera_control.py.
    # ------------------------------------------------------------------

    camera_rig = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CameraRig",
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.16, 0.14),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.20, 0.23),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.38, 0.0, 1.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


# -----------------------------------------------------------------------------
# Camera-centric RAVEN observations
# -----------------------------------------------------------------------------


@configclass
class RavenObservationsCfg(ObservationsCfg):

    @configclass
    class RavenCfg(ObsGroup):

        # Target XYZ in camera frame.
        # Shape: 3
        target_position_camera = ObsTerm(
            func=mdp.target_position_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Target velocity XYZ relative to camera.
        # Shape: 3
        target_relative_velocity_camera = ObsTerm(
            func=mdp.target_relative_velocity_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Full 3D shot distance.
        # Shape: 1
        target_distance_camera = ObsTerm(
            func=mdp.target_distance_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Desired shot distance error.
        # Shape: 1
        target_distance_error_camera = ObsTerm(
            func=mdp.target_distance_error_camera,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
                "desired_distance": 2.0,
            },
        )

        # Horizontal framing error.
        # Shape: 2
        target_azimuth_sin_cos = ObsTerm(
            func=mdp.target_azimuth_sin_cos,
            params={
                "camera_rig_cfg": SceneEntityCfg("camera_rig"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )

        # Vertical framing error.
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