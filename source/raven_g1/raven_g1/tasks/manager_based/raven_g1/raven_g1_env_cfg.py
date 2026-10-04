import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    EventCfg,
    MySceneCfg,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg,
    G1FlatEnvCfg_PLAY,
)

from . import mdp


@configclass
class RavenSceneCfg(MySceneCfg):
    """G1 locomotion scene with a RAVEN actor target."""

    # Target representing the actor/person being followed.
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
            pos=(2.0, 0.0, 1.0),
        ),
    )


@configclass
class RavenEventCfg(EventCfg):
    """Events specific to the RAVEN environment."""

    # Move the target continuously around the G1.
    move_target = EventTerm(
        func=mdp.move_target_on_circle,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "radius": 2.5,
            "angular_speed": 0.4,
            "height": 1.0,
            "step_dt": 0.02,
        },
    )


@configclass
class RavenG1EnvCfg(G1FlatEnvCfg):
    """RAVEN G1 training environment."""

    scene: RavenSceneCfg = RavenSceneCfg(
        num_envs=256,
        env_spacing=6.0,
    )

    events: RavenEventCfg = RavenEventCfg()

    def __post_init__(self):
        super().__post_init__()


@configclass
class RavenG1EnvCfg_PLAY(G1FlatEnvCfg_PLAY):
    """Single-environment RAVEN visual validation environment."""

    scene: RavenSceneCfg = RavenSceneCfg(
        num_envs=1,
        env_spacing=6.0,
    )

    events: RavenEventCfg = RavenEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 6.0