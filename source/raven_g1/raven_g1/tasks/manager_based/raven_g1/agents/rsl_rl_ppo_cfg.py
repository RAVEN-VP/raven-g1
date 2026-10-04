from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatPPORunnerCfg,
)


@configclass
class RavenG1PPORunnerCfg(G1FlatPPORunnerCfg):
    """RSL-RL PPO configuration for RAVEN G1."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "raven_g1_follow_target"