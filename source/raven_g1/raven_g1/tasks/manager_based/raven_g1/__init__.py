import gymnasium as gym

from . import agents


gym.register(
    id="Raven-G1-FollowTarget-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.raven_g1_env_cfg:RavenG1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RavenG1PPORunnerCfg",
    },
)


gym.register(
    id="Raven-G1-FollowTarget-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.raven_g1_env_cfg:RavenG1EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RavenG1PPORunnerCfg",
    },
)