from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def move_target_on_circle(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    radius: float = 2.5,
    angular_speed: float = 0.4,
    height: float = 1.0,
    step_dt: float = 0.02,
):
    """Move the RAVEN target around each environment on a circular trajectory."""

    target = env.scene[asset_cfg.name]

    # A global interval event passes env_ids=None.
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Create persistent phase state the first time this function runs.
    if not hasattr(env, "_raven_target_phase"):
        env._raven_target_phase = torch.zeros(
            env.scene.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    env._raven_target_phase[env_ids] += angular_speed * step_dt

    phase = env._raven_target_phase[env_ids]
    origins = env.scene.env_origins[env_ids]

    # Start from the target's current pose so that we preserve a valid quaternion.
    root_pose = target.data.root_state_w[env_ids, :7].clone()

    root_pose[:, 0] = origins[:, 0] + radius * torch.cos(phase)
    root_pose[:, 1] = origins[:, 1] + radius * torch.sin(phase)
    root_pose[:, 2] = origins[:, 2] + height

    # Keep orientation fixed.
    root_pose[:, 3] = 1.0
    root_pose[:, 4:] = 0.0

    target.write_root_pose_to_sim(root_pose, env_ids=env_ids)