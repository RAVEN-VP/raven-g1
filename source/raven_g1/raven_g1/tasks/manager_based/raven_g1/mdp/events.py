from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def move_target_on_3d_path(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg,
    radius: float = 2.5,
    angular_speed: float = 0.4,
    base_height: float = 1.2,
    vertical_amplitude: float = 0.5,
    vertical_angular_speed: float = 0.6,
    step_dt: float = 0.02,
):
    """Move the RAVEN target along a continuous 3D trajectory."""

    target = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(
            env.scene.num_envs,
            device=env.device,
        )

    # ------------------------------------------------------------------
    # Persistent trajectory phases
    # ------------------------------------------------------------------

    if not hasattr(env, "_raven_target_phase_xy"):
        env._raven_target_phase_xy = torch.zeros(
            env.scene.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    if not hasattr(env, "_raven_target_phase_z"):
        env._raven_target_phase_z = torch.zeros(
            env.scene.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    env._raven_target_phase_xy[env_ids] += (
        angular_speed * step_dt
    )

    env._raven_target_phase_z[env_ids] += (
        vertical_angular_speed * step_dt
    )

    phase_xy = env._raven_target_phase_xy[env_ids]
    phase_z = env._raven_target_phase_z[env_ids]

    origins = env.scene.env_origins[env_ids]

    root_state = target.data.root_state_w[env_ids].clone()

    # ------------------------------------------------------------------
    # 3D target position
    # ------------------------------------------------------------------

    root_state[:, 0] = (
        origins[:, 0]
        + radius * torch.cos(phase_xy)
    )

    root_state[:, 1] = (
        origins[:, 1]
        + radius * torch.sin(phase_xy)
    )

    root_state[:, 2] = (
        origins[:, 2]
        + base_height
        + vertical_amplitude * torch.sin(phase_z)
    )

    # Keep target orientation fixed.
    root_state[:, 3] = 1.0
    root_state[:, 4:7] = 0.0

    # ------------------------------------------------------------------
    # Analytic 3D target velocity
    # ------------------------------------------------------------------

    root_state[:, 7] = (
        -radius
        * angular_speed
        * torch.sin(phase_xy)
    )

    root_state[:, 8] = (
        radius
        * angular_speed
        * torch.cos(phase_xy)
    )

    root_state[:, 9] = (
        vertical_amplitude
        * vertical_angular_speed
        * torch.cos(phase_z)
    )

    # Target does not rotate.
    root_state[:, 10:13] = 0.0

    target.write_root_state_to_sim(
        root_state,
        env_ids=env_ids,
    )