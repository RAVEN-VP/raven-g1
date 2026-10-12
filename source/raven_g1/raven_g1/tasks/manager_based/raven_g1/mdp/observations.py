from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Camera-link state
# -----------------------------------------------------------------------------


def _camera_state_w(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
):
    """Return physical camera-link state in world coordinates."""

    tripod_camera = env.scene[
        camera_cfg.name
    ]

    camera_body_id = (
        camera_cfg.body_ids[0]
    )

    camera_position_w = (
        tripod_camera.data.body_pos_w[
            :,
            camera_body_id,
        ]
    )

    camera_quat_w = (
        tripod_camera.data.body_quat_w[
            :,
            camera_body_id,
        ]
    )

    camera_velocity_w = (
        tripod_camera.data.body_lin_vel_w[
            :,
            camera_body_id,
        ]
    )

    return (
        camera_position_w,
        camera_quat_w,
        camera_velocity_w,
    )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _target_position_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    target = env.scene[
        target_cfg.name
    ]

    (
        camera_position_w,
        camera_quat_w,
        _,
    ) = _camera_state_w(
        env,
        camera_cfg,
    )

    target_delta_w = (
        target.data.root_pos_w
        - camera_position_w
    )

    return quat_apply_inverse(
        camera_quat_w,
        target_delta_w,
    )


def _target_relative_velocity_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    target = env.scene[
        target_cfg.name
    ]

    (
        _,
        camera_quat_w,
        camera_velocity_w,
    ) = _camera_state_w(
        env,
        camera_cfg,
    )

    relative_velocity_w = (
        target.data.root_lin_vel_w
        - camera_velocity_w
    )

    return quat_apply_inverse(
        camera_quat_w,
        relative_velocity_w,
    )


# -----------------------------------------------------------------------------
# RAVEN observations
# -----------------------------------------------------------------------------


def target_position_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    return _target_position_camera(
        env,
        camera_cfg,
        target_cfg,
    )


def target_relative_velocity_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    return _target_relative_velocity_camera(
        env,
        camera_cfg,
        target_cfg,
    )


def target_distance_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    position_camera = (
        _target_position_camera(
            env,
            camera_cfg,
            target_cfg,
        )
    )

    return torch.linalg.vector_norm(
        position_camera,
        dim=-1,
        keepdim=True,
    )


def target_distance_error_camera(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    desired_distance: float = 2.0,
) -> torch.Tensor:

    distance = target_distance_camera(
        env,
        camera_cfg,
        target_cfg,
    )

    return (
        distance
        - desired_distance
    )


def target_azimuth(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    position_camera = (
        _target_position_camera(
            env,
            camera_cfg,
            target_cfg,
        )
    )

    azimuth = torch.atan2(
        position_camera[:, 1],
        position_camera[:, 0],
    )

    return azimuth.unsqueeze(-1)


def target_azimuth_sin_cos(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    azimuth = target_azimuth(
        env,
        camera_cfg,
        target_cfg,
    )

    return torch.cat(
        (
            torch.sin(azimuth),

            torch.cos(azimuth),
        ),
        dim=-1,
    )


def target_elevation(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    position_camera = (
        _target_position_camera(
            env,
            camera_cfg,
            target_cfg,
        )
    )

    horizontal_distance = torch.sqrt(

        position_camera[:, 0].square()

        + position_camera[:, 1].square()
    )

    elevation = torch.atan2(

        position_camera[:, 2],

        horizontal_distance,
    )

    return elevation.unsqueeze(-1)


def target_elevation_sin_cos(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:

    elevation = target_elevation(
        env,
        camera_cfg,
        target_cfg,
    )

    return torch.cat(
        (
            torch.sin(elevation),

            torch.cos(elevation),
        ),
        dim=-1,
    )