from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _target_position_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return target position relative to the camera frame in full 3D.

    Camera-frame convention used by RAVEN:

        +X = forward / depth
        +Y = left-right
        +Z = up-down
    """

    camera_rig = env.scene[camera_rig_cfg.name]
    target = env.scene[target_cfg.name]

    # World-space vector from camera to target.
    target_delta_w = (
        target.data.root_pos_w
        - camera_rig.data.root_pos_w
    )

    # Convert the world-space vector into the camera's local frame.
    target_delta_camera = quat_apply_inverse(
        camera_rig.data.root_quat_w,
        target_delta_w,
    )

    return target_delta_camera


def _target_relative_velocity_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return target velocity relative to the camera in full 3D."""

    camera_rig = env.scene[camera_rig_cfg.name]
    target = env.scene[target_cfg.name]

    relative_velocity_w = (
        target.data.root_lin_vel_w
        - camera_rig.data.root_lin_vel_w
    )

    relative_velocity_camera = quat_apply_inverse(
        camera_rig.data.root_quat_w,
        relative_velocity_w,
    )

    return relative_velocity_camera


# -----------------------------------------------------------------------------
# RAVEN camera-tracking observations
# -----------------------------------------------------------------------------


def target_position_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target position in camera frame: [forward, lateral, vertical]."""

    return _target_position_camera(
        env,
        camera_rig_cfg,
        target_cfg,
    )


def target_relative_velocity_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Target velocity relative to camera: [vx, vy, vz]."""

    return _target_relative_velocity_camera(
        env,
        camera_rig_cfg,
        target_cfg,
    )


def target_distance_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """3D distance from camera to target."""

    position_camera = _target_position_camera(
        env,
        camera_rig_cfg,
        target_cfg,
    )

    return torch.linalg.vector_norm(
        position_camera,
        dim=-1,
        keepdim=True,
    )


def target_distance_error_camera(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    desired_distance: float = 2.0,
) -> torch.Tensor:
    """Signed camera-to-target distance error.

    Positive = camera is too far away.
    Negative = camera is too close.
    """

    distance = target_distance_camera(
        env,
        camera_rig_cfg,
        target_cfg,
    )

    return distance - desired_distance


def target_azimuth(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Horizontal target angle relative to camera forward."""

    position_camera = _target_position_camera(
        env,
        camera_rig_cfg,
        target_cfg,
    )

    azimuth = torch.atan2(
        position_camera[:, 1],
        position_camera[:, 0],
    )

    return azimuth.unsqueeze(-1)


def target_azimuth_sin_cos(
    env: ManagerBasedRLEnv,
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Continuous horizontal framing angle representation."""

    azimuth = target_azimuth(
        env,
        camera_rig_cfg,
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
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Vertical target angle relative to camera forward."""

    position_camera = _target_position_camera(
        env,
        camera_rig_cfg,
        target_cfg,
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
    camera_rig_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Continuous vertical framing angle representation."""

    elevation = target_elevation(
        env,
        camera_rig_cfg,
        target_cfg,
    )

    return torch.cat(
        (
            torch.sin(elevation),
            torch.cos(elevation),
        ),
        dim=-1,
    )