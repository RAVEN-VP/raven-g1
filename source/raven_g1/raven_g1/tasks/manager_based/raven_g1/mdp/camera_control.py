from __future__ import annotations

import math

import torch

from isaaclab.utils.math import (
    quat_apply,
    quat_from_euler_xyz,
    yaw_quat,
)


# -----------------------------------------------------------------------------
# RAVEN 5-DOF camera/tripod proxy controller
#
# Command layout:
#
# [0] forward velocity   m/s
# [1] lateral velocity   m/s
# [2] vertical velocity  m/s
# [3] pan rate           rad/s
# [4] tilt rate          rad/s
#
# The camera/tripod proxy is intentionally kinematic at this stage.
# Later the G1 will physically manipulate a dynamic tripod.
# -----------------------------------------------------------------------------


def apply_camera_rig_command(
    env,
    command: torch.Tensor,
    dt: float,
    min_camera_height: float = 0.90,
    max_camera_height: float = 1.15,
    min_horizontal_reach: float = 0.25,
    max_horizontal_reach: float = 0.50,
    max_forward_speed: float = 0.25,
    max_lateral_speed: float = 0.20,
    max_vertical_speed: float = 0.10,
    max_pan_rate: float = 0.80,
    max_tilt_rate: float = 0.60,
    max_tilt_angle: float = math.radians(45.0),
):
    """Apply a 5-DOF velocity command to the RAVEN camera/tripod proxy."""

    if command.ndim != 2 or command.shape[1] != 5:
        raise ValueError(
            f"Expected camera command shape (num_envs, 5), "
            f"received {tuple(command.shape)}."
        )

    camera_rig = env.scene["camera_rig"]
    tripod_base = env.scene["tripod_base"]
    tripod_column = env.scene["tripod_column"]
    robot = env.scene["robot"]

    num_envs = env.scene.num_envs
    device = env.device

    # ------------------------------------------------------------------
    # Persistent pan / tilt state
    # ------------------------------------------------------------------

    if not hasattr(env, "_raven_camera_pan"):
        env._raven_camera_pan = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.float32,
        )

    if not hasattr(env, "_raven_camera_tilt"):
        env._raven_camera_tilt = torch.zeros(
            num_envs,
            device=device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Clamp command to safe proxy limits
    # ------------------------------------------------------------------

    command = command.clone()

    command[:, 0] = torch.clamp(
        command[:, 0],
        -max_forward_speed,
        max_forward_speed,
    )

    command[:, 1] = torch.clamp(
        command[:, 1],
        -max_lateral_speed,
        max_lateral_speed,
    )

    command[:, 2] = torch.clamp(
        command[:, 2],
        -max_vertical_speed,
        max_vertical_speed,
    )

    command[:, 3] = torch.clamp(
        command[:, 3],
        -max_pan_rate,
        max_pan_rate,
    )

    command[:, 4] = torch.clamp(
        command[:, 4],
        -max_tilt_rate,
        max_tilt_rate,
    )

    # ------------------------------------------------------------------
    # Integrate pan / tilt
    # ------------------------------------------------------------------

    env._raven_camera_pan += (
        command[:, 3] * dt
    )

    env._raven_camera_tilt += (
        command[:, 4] * dt
    )

    # Keep pan wrapped to [-pi, pi].
    env._raven_camera_pan = (
        (
            env._raven_camera_pan
            + math.pi
        )
        % (2.0 * math.pi)
    ) - math.pi

    # Physical-style tilt limit.
    env._raven_camera_tilt = torch.clamp(
        env._raven_camera_tilt,
        -max_tilt_angle,
        max_tilt_angle,
    )

    pan = env._raven_camera_pan
    tilt = env._raven_camera_tilt

    zeros = torch.zeros_like(pan)

    # ------------------------------------------------------------------
    # Translate in the camera's PAN frame
    #
    # Translation does not pitch with the camera:
    #
    # X = dolly forward/back
    # Y = truck left/right
    # Z = pedestal up/down
    # ------------------------------------------------------------------

    pan_quat = quat_from_euler_xyz(
        zeros,
        zeros,
        pan,
    )

    local_velocity = command[:, 0:3]

    world_velocity = quat_apply(
        pan_quat,
        local_velocity,
    )

    old_position = (
        camera_rig.data.root_pos_w.clone()
    )

    new_position = (
        old_position
        + world_velocity * dt
    )

    origins = env.scene.env_origins

    # ------------------------------------------------------------------
    # Clamp camera height
    # ------------------------------------------------------------------

    relative_height = (
        new_position[:, 2]
        - origins[:, 2]
    )

    relative_height = torch.clamp(
        relative_height,
        min_camera_height,
        max_camera_height,
    )

    new_position[:, 2] = (
        origins[:, 2]
        + relative_height
    )

    # ------------------------------------------------------------------
    # Keep tripod/camera inside a conservative G1 manipulation envelope.
    #
    # This is an approximation for the proxy phase.
    # Later this will be replaced by actual hand / IK reach constraints.
    # ------------------------------------------------------------------

    robot_xy = robot.data.root_pos_w[:, :2]

    camera_offset_xy = (
        new_position[:, :2]
        - robot_xy
    )

    reach = torch.linalg.vector_norm(
        camera_offset_xy,
        dim=-1,
        keepdim=True,
    )

    # Forward direction fallback if the camera somehow reaches exactly
    # the robot root.
    robot_yaw = yaw_quat(
        robot.data.root_quat_w
    )

    forward_local = torch.zeros(
        (num_envs, 3),
        device=device,
        dtype=torch.float32,
    )

    forward_local[:, 0] = 1.0

    robot_forward_w = quat_apply(
        robot_yaw,
        forward_local,
    )[:, :2]

    safe_reach = torch.clamp(
        reach,
        min=1.0e-6,
    )

    reach_direction = (
        camera_offset_xy
        / safe_reach
    )

    zero_reach = (
        reach[:, 0]
        < 1.0e-6
    )

    reach_direction[zero_reach] = (
        robot_forward_w[zero_reach]
    )

    clamped_reach = torch.clamp(
        reach,
        min_horizontal_reach,
        max_horizontal_reach,
    )

    new_position[:, :2] = (
        robot_xy
        + reach_direction * clamped_reach
    )

    # ------------------------------------------------------------------
    # Camera orientation
    #
    # +pan  = turn left
    # +tilt = look upward
    #
    # quat_from_euler_xyz uses positive rotation about +Y for pitch;
    # for our +X-forward frame, looking upward therefore uses -tilt.
    # ------------------------------------------------------------------

    camera_quat = quat_from_euler_xyz(
        zeros,
        -tilt,
        pan,
    )

    camera_pose = torch.cat(
        (
            new_position,
            camera_quat,
        ),
        dim=-1,
    )

    camera_rig.write_root_pose_to_sim(
        camera_pose
    )

    # ------------------------------------------------------------------
    # Write physically meaningful camera velocity too.
    #
    # Our observations use relative camera/target velocity, so this
    # must reflect the camera's actual kinematic motion.
    # ------------------------------------------------------------------

    actual_linear_velocity = (
        new_position
        - old_position
    ) / dt

    angular_velocity_local = torch.zeros(
        (num_envs, 3),
        device=device,
        dtype=torch.float32,
    )

    # Our camera pitch is -tilt.
    angular_velocity_local[:, 1] = (
        -command[:, 4]
    )

    angular_velocity_local[:, 2] = (
        command[:, 3]
    )

    angular_velocity_world = quat_apply(
        pan_quat,
        angular_velocity_local,
    )

    camera_velocity = torch.cat(
        (
            actual_linear_velocity,
            angular_velocity_world,
        ),
        dim=-1,
    )

    camera_rig.write_root_velocity_to_sim(
        camera_velocity
    )

    # ------------------------------------------------------------------
    # Move tripod visual components with the camera in X/Y.
    #
    # Base remains on the floor.
    # Column remains upright.
    # Camera head controls Z, pan and tilt.
    # ------------------------------------------------------------------

    tripod_base_pose = (
        tripod_base.data.root_state_w[:, :7].clone()
    )

    tripod_base_pose[:, 0:2] = (
        new_position[:, 0:2]
    )

    tripod_base_pose[:, 2] = (
        origins[:, 2] + 0.03
    )

    tripod_base_pose[:, 3] = 1.0
    tripod_base_pose[:, 4:7] = 0.0

    tripod_base.write_root_pose_to_sim(
        tripod_base_pose
    )

    tripod_column_pose = (
        tripod_column.data.root_state_w[:, :7].clone()
    )

    tripod_column_pose[:, 0:2] = (
        new_position[:, 0:2]
    )

    tripod_column_pose[:, 2] = (
        origins[:, 2] + 0.44
    )

    tripod_column_pose[:, 3] = 1.0
    tripod_column_pose[:, 4:7] = 0.0

    tripod_column.write_root_pose_to_sim(
        tripod_column_pose
    )