"""RAVEN G1 camera-tracking validation demo.

This script currently runs two controllers:

1. A deterministic G1-follow controller.
   - Generates [vx, vy, yaw_rate].
   - NVIDIA's pretrained G1 policy converts this into 37 joint actions.

2. A deterministic camera/tripod controller.
   - Consumes the 12-value camera-centric RAVEN observation.
   - Generates:
       [forward, lateral, vertical, pan_rate, tilt_rate]

The camera controller is a temporary deterministic baseline.
Later it will be replaced by the trainable RAVEN policy.
"""

import argparse

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Command-line arguments / Isaac Sim launch
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="RAVEN G1 camera tracking and target-following demo."
)

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments.",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Imports requiring Isaac Sim
# -----------------------------------------------------------------------------

import importlib.metadata as metadata
import math

import gymnasium as gym
import torch

from packaging import version
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import (
    quat_apply_inverse,
    yaw_quat,
)

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

from isaaclab_rl.utils.pretrained_checkpoint import (
    get_published_pretrained_checkpoint,
)

# Register RAVEN Gym environments.
import raven_g1.tasks  # noqa: F401

from raven_g1.tasks.manager_based.raven_g1.raven_g1_env_cfg import (
    RavenG1EnvCfg_PLAY,
)

from raven_g1.tasks.manager_based.raven_g1.agents.rsl_rl_ppo_cfg import (
    RavenG1PPORunnerCfg,
)

from raven_g1.tasks.manager_based.raven_g1 import mdp


# =============================================================================
# G1 deterministic follower settings
# =============================================================================

DESIRED_DISTANCE = 2.0

DISTANCE_GAIN = 0.8
YAW_GAIN = 1.5

MAX_FORWARD_SPEED = 0.8
MAX_YAW_RATE = 1.0


# =============================================================================
# Camera / tripod deterministic controller settings
# =============================================================================

CAMERA_DISTANCE_GAIN = 0.45

CAMERA_LATERAL_GAIN = 0.20
CAMERA_VERTICAL_GAIN = 0.20

CAMERA_PAN_GAIN = 1.8
CAMERA_TILT_GAIN = 1.5

CAMERA_MAX_FORWARD_SPEED = 0.25
CAMERA_MAX_LATERAL_SPEED = 0.20
CAMERA_MAX_VERTICAL_SPEED = 0.10

CAMERA_MAX_PAN_RATE = 0.80
CAMERA_MAX_TILT_RATE = 0.60


# =============================================================================
# G1 high-level follower
# =============================================================================


def compute_follow_command(robot, target):
    """Generate [forward velocity, lateral velocity, yaw rate] for G1."""

    # ------------------------------------------------------------------
    # Vector from robot to target in world space.
    # ------------------------------------------------------------------

    target_delta_w = (
        target.data.root_pos_w
        - robot.data.root_pos_w
    ).clone()

    # G1 locomotion remains ground-plane locomotion.
    target_delta_w[:, 2] = 0.0

    # ------------------------------------------------------------------
    # Convert target vector into the G1 yaw frame.
    # ------------------------------------------------------------------

    robot_yaw_w = yaw_quat(
        robot.data.root_quat_w
    )

    target_delta_b = quat_apply_inverse(
        robot_yaw_w,
        target_delta_w,
    )

    forward = target_delta_b[:, 0]
    lateral = target_delta_b[:, 1]

    distance = torch.sqrt(
        forward.square()
        + lateral.square()
    )

    bearing = torch.atan2(
        lateral,
        forward,
    )

    # ------------------------------------------------------------------
    # Forward velocity
    # ------------------------------------------------------------------

    distance_error = (
        distance
        - DESIRED_DISTANCE
    )

    forward_speed = (
        DISTANCE_GAIN
        * distance_error
    )

    forward_speed = torch.clamp(
        forward_speed,
        min=0.0,
        max=MAX_FORWARD_SPEED,
    )

    # Turn before moving aggressively if the target is off-axis.
    facing_factor = torch.clamp(
        torch.cos(bearing),
        min=0.0,
        max=1.0,
    )

    forward_speed *= facing_factor

    # ------------------------------------------------------------------
    # Yaw
    # ------------------------------------------------------------------

    yaw_rate = (
        YAW_GAIN
        * bearing
    )

    yaw_rate = torch.clamp(
        yaw_rate,
        min=-MAX_YAW_RATE,
        max=MAX_YAW_RATE,
    )

    # No G1 strafing for this baseline.
    lateral_speed = torch.zeros_like(
        forward_speed
    )

    command = torch.stack(
        (
            forward_speed,
            lateral_speed,
            yaw_rate,
        ),
        dim=-1,
    )

    return (
        command,
        distance,
        bearing,
        target_delta_b,
    )


# =============================================================================
# Camera / tripod high-level controller
# =============================================================================


def compute_camera_rig_command(raven_obs):
    """Convert the 12 RAVEN observations into a 5-DOF camera command.

    Input observation layout:

        [0]  target X in camera frame
        [1]  target Y in camera frame
        [2]  target Z in camera frame

        [3]  target relative velocity X
        [4]  target relative velocity Y
        [5]  target relative velocity Z

        [6]  camera-target 3D distance
        [7]  shot-distance error

        [8]  sin(azimuth)
        [9]  cos(azimuth)

        [10] sin(elevation)
        [11] cos(elevation)

    Output command:

        [0] forward velocity
        [1] lateral velocity
        [2] vertical velocity
        [3] pan rate
        [4] tilt rate
    """

    # ------------------------------------------------------------------
    # Target position
    # ------------------------------------------------------------------

    target_y = raven_obs[:, 1]
    target_z = raven_obs[:, 2]

    # ------------------------------------------------------------------
    # Shot distance
    # ------------------------------------------------------------------

    distance_error = raven_obs[:, 7]

    # ------------------------------------------------------------------
    # Recover angular errors
    # ------------------------------------------------------------------

    azimuth = torch.atan2(
        raven_obs[:, 8],
        raven_obs[:, 9],
    )

    elevation = torch.atan2(
        raven_obs[:, 10],
        raven_obs[:, 11],
    )

    # ------------------------------------------------------------------
    # Dolly forward / backward
    # ------------------------------------------------------------------

    forward_speed = (
        CAMERA_DISTANCE_GAIN
        * distance_error
    )

    forward_speed = torch.clamp(
        forward_speed,
        -CAMERA_MAX_FORWARD_SPEED,
        CAMERA_MAX_FORWARD_SPEED,
    )

    # ------------------------------------------------------------------
    # Truck left / right
    #
    # Pan does most of the immediate horizontal framing correction.
    # Lateral movement is deliberately slower and smoother.
    # ------------------------------------------------------------------

    lateral_speed = (
        CAMERA_LATERAL_GAIN
        * target_y
    )

    lateral_speed = torch.clamp(
        lateral_speed,
        -CAMERA_MAX_LATERAL_SPEED,
        CAMERA_MAX_LATERAL_SPEED,
    )

    # ------------------------------------------------------------------
    # Pedestal up / down
    # ------------------------------------------------------------------

    vertical_speed = (
        CAMERA_VERTICAL_GAIN
        * target_z
    )

    vertical_speed = torch.clamp(
        vertical_speed,
        -CAMERA_MAX_VERTICAL_SPEED,
        CAMERA_MAX_VERTICAL_SPEED,
    )

    # ------------------------------------------------------------------
    # Pan
    # ------------------------------------------------------------------

    pan_rate = (
        CAMERA_PAN_GAIN
        * azimuth
    )

    pan_rate = torch.clamp(
        pan_rate,
        -CAMERA_MAX_PAN_RATE,
        CAMERA_MAX_PAN_RATE,
    )

    # ------------------------------------------------------------------
    # Tilt
    # ------------------------------------------------------------------

    tilt_rate = (
        CAMERA_TILT_GAIN
        * elevation
    )

    tilt_rate = torch.clamp(
        tilt_rate,
        -CAMERA_MAX_TILT_RATE,
        CAMERA_MAX_TILT_RATE,
    )

    return torch.stack(
        (
            forward_speed,
            lateral_speed,
            vertical_speed,
            pan_rate,
            tilt_rate,
        ),
        dim=-1,
    )


# =============================================================================
# Main
# =============================================================================


def main():

    # ------------------------------------------------------------------
    # RSL-RL version
    # ------------------------------------------------------------------

    installed_rsl_rl_version = metadata.version(
        "rsl-rl-lib"
    )

    print(
        f"[RAVEN] Installed RSL-RL version: "
        f"{installed_rsl_rl_version}"
    )

    # ------------------------------------------------------------------
    # Environment configuration
    # ------------------------------------------------------------------

    env_cfg = RavenG1EnvCfg_PLAY()

    env_cfg.scene.num_envs = (
        args_cli.num_envs
    )

    # Deterministic validation.
    env_cfg.observations.policy.enable_corruption = False

    # ------------------------------------------------------------------
    # Prevent Isaac Lab's stock velocity command generator from
    # overwriting our G1 follower command.
    # ------------------------------------------------------------------

    env_cfg.commands.base_velocity.resampling_time_range = (
        1.0e9,
        1.0e9,
    )

    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.heading = None

    # ------------------------------------------------------------------
    # Start G1 at origin facing +X.
    # ------------------------------------------------------------------

    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.episode_length_s = 120.0

    # ------------------------------------------------------------------
    # RSL-RL agent configuration
    # ------------------------------------------------------------------

    agent_cfg = RavenG1PPORunnerCfg()

    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        installed_rsl_rl_version,
    )

    print(
        "[RAVEN] RSL-RL configuration compatibility applied."
    )

    # ------------------------------------------------------------------
    # Load NVIDIA's pretrained G1 locomotion checkpoint.
    # ------------------------------------------------------------------

    checkpoint_path = (
        get_published_pretrained_checkpoint(
            "rsl_rl",
            "Isaac-Velocity-Flat-G1-v0",
        )
    )

    if checkpoint_path is None:
        raise RuntimeError(
            "Could not obtain the published "
            "Isaac-Velocity-Flat-G1-v0 checkpoint."
        )

    checkpoint_path = (
        handle_deprecated_rsl_rl_checkpoint(
            checkpoint_path,
            installed_rsl_rl_version,
        )
    )

    print(
        "[RAVEN] Loading pretrained G1 locomotion checkpoint:"
    )

    print(
        f"[RAVEN] {checkpoint_path}"
    )

    # ------------------------------------------------------------------
    # Create RAVEN environment.
    # ------------------------------------------------------------------

    env = gym.make(
        "Raven-G1-FollowTarget-Play-v0",
        cfg=env_cfg,
    )

    env = RslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
    )

    base_env = env.unwrapped

    # ------------------------------------------------------------------
    # Scene references
    # ------------------------------------------------------------------

    robot = base_env.scene["robot"]
    target = base_env.scene["target"]

    # Existing G1 velocity command generator.
    command_term = (
        base_env.command_manager.get_term(
            "base_velocity"
        )
    )

    # ------------------------------------------------------------------
    # Load pretrained G1 locomotion network.
    # ------------------------------------------------------------------

    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )

    runner.load(
        checkpoint_path
    )

    policy = runner.get_inference_policy(
        device=base_env.device,
    )

    print()
    print(
        "[RAVEN] Pretrained G1 locomotion policy loaded successfully."
    )
    print(
        "[RAVEN] G1 follower active."
    )
    print(
        "[RAVEN] Camera/tripod controller active."
    )
    print(
        f"[RAVEN] Desired shot distance: "
        f"{DESIRED_DISTANCE:.2f} m"
    )
    print()

    # Initial observation.
    obs = env.get_observations()

    step = 0

    # ==================================================================
    # Simulation loop
    # ==================================================================

    while simulation_app.is_running():

        with torch.inference_mode():

            # ==========================================================
            # 1. CAMERA / TRIPOD CONTROLLER
            # ==========================================================

            # Read the current camera-centric observation.
            raven_obs = (
                base_env.observation_manager.compute_group(
                    "raven"
                )
            )

            # Convert observation into:
            #
            # [forward,
            #  lateral,
            #  vertical,
            #  pan_rate,
            #  tilt_rate]
            camera_command = (
                compute_camera_rig_command(
                    raven_obs
                )
            )

            # Apply camera/tripod command.
            mdp.apply_camera_rig_command(
                base_env,
                camera_command,
                dt=base_env.step_dt,
            )

            # ==========================================================
            # 2. EXISTING G1 FOLLOWER
            # ==========================================================

            (
                command,
                distance,
                bearing,
                target_delta_b,
            ) = compute_follow_command(
                robot,
                target,
            )

            # Write our G1 motion command directly into Isaac Lab's
            # velocity command term.
            command_term.vel_command_b.copy_(
                command
            )

            command_term.is_standing_env[:] = False
            command_term.is_heading_env[:] = False

            # Recompute locomotion observation now that we changed
            # the velocity command.
            obs = env.get_observations()

            # ==========================================================
            # 3. PRETRAINED LOW-LEVEL G1 POLICY
            # ==========================================================

            actions = policy(
                obs
            )

            obs, _, dones, _ = env.step(
                actions
            )

            # ----------------------------------------------------------
            # Reset policy state if an environment terminates.
            # ----------------------------------------------------------

            if version.parse(
                installed_rsl_rl_version
            ) >= version.parse("4.0.0"):

                policy.reset(
                    dones
                )

            else:

                try:
                    policy_nn = runner.alg.policy

                except AttributeError:
                    policy_nn = runner.alg.actor_critic

                policy_nn.reset(
                    dones
                )

            # ==========================================================
            # 4. TELEMETRY
            # ==========================================================

            if step % 50 == 0:

                # ------------------------------------------------------
                # Existing G1 telemetry
                # ------------------------------------------------------

                distance_value = (
                    distance[0].item()
                )

                bearing_deg = math.degrees(
                    bearing[0].item()
                )

                forward_value = (
                    target_delta_b[0, 0].item()
                )

                lateral_value = (
                    target_delta_b[0, 1].item()
                )

                vx = (
                    command[0, 0].item()
                )

                yaw_rate = (
                    command[0, 2].item()
                )

                # ------------------------------------------------------
                # Re-read the camera observation for telemetry.
                # ------------------------------------------------------

                raven_obs_telemetry = (
                    base_env.observation_manager.compute_group(
                        "raven"
                    )
                )

                raven_values = (
                    raven_obs_telemetry[0]
                )

                # ------------------------------------------------------
                # Camera-relative target position
                # ------------------------------------------------------

                camera_target_x = (
                    raven_values[0].item()
                )

                camera_target_y = (
                    raven_values[1].item()
                )

                camera_target_z = (
                    raven_values[2].item()
                )

                # ------------------------------------------------------
                # Camera-relative target velocity
                # ------------------------------------------------------

                target_vel_x = (
                    raven_values[3].item()
                )

                target_vel_y = (
                    raven_values[4].item()
                )

                target_vel_z = (
                    raven_values[5].item()
                )

                # ------------------------------------------------------
                # Shot distance
                # ------------------------------------------------------

                camera_distance = (
                    raven_values[6].item()
                )

                camera_distance_error = (
                    raven_values[7].item()
                )

                # ------------------------------------------------------
                # Azimuth
                # ------------------------------------------------------

                azimuth_sin = (
                    raven_values[8].item()
                )

                azimuth_cos = (
                    raven_values[9].item()
                )

                azimuth_deg = math.degrees(
                    math.atan2(
                        azimuth_sin,
                        azimuth_cos,
                    )
                )

                # ------------------------------------------------------
                # Elevation
                # ------------------------------------------------------

                elevation_sin = (
                    raven_values[10].item()
                )

                elevation_cos = (
                    raven_values[11].item()
                )

                elevation_deg = math.degrees(
                    math.atan2(
                        elevation_sin,
                        elevation_cos,
                    )
                )

                # ------------------------------------------------------
                # Camera command telemetry
                # ------------------------------------------------------

                camera_vx = (
                    camera_command[0, 0].item()
                )

                camera_vy = (
                    camera_command[0, 1].item()
                )

                camera_vz = (
                    camera_command[0, 2].item()
                )

                camera_pan_rate = (
                    camera_command[0, 3].item()
                )

                camera_tilt_rate = (
                    camera_command[0, 4].item()
                )

                # ------------------------------------------------------
                # Print G1 state
                # ------------------------------------------------------

                print(
                    "[RAVEN G1] "
                    f"distance={distance_value:5.2f} m | "
                    f"bearing={bearing_deg:6.1f} deg | "
                    f"target_b=("
                    f"{forward_value:5.2f}, "
                    f"{lateral_value:5.2f}) | "
                    f"cmd vx={vx:4.2f} | "
                    f"yaw={yaw_rate:5.2f}"
                )

                # ------------------------------------------------------
                # Print camera tracking state
                # ------------------------------------------------------

                print(
                    "[CAMERA OBS] "
                    f"target_xyz=("
                    f"{camera_target_x:5.2f}, "
                    f"{camera_target_y:5.2f}, "
                    f"{camera_target_z:5.2f}) | "
                    f"rel_vel=("
                    f"{target_vel_x:5.2f}, "
                    f"{target_vel_y:5.2f}, "
                    f"{target_vel_z:5.2f}) | "
                    f"dist={camera_distance:5.2f} | "
                    f"dist_err={camera_distance_error:+5.2f} | "
                    f"az={azimuth_deg:+6.1f} deg | "
                    f"el={elevation_deg:+6.1f} deg"
                )

                # ------------------------------------------------------
                # Print camera control command
                # ------------------------------------------------------

                print(
                    "[CAMERA CMD] "
                    f"dolly={camera_vx:+5.2f} | "
                    f"truck={camera_vy:+5.2f} | "
                    f"pedestal={camera_vz:+5.2f} | "
                    f"pan={camera_pan_rate:+5.2f} rad/s | "
                    f"tilt={camera_tilt_rate:+5.2f} rad/s"
                )

                print()

            step += 1

    env.close()


# =============================================================================
# Entry point
# =============================================================================


if __name__ == "__main__":

    main()

    simulation_app.close()