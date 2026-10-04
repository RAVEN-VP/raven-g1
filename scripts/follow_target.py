"""RAVEN G1 target-following demo.

A deterministic high-level controller converts the moving target position
into velocity commands. NVIDIA's pretrained G1 locomotion policy executes
those commands at the joint level.
"""

import argparse

from isaaclab.app import AppLauncher


# -----------------------------------------------------------------------------
# Arguments / Isaac Sim launch
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="RAVEN G1 target-following demo."
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

from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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


# -----------------------------------------------------------------------------
# Follow-controller settings
# -----------------------------------------------------------------------------

DESIRED_DISTANCE = 2.0

DISTANCE_GAIN = 0.8
YAW_GAIN = 1.5

MAX_FORWARD_SPEED = 0.8
MAX_YAW_RATE = 1.0


# -----------------------------------------------------------------------------
# High-level RAVEN controller
# -----------------------------------------------------------------------------

def compute_follow_command(robot, target):
    """Generate [forward velocity, lateral velocity, yaw rate]."""

    # World-space vector from G1 to target.
    target_delta_w = (
        target.data.root_pos_w
        - robot.data.root_pos_w
    ).clone()

    # Ignore vertical separation for ground-plane following.
    target_delta_w[:, 2] = 0.0

    # Convert the vector into the G1's yaw-relative body frame.
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
    # Forward command
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

    # If the target is significantly off-axis, prioritise turning.
    facing_factor = torch.clamp(
        torch.cos(bearing),
        min=0.0,
        max=1.0,
    )

    forward_speed *= facing_factor

    # ------------------------------------------------------------------
    # Yaw command
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

    # No strafing for our first deterministic baseline.
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

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
    # Environment config
    # ------------------------------------------------------------------

    env_cfg = RavenG1EnvCfg_PLAY()

    env_cfg.scene.num_envs = (
        args_cli.num_envs
    )

    # Disable observation corruption for visual validation.
    env_cfg.observations.policy.enable_corruption = False

    # Prevent stock command resampling from replacing
    # the RAVEN-generated velocity command.
    env_cfg.commands.base_velocity.resampling_time_range = (
        1.0e9,
        1.0e9,
    )

    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.heading = None

    # Start G1 at the environment origin, facing +X.
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.episode_length_s = 120.0

    # ------------------------------------------------------------------
    # Agent config
    # ------------------------------------------------------------------

    agent_cfg = RavenG1PPORunnerCfg()

    # CRITICAL:
    #
    # Convert the stock Isaac Lab / G1 configuration into the format
    # expected by whichever RSL-RL version is actually installed.
    #
    # This handles the RSL-RL 4.x / 5.x configuration transition.
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg,
        installed_rsl_rl_version,
    )

    print(
        "[RAVEN] RSL-RL configuration compatibility applied."
    )

    # ------------------------------------------------------------------
    # Published G1 checkpoint
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

    # Convert an older checkpoint layout if the installed RSL-RL
    # version requires the newer actor/critic checkpoint structure.
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
    # Environment
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

    robot = base_env.scene["robot"]
    target = base_env.scene["target"]

    command_term = (
        base_env.command_manager.get_term(
            "base_velocity"
        )
    )

    # ------------------------------------------------------------------
    # Pretrained locomotion policy
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
        "[RAVEN] Target follower active."
    )
    print(
        f"[RAVEN] Desired stand-off distance: "
        f"{DESIRED_DISTANCE:.2f} m"
    )
    print()

    # Initial observation.
    obs = env.get_observations()

    step = 0

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    while simulation_app.is_running():

        with torch.inference_mode():

            # ----------------------------------------------------------
            # RAVEN high-level target controller
            # ----------------------------------------------------------

            (
                command,
                distance,
                bearing,
                target_delta_b,
            ) = compute_follow_command(
                robot,
                target,
            )

            # Write RAVEN's velocity target directly into the existing
            # Unitree G1 velocity-command term.
            command_term.vel_command_b.copy_(
                command
            )

            command_term.is_standing_env[:] = False
            command_term.is_heading_env[:] = False

            # Recompute policy observation after injecting our command.
            obs = env.get_observations()

            # ----------------------------------------------------------
            # Low-level pretrained G1 locomotion policy
            # ----------------------------------------------------------

            actions = policy(
                obs
            )

            obs, _, dones, _ = env.step(
                actions
            )

            # RSL-RL >= 4 returns an inference policy object with reset().
            if version.parse(
                installed_rsl_rl_version
            ) >= version.parse("4.0.0"):

                policy.reset(
                    dones
                )

            else:
                # Compatibility for old RSL-RL versions.
                try:
                    policy_nn = runner.alg.policy
                except AttributeError:
                    policy_nn = runner.alg.actor_critic

                policy_nn.reset(
                    dones
                )

            # ----------------------------------------------------------
            # Telemetry
            # ----------------------------------------------------------

            if step % 50 == 0:

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

                print(
                    "[RAVEN] "
                    f"distance={distance_value:5.2f} m | "
                    f"bearing={bearing_deg:6.1f} deg | "
                    f"target_b=("
                    f"{forward_value:5.2f}, "
                    f"{lateral_value:5.2f}) | "
                    f"cmd vx={vx:4.2f} "
                    f"yaw={yaw_rate:5.2f}"
                )

            step += 1

    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()