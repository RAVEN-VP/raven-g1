"""RAVEN stationary G1 + tripod left-arm IK validation.

Current stage:

- G1 root is temporarily fixed to the world.
- Tripod base is fixed to the world.
- Camera pan and tilt remain passive.
- No locomotion policy is used.
- Lower body/right arm remain at their default pose.
- Differential IK gradually moves the LEFT palm toward a safe
  pre-grasp position beside the tripod control handle.

This stage intentionally does NOT grip or push the handle yet.
"""

import argparse

from isaaclab.app import AppLauncher


# =============================================================================
# Isaac Sim launch
# =============================================================================

parser = argparse.ArgumentParser(
    description=(
        "RAVEN stationary G1 left-arm IK "
        "pre-grasp validation."
    )
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app


# =============================================================================
# Imports requiring Isaac Sim
# =============================================================================

import math

import gymnasium as gym
import torch

from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)

from isaaclab.managers import SceneEntityCfg

from isaaclab.utils.math import (
    quat_apply,
    subtract_frame_transforms,
)

# Register RAVEN task.
import raven_g1.tasks  # noqa: F401

from raven_g1.tasks.manager_based.raven_g1.raven_g1_env_cfg import (
    RavenG1EnvCfg_PLAY,
)


# =============================================================================
# Configuration
# =============================================================================

# How long the hand takes to move from its default pose
# toward the pre-grasp point.
REACH_DURATION_S = 4.0

# The handle is 0.32 m long along its local X axis.
#
# We target the G1-facing portion of the handle rather than
# its centre.
HANDLE_REAR_OFFSET_X = -0.12

# Stay slightly below the handle during this validation.
#
# Once this works, the next stage will move the palm into
# actual contact and create the grip.
PREGRASP_OFFSET_Z = -0.05


# =============================================================================
# Helpers
# =============================================================================


def resolve_tripod_body(
    tripod,
    body_name: str,
) -> int:

    body_ids, _ = tripod.find_bodies(
        [body_name]
    )

    if len(body_ids) == 0:

        raise RuntimeError(
            f"Tripod body '{body_name}' was not found."
        )

    return body_ids[0]


def resolve_tripod_joint(
    tripod,
    joint_name: str,
) -> int:

    joint_ids, _ = tripod.find_joints(
        [joint_name]
    )

    if len(joint_ids) == 0:

        raise RuntimeError(
            f"Tripod joint '{joint_name}' was not found."
        )

    return joint_ids[0]


def build_action_index_map(
    action_term,
    num_robot_joints: int,
):
    """Map robot joint IDs to indices in the ActionManager vector."""

    joint_ids = action_term._joint_ids

    # Isaac Lab optimizes the all-joints case to slice(None).
    if isinstance(joint_ids, slice):

        return {
            joint_id: joint_id
            for joint_id in range(num_robot_joints)
        }

    return {
        int(robot_joint_id): action_index
        for action_index, robot_joint_id
        in enumerate(joint_ids)
    }


# =============================================================================
# Main
# =============================================================================


def main():

    # -------------------------------------------------------------------------
    # Environment configuration
    # -------------------------------------------------------------------------

    env_cfg = RavenG1EnvCfg_PLAY()

    env_cfg.scene.num_envs = 1

    env_cfg.observations.policy.enable_corruption = False

    # -------------------------------------------------------------------------
    # Temporarily fix G1 root.
    #
    # This isolates arm manipulation from locomotion/balance.
    # -------------------------------------------------------------------------

    env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # -------------------------------------------------------------------------
    # Keep velocity command inactive.
    # -------------------------------------------------------------------------

    env_cfg.commands.base_velocity.resampling_time_range = (
        1.0e9,
        1.0e9,
    )

    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0

    env_cfg.commands.base_velocity.heading_command = False
    env_cfg.commands.base_velocity.ranges.heading = None

    # -------------------------------------------------------------------------
    # Deterministic G1 spawn.
    # -------------------------------------------------------------------------

    env_cfg.events.reset_base.params[
        "pose_range"
    ] = {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    }

    env_cfg.episode_length_s = 120.0

    # -------------------------------------------------------------------------
    # Create environment
    # -------------------------------------------------------------------------

    env = gym.make(
        "Raven-G1-FollowTarget-Play-v0",
        cfg=env_cfg,
    )

    base_env = env.unwrapped

    robot = base_env.scene[
        "robot"
    ]

    tripod = base_env.scene[
        "tripod_camera"
    ]

    # -------------------------------------------------------------------------
    # Reset BEFORE stepping.
    # -------------------------------------------------------------------------

    env.reset()

    # -------------------------------------------------------------------------
    # Define the left-arm kinematic chain.
    #
    # We deliberately exclude fingers for this stage.
    # -------------------------------------------------------------------------

    left_arm_cfg = SceneEntityCfg(
        "robot",

        joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_elbow_roll_joint",
        ],

        body_names=[
            "left_palm_link",
        ],

        preserve_order=True,
    )

    left_arm_cfg.resolve(
        base_env.scene
    )

    left_arm_joint_ids = list(
        left_arm_cfg.joint_ids
    )

    left_palm_id = (
        left_arm_cfg.body_ids[0]
    )

    # -------------------------------------------------------------------------
    # Resolve tripod pieces.
    # -------------------------------------------------------------------------

    handle_body_id = resolve_tripod_body(
        tripod,
        "control_handle_link",
    )

    camera_body_id = resolve_tripod_body(
        tripod,
        "camera_link",
    )

    pan_joint_id = resolve_tripod_joint(
        tripod,
        "pan_joint",
    )

    tilt_joint_id = resolve_tripod_joint(
        tripod,
        "tilt_joint",
    )

    # -------------------------------------------------------------------------
    # Differential IK controller.
    #
    # POSITION only:
    #
    # We currently care about getting the palm to the handle region.
    # Hand orientation/grasp alignment comes next.
    # -------------------------------------------------------------------------

    ik_cfg = DifferentialIKControllerCfg(

        command_type="position",

        use_relative_mode=False,

        ik_method="dls",
    )

    ik_controller = DifferentialIKController(

        cfg=ik_cfg,

        num_envs=base_env.num_envs,

        device=base_env.device,
    )

    ik_controller.reset()

    # -------------------------------------------------------------------------
    # Jacobian body index.
    #
    # For a fixed-base articulation, PhysX's Jacobian body array does not
    # contain the fixed root body.
    # -------------------------------------------------------------------------

    if not robot.is_fixed_base:

        raise RuntimeError(
            "This validation script expects the G1 "
            "to have a fixed root."
        )

    ee_jacobian_index = (
        left_palm_id - 1
    )

    # -------------------------------------------------------------------------
    # ActionManager setup.
    #
    # Current RAVEN config has one 37-dimensional JointPositionAction.
    # -------------------------------------------------------------------------

    action_term = (
        base_env.action_manager.get_term(
            "joint_pos"
        )
    )

    action_dim = (
        base_env.action_manager.total_action_dim
    )

    if action_dim != robot.num_joints:

        raise RuntimeError(
            "This script expects one action value per G1 joint. "
            f"Action dimension={action_dim}, "
            f"robot joints={robot.num_joints}."
        )

    action_index_map = build_action_index_map(
        action_term,
        robot.num_joints,
    )

    left_arm_action_indices = [

        action_index_map[
            joint_id
        ]

        for joint_id
        in left_arm_joint_ids
    ]

    # -------------------------------------------------------------------------
    # Current action configuration:
    #
    # target = default_joint_position + action * scale
    # -------------------------------------------------------------------------

    action_scale_cfg = (
        base_env.cfg.actions.joint_pos.scale
    )

    if not isinstance(
        action_scale_cfg,
        (float, int),
    ):

        raise RuntimeError(
            "Expected scalar joint-position action scale "
            f"but received {action_scale_cfg}."
        )

    action_scale = float(
        action_scale_cfg
    )

    if action_scale == 0.0:

        raise RuntimeError(
            "Joint action scale cannot be zero."
        )

    if not base_env.cfg.actions.joint_pos.use_default_offset:

        raise RuntimeError(
            "This script expects use_default_offset=True."
        )

    # -------------------------------------------------------------------------
    # Full action vector.
    #
    # Zero keeps every non-controlled joint at its default pose.
    # -------------------------------------------------------------------------

    actions = torch.zeros(
        (
            base_env.num_envs,
            action_dim,
        ),
        device=base_env.device,
        dtype=torch.float32,
    )

    # -------------------------------------------------------------------------
    # Default positions for left arm.
    # -------------------------------------------------------------------------

    left_arm_default_joint_pos = (
        robot.data.default_joint_pos[
            :,
            left_arm_joint_ids,
        ].clone()
    )

    # -------------------------------------------------------------------------
    # Initial palm pose.
    # -------------------------------------------------------------------------

    initial_ee_pose_w = (
        robot.data.body_state_w[
            :,
            left_palm_id,
            0:7,
        ].clone()
    )

    root_pose_w = (
        robot.data.root_state_w[
            :,
            0:7,
        ]
    )

    (
        initial_ee_pos_b,
        _,
    ) = subtract_frame_transforms(

        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],

        initial_ee_pose_w[:, 0:3],
        initial_ee_pose_w[:, 3:7],
    )

    initial_robot_root_position = (
        robot.data.root_pos_w.clone()
    )

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    print()
    print(
        "============================================================"
    )

    print(
        " RAVEN LEFT-ARM IK PRE-GRASP TEST"
    )

    print(
        "============================================================"
    )
    print()

    print(
        "[RAVEN] G1 root fixed: "
        f"{robot.is_fixed_base}"
    )

    print(
        "[RAVEN] Tripod fixed: "
        f"{tripod.is_fixed_base}"
    )

    print(
        "[RAVEN] Left arm joints:"
    )

    for joint_id in left_arm_joint_ids:

        print(
            f"        {joint_id:2d}: "
            f"{robot.joint_names[joint_id]}"
        )

    print()

    print(
        "[RAVEN] End effector: left_palm_link"
    )

    print(
        f"[RAVEN] Palm body ID: {left_palm_id}"
    )

    print(
        f"[RAVEN] Jacobian body index: "
        f"{ee_jacobian_index}"
    )

    print(
        f"[RAVEN] Joint action scale: "
        f"{action_scale}"
    )

    print()

    print(
        "[RAVEN] The left hand will move over "
        f"{REACH_DURATION_S:.1f} seconds."
    )

    print(
        "[RAVEN] Target is a PRE-GRASP point "
        "near the rear of the tripod handle."
    )

    print(
        "[RAVEN] No physical grip is created yet."
    )

    print()

    step = 0

    # =========================================================================
    # Simulation loop
    # =========================================================================

    while simulation_app.is_running():

        with torch.inference_mode():

            # -----------------------------------------------------------------
            # Current robot/root pose
            # -----------------------------------------------------------------

            root_pose_w = (
                robot.data.root_state_w[
                    :,
                    0:7,
                ]
            )

            ee_pose_w = (
                robot.data.body_state_w[
                    :,
                    left_palm_id,
                    0:7,
                ]
            )

            (
                ee_pos_b,
                ee_quat_b,
            ) = subtract_frame_transforms(

                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],

                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )

            # -----------------------------------------------------------------
            # Handle pose
            # -----------------------------------------------------------------

            handle_pos_w = (
                tripod.data.body_pos_w[
                    :,
                    handle_body_id,
                ]
            )

            handle_quat_w = (
                tripod.data.body_quat_w[
                    :,
                    handle_body_id,
                ]
            )

            # -----------------------------------------------------------------
            # Safe pre-grasp point expressed in HANDLE coordinates.
            #
            # -12 cm in local X:
            #     near the G1-facing end of the handle.
            #
            # -5 cm in local Z:
            #     keep the palm just below it for this test.
            # -----------------------------------------------------------------

            pregrasp_offset_handle = (
                torch.zeros(
                    (
                        base_env.num_envs,
                        3,
                    ),
                    device=base_env.device,
                )
            )

            pregrasp_offset_handle[:, 0] = (
                HANDLE_REAR_OFFSET_X
            )

            pregrasp_offset_handle[:, 2] = (
                PREGRASP_OFFSET_Z
            )

            pregrasp_offset_w = quat_apply(

                handle_quat_w,

                pregrasp_offset_handle,
            )

            pregrasp_pos_w = (
                handle_pos_w
                + pregrasp_offset_w
            )

            # -----------------------------------------------------------------
            # Convert target into G1 root frame.
            # -----------------------------------------------------------------

            (
                pregrasp_pos_b,
                _,
            ) = subtract_frame_transforms(

                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],

                pregrasp_pos_w,
                handle_quat_w,
            )

            # -----------------------------------------------------------------
            # Smoothly interpolate from the starting hand position.
            # -----------------------------------------------------------------

            elapsed_s = (
                step
                * base_env.step_dt
            )

            reach_alpha = min(
                elapsed_s
                / REACH_DURATION_S,
                1.0,
            )

            desired_ee_pos_b = (

                initial_ee_pos_b
                * (1.0 - reach_alpha)

                + pregrasp_pos_b
                * reach_alpha
            )

            # -----------------------------------------------------------------
            # Give desired Cartesian position to IK.
            #
            # For position mode, Isaac Lab still expects the current EE
            # quaternion when setting the command.
            # -----------------------------------------------------------------

            ik_controller.set_command(

                desired_ee_pos_b,

                ee_quat=ee_quat_b,
            )

            # -----------------------------------------------------------------
            # G1 left-arm Jacobian
            # -----------------------------------------------------------------

            jacobian = (
                robot.root_physx_view.get_jacobians()[
                    :,
                    ee_jacobian_index,
                    :,
                    left_arm_joint_ids,
                ]
            )

            current_left_arm_joint_pos = (
                robot.data.joint_pos[
                    :,
                    left_arm_joint_ids,
                ]
            )

            # -----------------------------------------------------------------
            # Solve IK
            # -----------------------------------------------------------------

            desired_left_arm_joint_pos = (
                ik_controller.compute(

                    ee_pos_b,

                    ee_quat_b,

                    jacobian,

                    current_left_arm_joint_pos,
                )
            )

            # -----------------------------------------------------------------
            # Keep solution inside G1's soft joint limits.
            # -----------------------------------------------------------------

            arm_limits = (
                robot.data.soft_joint_pos_limits[
                    :,
                    left_arm_joint_ids,
                    :
                ]
            )

            desired_left_arm_joint_pos = torch.maximum(

                desired_left_arm_joint_pos,

                arm_limits[:, :, 0],
            )

            desired_left_arm_joint_pos = torch.minimum(

                desired_left_arm_joint_pos,

                arm_limits[:, :, 1],
            )

            # -----------------------------------------------------------------
            # Convert absolute joint targets into the environment's
            # JointPositionAction representation:
            #
            # target = default + action * scale
            #
            # therefore:
            #
            # action = (target - default) / scale
            # -----------------------------------------------------------------

            left_arm_actions = (

                desired_left_arm_joint_pos
                - left_arm_default_joint_pos

            ) / action_scale

            # Everything else remains at default.
            actions.zero_()

            actions[
                :,
                left_arm_action_indices,
            ] = left_arm_actions

            # -----------------------------------------------------------------
            # Step environment
            # -----------------------------------------------------------------

            env.step(
                actions
            )

            # -----------------------------------------------------------------
            # Telemetry
            # -----------------------------------------------------------------

            if step % 25 == 0:

                left_palm_pos_w = (
                    robot.data.body_pos_w[
                        0,
                        left_palm_id,
                    ]
                )

                handle_position_w = (
                    tripod.data.body_pos_w[
                        0,
                        handle_body_id,
                    ]
                )

                current_pregrasp_pos_w = (
                    pregrasp_pos_w[0]
                )

                palm_to_pregrasp = (
                    torch.linalg.vector_norm(

                        current_pregrasp_pos_w
                        - left_palm_pos_w

                    ).item()
                )

                palm_to_handle = (
                    torch.linalg.vector_norm(

                        handle_position_w
                        - left_palm_pos_w

                    ).item()
                )

                robot_root_position = (
                    robot.data.root_pos_w[
                        0
                    ]
                )

                root_drift = (
                    torch.linalg.vector_norm(

                        robot_root_position
                        - initial_robot_root_position[
                            0
                        ]

                    ).item()
                )

                pan_deg = math.degrees(

                    tripod.data.joint_pos[
                        0,
                        pan_joint_id,
                    ].item()
                )

                tilt_deg = math.degrees(

                    tripod.data.joint_pos[
                        0,
                        tilt_joint_id,
                    ].item()
                )

                raven_obs = (
                    base_env.observation_manager.compute_group(
                        "raven"
                    )
                )

                raven_values = (
                    raven_obs[0]
                )

                azimuth_deg = math.degrees(
                    math.atan2(
                        raven_values[8].item(),
                        raven_values[9].item(),
                    )
                )

                elevation_deg = math.degrees(
                    math.atan2(
                        raven_values[10].item(),
                        raven_values[11].item(),
                    )
                )

                print(
                    "[IK REACH] "
                    f"alpha={reach_alpha:4.2f} | "
                    f"palm=("
                    f"{left_palm_pos_w[0].item():5.2f}, "
                    f"{left_palm_pos_w[1].item():5.2f}, "
                    f"{left_palm_pos_w[2].item():5.2f}) | "
                    f"pregrasp=("
                    f"{current_pregrasp_pos_w[0].item():5.2f}, "
                    f"{current_pregrasp_pos_w[1].item():5.2f}, "
                    f"{current_pregrasp_pos_w[2].item():5.2f})"
                )

                print(
                    "[IK ERROR] "
                    f"palm->pregrasp="
                    f"{palm_to_pregrasp:5.3f} m | "
                    f"palm->handle="
                    f"{palm_to_handle:5.3f} m | "
                    f"root_drift="
                    f"{root_drift:7.4f} m"
                )

                print(
                    "[TRIPOD] "
                    f"pan={pan_deg:+6.2f} deg | "
                    f"tilt={tilt_deg:+6.2f} deg | "
                    f"camera_target_az="
                    f"{azimuth_deg:+6.1f} deg | "
                    f"camera_target_el="
                    f"{elevation_deg:+6.1f} deg"
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