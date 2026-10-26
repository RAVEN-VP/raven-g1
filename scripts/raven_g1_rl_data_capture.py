
"""RAVEN 120 Hz G1-tripod state/action recorder (offline RL data collection).

Runs the UNMODIFIED raven_g1_actor_tracking.py moving-actor controller and
captures one post-physics state paired with the command applied for that step.

Output: raven_training_results/supermimic_capture/<timestamp>/
  raven_g1_state_action_120hz.jsonl
  raven_g1_state_action.summary.json
  raven_actor_tracking.csv
  raven_actor_tracking.summary.json
  g1_rgbd_*.png

Assisted grasp and IK are unchanged. This is data collection, NOT RL training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACKER = HERE / "raven_g1_actor_tracking.py"
if not TRACKER.is_file():
    raise SystemExit(f"Missing validated controller: {TRACKER}")

OUT = (
    Path.cwd()
    / "raven_training_results"
    / "supermimic_capture"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)

sys.argv = [str(TRACKER), *sys.argv[1:], "--output-dir", str(OUT)]

spec = importlib.util.spec_from_file_location(
    "raven_validated_tracker_capture", TRACKER
)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {TRACKER}")

base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

import torch

original_run = base.run
original_detector = base.detect_actor

observation = {
    "detected": False,
    "bearing_deg": None,
    "u_px": None,
    "v_px": None,
    "median_depth_m": None,
    "pixels": None,
    "status": "not sampled",
    "read_after_physics_step": None,
}

read_count = 0
physics_steps = 0


def get_floats(tensor):
    """Snapshot live Isaac Lab data without retaining tensor references."""
    if tensor is None:
        return None
    with torch.no_grad():
        values = tensor.detach().cpu().reshape(-1).tolist()
    return [
        float(v) if math.isfinite(float(v)) else None
        for v in values
    ]


def pose_of(asset, body_name):
    index = asset.data.body_names.index(body_name)
    return get_floats(asset.data.body_pose_w[0, index])


def read_camera(g1_cam, tripod, robot, torso_body):
    """Record perception without modifying the original detector."""
    global read_count

    result = original_detector(g1_cam, tripod, robot, torso_body)
    detection, _rgb, status = result
    read_count += 1

    observation.update({
        "detected": detection is not None,
        "bearing_deg": (
            float(detection["bearing_deg"]) if detection else None
        ),
        "u_px": float(detection["u"]) if detection else None,
        "v_px": float(detection["v"]) if detection else None,
        "median_depth_m": (
            float(detection["depth_m"]) if detection else None
        ),
        "pixels": int(detection["pixels"]) if detection else None,
        "status": status,
        "read_after_physics_step": physics_steps,
    })

    return result


def wrap_run(sim, scene, actor_root):
    global physics_steps

    robot = scene["robot"]
    tripod = scene["tripod"]

    joint_names = list(robot.data.joint_names)
    tripod_names = list(tripod.data.joint_names)

    n_robot = len(joint_names)
    n_tripod = len(tripod_names)
    dt = float(sim.get_physics_dt())

    original_robot_target = robot.set_joint_position_target
    original_tripod_target = tripod.set_joint_position_target
    original_scene_update = scene.update

    robot_target = [None] * n_robot
    tripod_target = [None] * n_tripod
    fresh_robot_target = False
    fresh_tripod_target = False
    commands_seen = 0

    data_path = OUT / "raven_g1_state_action_120hz.jsonl"
    error_message = None

    def capture_setpoint(target, args, kwargs, previous):
        values = get_floats(target)
        indices = kwargs.get(
            "joint_ids", args[0] if args else None
        )

        if indices is None:
            if len(values) != len(previous):
                raise RuntimeError(
                    "Unexpected joint target vector shape"
                )
            return values

        if isinstance(indices, slice):
            indices = list(range(len(previous)))[indices]
        elif isinstance(indices, torch.Tensor):
            indices = [
                int(i)
                for i in indices.detach().cpu().reshape(-1).tolist()
            ]
        else:
            indices = [int(i) for i in indices]

        if len(indices) != len(values):
            raise RuntimeError(
                "Joint IDs / joint target count mismatch"
            )

        updated = list(previous)
        for index, val in zip(indices, values):
            updated[index] = val

        return updated

    def record_robot_target(target, *args, **kwargs):
        nonlocal robot_target, fresh_robot_target, commands_seen

        result = original_robot_target(target, *args, **kwargs)

        robot_target = capture_setpoint(
            target, args, kwargs, robot_target
        )
        fresh_robot_target = True
        commands_seen += 1

        return result

    def record_tripod_target(target, *args, **kwargs):
        nonlocal tripod_target, fresh_tripod_target

        result = original_tripod_target(target, *args, **kwargs)

        tripod_target = capture_setpoint(
            target, args, kwargs, tripod_target
        )
        fresh_tripod_target = True

        return result

    def sample_after_update(*args, **kwargs):
        """Pair the applied joint targets with the post-step state."""
        nonlocal fresh_robot_target, fresh_tripod_target
        global physics_steps

        result = original_scene_update(*args, **kwargs)
        physics_steps += 1

        age_steps = observation["read_after_physics_step"]
        age_s = (
            (physics_steps - age_steps) * dt
            if age_steps is not None else None
        )

        record = {
            "physics_step": physics_steps,
            "sim_time_s": round(physics_steps * dt, 7),
            "control_dt_s": dt,

            "robot_joint_pos_rad":
                get_floats(robot.data.joint_pos[0]),

            "robot_joint_vel_rad_s":
                get_floats(robot.data.joint_vel[0]),

            "robot_applied_torque_nm": (
                get_floats(robot.data.applied_torque[0])
                if getattr(robot.data, "applied_torque", None)
                is not None else None
            ),

            "robot_joint_position_targets_rad":
                list(robot_target),

            "robot_target_updated_this_step":
                fresh_robot_target,

            "tripod_joint_pos_rad":
                get_floats(tripod.data.joint_pos[0]),

            "tripod_joint_vel_rad_s":
                get_floats(tripod.data.joint_vel[0]),

            "tripod_joint_stiffness_nm_rad": (
                get_floats(tripod.data.joint_stiffness[0])
                if getattr(tripod.data, "joint_stiffness", None)
                is not None else None
            ),

            "tripod_joint_damping_nm_s_rad": (
                get_floats(tripod.data.joint_damping[0])
                if getattr(tripod.data, "joint_damping", None)
                is not None else None
            ),

            "tripod_joint_position_targets_rad":
                list(tripod_target),

            "tripod_target_updated_this_step":
                fresh_tripod_target,

            "robot_root_pose_w_xyz_quat_wxyz":
                get_floats(robot.data.root_pose_w[0]),

            "torso_pose_w_xyz_quat_wxyz":
                pose_of(robot, "torso_link"),

            "hand_pose_w_xyz_quat_wxyz":
                pose_of(robot, "left_base_link"),

            "handle_pose_w_xyz_quat_wxyz":
                pose_of(tripod, "control_handle_link"),

            "tripod_root_pose_w_xyz_quat_wxyz":
                get_floats(tripod.data.root_pose_w[0]),

            "rgbd_detection":
                observation["detected"],

            "rgbd_actor_bearing_deg":
                observation["bearing_deg"],

            "rgbd_u_px":
                observation["u_px"],

            "rgbd_v_px":
                observation["v_px"],

            "rgbd_median_depth_m":
                observation["median_depth_m"],

            "rgbd_actor_pixels":
                observation["pixels"],

            "rgbd_status":
                observation["status"],

            "rgbd_measurement_read_age_s": (
                round(age_s, 7) if age_s is not None else None
            ),

            "rgbd_measurement_read_this_step":
                age_steps == physics_steps - 1,
        }

        stream.write(
            json.dumps(
                record,
                allow_nan=False,
                separators=(",", ":")
            ) + "\n"
        )

        fresh_robot_target = False
        fresh_tripod_target = False

        if physics_steps % 1200 == 0:
            print(
                f"[RL CAPTURE] {physics_steps} physics steps recorded"
                f" ({physics_steps * dt:.1f}s simulated)"
            )

        return result

    print("[RL CAPTURE] 120 Hz joint state/action logging enabled")
    print("[RL CAPTURE] Original tracking, grasp and IK untouched")
    print(f"[RL CAPTURE] Writing: {data_path}")

    with data_path.open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as stream:
        try:
            robot.set_joint_position_target = record_robot_target
            tripod.set_joint_position_target = record_tripod_target
            scene.update = sample_after_update

            return original_run(sim, scene, actor_root)

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            raise

        finally:
            scene.update = original_scene_update
            robot.set_joint_position_target = original_robot_target
            tripod.set_joint_position_target = original_tripod_target

            stream.flush()

            summary = {
                "experiment":
                    "RAVEN 120Hz state-action transition capture",

                "status":
                    "ERROR" if error_message else "RECORDED",

                "error": error_message,

                "physics_hz": 1.0 / dt,

                "time_basis":
                    "Post-physics state; perception age since read, not exposure",

                "samples": physics_steps,

                "g1_joint_count": n_robot,
                "g1_joint_order": joint_names,

                "tripod_joint_count": n_tripod,
                "tripod_joint_order": tripod_names,

                "g1_setpoint_calls": commands_seen,
                "perception_reads": read_count,

                "command_semantics":
                    "Position targets, not directly commanded torques",

                "tripod_semantics":
                    "Pan is passive during tracking; actuator stiffness "
                    "and damping recorded when available",

                "actor_position_privileged_to_controller": False,
                "assisted_grasp": True,

                "original_tracker_sha256":
                    hashlib.sha256(TRACKER.read_bytes()).hexdigest(),

                "transition_jsonl": str(data_path),
                "original_10hz_csv":
                    str(OUT / "raven_actor_tracking.csv"),
            }

            summary_path = (
                OUT / "raven_g1_state_action.summary.json"
            )

            summary_path.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8"
            )

            print(
                f"[RL CAPTURE] Recorded {physics_steps} "
                "post-step samples"
            )
            print(f"[RL CAPTURE] Summary: {summary_path}")


base.detect_actor = read_camera
base.run = wrap_run


if __name__ == "__main__":
    try:
        base.main()
    finally:
        base.simulation_app.close()
