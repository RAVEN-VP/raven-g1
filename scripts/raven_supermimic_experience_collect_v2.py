"""RAVEN: simulator experience collection with blended learned actor control.

Runs the UNMODIFIED raven_g1_actor_tracking.py moving-actor controller and
captures one post-physics state paired with the command applied for that step.

Output: raven_training_results/supermimic_capture/<timestamp>/
  raven_g1_state_action_120hz.jsonl  -- learning transitions, one line per step
  raven_g1_state_action.summary.json -- schema, joint ordering, provenance
  raven_actor_tracking.csv          -- original controller's 10 Hz log
  raven_actor_tracking.summary.json -- original controller's summary
  g1_rgbd_*.png                     -- original RGB-D diagnostic images

Assisted grasp and original tracker source are unchanged. Real simulator transitions
are collected under a bounded blend of an existing trained actor and the baseline.
A separate incremental learner consumes these transitions and updates weights.
"""
from __future__ import annotations

import hashlib
import ast
import math
import argparse
import random
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACKER = HERE / "raven_g1_actor_tracking.py"

# Read our arguments without confusing the underlying tracker parser.
EXTRA = argparse.ArgumentParser(add_help=False)
EXTRA.add_argument("--policy", type=Path, required=True)
EXTRA.add_argument("--online-out", type=Path, default=None)
EXTRA.add_argument("--blend", type=float, default=0.35)
EXTRA.add_argument("--seed", type=int, default=42)
EXTRA.add_argument("--motion-profile", choices=("sine", "triangle", "dwell", "multisine", "chirp"),
                   default="sine", help="Actor motion curriculum, applied without changing the validated tracker")
EXTRA.add_argument("--bearing-source", choices=("legacy", "pixel"), default="legacy",
                   help="legacy=original world bearing; pixel=experimental optical pixel bearing")
EXTRA.add_argument("--explore-noise-rad", type=float, default=0.0003)
explore_args, tracker_argv = EXTRA.parse_known_args(sys.argv[1:])
if not 0.0 < explore_args.blend <= 0.5:
    EXTRA.error("--blend must be in (0, 0.5]")
if not 0.0 <= explore_args.explore_noise_rad <= 0.001:
    EXTRA.error("--explore-noise-rad must be in [0, 0.001]")
random.seed(explore_args.seed)
tracker_hash = hashlib.sha256(TRACKER.read_bytes()).hexdigest()
EXPECTED_SHA = "2b7da3c47a8614b690bb20f2f461d0bb93a90a90b9f0c95a3d6c6b4648ecd038"
if tracker_hash != EXPECTED_SHA:
    raise SystemExit("Original validated tracker hash changed; refusing action injection")

if not TRACKER.is_file():
    raise SystemExit(f"Missing validated controller: {TRACKER}")

OUT = (explore_args.online_out or
       (Path.cwd() / "raven_training_results" / "supermimic_online_capture" /
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
OUT.mkdir(parents=True, exist_ok=False)

# Original arguments (--headless, --tracking-duration-s, etc.) still work.
# Force this experiment's output folder without touching other run folders.
sys.argv = [str(TRACKER), *tracker_argv, "--output-dir", str(OUT)]
spec = importlib.util.spec_from_file_location("raven_validated_tracker_capture", TRACKER)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {TRACKER}")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)  # Original AppLauncher starts Isaac Sim.

import torch

from raven_supermimic_actor_critic_v1 import Actor
from raven_supermimic_world_model_v2 import JOINTS

ckpt = torch.load(explore_args.policy.resolve(), map_location="cpu", weights_only=True)
if (ckpt.get("model_class") not in (
        "RAVENOfflineActorCriticV1_1", "RAVENOnlineActorCriticV2")
        or tuple(ckpt.get("joint_names", ())) != tuple(JOINTS)):
    raise SystemExit("Selected policy is not a compatible RAVEN actor checkpoint")
w = ckpt["actor_state_dict"]
actor = Actor(w["state_mean"], w["state_scale"])
actor.load_state_dict(w, strict=True)
actor.eval()
for param in actor.parameters():
    param.requires_grad_(False)

intervention = {"enabled": True, "applied": 0, "fallback": 0,
                "fallback_reasons": {"nonfinite_reference": 0, "nonfinite_action": 0,
                                     "joint_limit": 0, "joint_step_limit": 0,
                                     "exception": 0},
                "proposed": 0, "last_actor_used": False,
                "last_reference_abs_rad": None}


def blend_actor_action(command, robot, tripod, desired_pan, pan_zero,
                       phase, grip_attached):
    """Return safe bounded exploratory setpoints, preserving every other joint."""
    intervention["last_actor_used"] = False
    if (phase != "ACTOR_TRACKING" or not grip_attached
            or pan_zero is None or not intervention["enabled"]):
        return command
    intervention["proposed"] += 1
    try:
        names = list(robot.data.joint_names)
        ids = [names.index(name) for name in JOINTS]
        joint_names = list(tripod.data.joint_names)
        pan_id, tilt_id = joint_names.index("pan_joint"), joint_names.index("tilt_joint")
        q = robot.data.joint_pos[:, ids]
        dq = robot.data.joint_vel[:, ids]
        tripod_q = tripod.data.joint_pos[:, [pan_id, tilt_id]]
        tripod_dq = tripod.data.joint_vel[:, [pan_id, tilt_id]]
        state = torch.cat((q, dq, tripod_q, tripod_dq), dim=-1).detach()
        held = command[:, ids].detach()
        ref = float(pan_zero[0]) + math.radians(float(desired_pan))
        if not math.isfinite(ref):
            intervention["fallback"] += 1
            intervention["fallback_reasons"]["nonfinite_reference"] += 1
            return command
        intervention["last_reference_abs_rad"] = ref
        actor.to(state.device)
        with torch.inference_mode():
            predicted = actor(state, held, torch.full((1, 1), ref,
                                                       device=state.device,
                                                       dtype=state.dtype))
            proposal = held + explore_args.blend * (predicted - held)
            if explore_args.explore_noise_rad:
                # On the target delta, NOT on the measured state.
                noise = torch.randn_like(proposal) * explore_args.explore_noise_rad
                proposal = proposal + noise
            bounds = robot.data.joint_pos_limits[0, ids]
            step_bounds = torch.full_like(held, base.MAX_ARM_STEP)
            step_bounds[:, 0] = base.MAX_WAIST_STEP
            reason = None
            if not bool(torch.isfinite(proposal).all()):
                reason = "nonfinite_action"
            elif not (bool((proposal >= bounds[:, 0] + 1e-5).all())
                      and bool((proposal <= bounds[:, 1] - 1e-5).all())):
                reason = "joint_limit"
            elif not bool(((proposal - q).abs() <= step_bounds + 1e-6).all()):
                reason = "joint_step_limit"
            if reason:
                intervention["fallback"] += 1
                intervention["fallback_reasons"][reason] += 1
                return command
            result = command.clone()
            result[:, ids] = proposal
        intervention["applied"] += 1
        intervention["last_actor_used"] = True
        return result
    except Exception as exc:
        intervention["fallback"] += 1
        intervention["fallback_reasons"]["exception"] += 1
        if intervention["fallback_reasons"]["exception"] <= 5:
            print(f"[ONLINE COLLECTOR] Actor fallback: {type(exc).__name__}: {exc}")
        return command


def curriculum_y(t: float, motion_start: float | None) -> float:
    """Deterministic bounded lateral actor motion; zero before perception lock."""
    if motion_start is None or t < motion_start:
        return 0.0
    elapsed = max(0.0, t - motion_start)
    amplitude = float(base.args.actor_amplitude_m)
    period = float(base.args.actor_period_s)
    cycles = elapsed / period
    theta = 2.0 * math.pi * cycles
    profile = explore_args.motion_profile
    if profile == "sine":
        value = math.sin(theta)
    elif profile == "triangle":
        # Constant-speed crossing with sharp direction reversals.
        value = (2.0 / math.pi) * math.asin(math.sin(theta))
    elif profile == "dwell":
        # Longer stays at each side with smoothly bounded transitions.
        value = math.tanh(2.2 * math.sin(theta)) / math.tanh(2.2)
    elif profile == "multisine":
        # Mixed frequencies, still within the original displacement bounds.
        value = 0.70 * math.sin(theta) + 0.30 * math.sin(2.3 * theta)
    elif profile == "chirp":
        # Increasing reversal frequency, without discontinuous teleportation.
        value = math.sin(2.0 * math.pi * (cycles + 0.06 * cycles * cycles))
    else:
        raise RuntimeError(f"Unrecognised motion profile: {profile}")
    return max(-amplitude, min(amplitude, amplitude * value))


# Inject a single call in the ORIGINAL tracker loop without modifying its file.
source = ast.parse(TRACKER.read_text(encoding="utf-8"), filename=str(TRACKER))
functions = [node for node in source.body
             if isinstance(node, ast.FunctionDef) and node.name == "run"]
if len(functions) != 1:
    raise RuntimeError("Tracker run definition not unique")
run_ast = functions[0]
finds = [node for node in ast.walk(run_ast)
         if isinstance(node, ast.Expr)
         and ast.unparse(node) == "robot.set_joint_position_target(command)"]
if len(finds) != 1:
    raise RuntimeError("Validated joint command injection site not found")
# Replace only the original sinusoidal actor_Y expression in an in-memory AST.
# Nothing changes on disk in raven_g1_actor_tracking.py.
motion_assignments = [node for node in ast.walk(run_ast)
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "actor_y"
                              for target in node.targets)
                      and "args.actor_amplitude_m" in ast.unparse(node.value)
                      and "math.sin" in ast.unparse(node.value)]
if len(motion_assignments) != 1:
    raise RuntimeError("Validated actor trajectory injection site not found")
motion_assignments[0].value = ast.parse(
    "curriculum_y(t, actor_motion_start_time)", mode="eval").body
base.curriculum_y = curriculum_y
finds[0].value.args[0] = ast.parse(
    "blend_actor_action(command, robot, tripod, desired_pan, pan_zero, phase, grip_attached)",
    mode="eval"
).body
module = ast.fix_missing_locations(ast.Module(body=[run_ast], type_ignores=[]))
base.blend_actor_action = blend_actor_action
exec(compile(module, str(TRACKER), "exec"), base.__dict__)

original_run = base.run
original_detector = base.detect_actor
observation = {
    "detected": False, "bearing_deg": None, "u_px": None,
    "v_px": None, "median_depth_m": None, "pixels": None,
    "status": "not sampled", "read_after_physics_step": None,
    "legacy_bearing_deg": None, "pixel_bearing_deg": None,
}
read_count = 0
physics_steps = 0


def get_floats(tensor):
    """Snapshot values without retaining tensors from Isaac Lab's live buffers."""
    if tensor is None:
        return None
    with torch.no_grad():
        x = tensor.detach().cpu().reshape(-1).tolist()
    return [float(v) if math.isfinite(float(v)) else None for v in x]


def pose_of(asset, body_name):
    index = asset.data.body_names.index(body_name)
    return get_floats(asset.data.body_pose_w[0, index])


def read_camera(g1_cam, tripod, robot, torso_body):
    """Observe the original RGB-D detector's result; never change its output."""
    global read_count
    result = original_detector(g1_cam, tripod, robot, torso_body)
    detection, rgb, status = result
    legacy_bearing = float(detection["bearing_deg"]) if detection else None
    pixel_bearing = None
    if detection:
        intrinsics = g1_cam.data.intrinsic_matrices[0]
        fx = float(intrinsics[0, 0])
        cx = float(intrinsics[0, 2])
        if math.isfinite(fx) and fx > 1.0 and math.isfinite(cx):
            pixel_bearing = math.degrees(math.atan2(cx - float(detection["u"]), fx))
            if explore_args.bearing_source == "pixel":
                detection = dict(detection)
                detection["bearing_deg"] = pixel_bearing
                result = (detection, rgb, status)
    read_count += 1
    observation.update({
        "detected": detection is not None,
        "bearing_deg": float(detection["bearing_deg"]) if detection else None,
        "legacy_bearing_deg": legacy_bearing,
        "pixel_bearing_deg": pixel_bearing,
        "u_px": float(detection["u"]) if detection else None,
        "v_px": float(detection["v"]) if detection else None,
        "median_depth_m": float(detection["depth_m"]) if detection else None,
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
    n_robot, n_tripod = len(joint_names), len(tripod_names)
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
        indices = kwargs.get("joint_ids", args[0] if args else None)
        if indices is None:
            if len(values) != len(previous):
                raise RuntimeError("Unexpected joint target vector shape")
            return values
        if isinstance(indices, slice):
            indices = list(range(len(previous)))[indices]
        elif isinstance(indices, torch.Tensor):
            indices = [int(i) for i in indices.detach().cpu().reshape(-1).tolist()]
        else:
            indices = [int(i) for i in indices]
        if len(indices) != len(values):
            raise RuntimeError("Joint IDs / joint target count mismatch")
        updated = list(previous)
        for index, val in zip(indices, values):
            updated[index] = val
        return updated

    def record_robot_target(target, *args, **kwargs):
        nonlocal robot_target, fresh_robot_target, commands_seen
        result = original_robot_target(target, *args, **kwargs)
        robot_target = capture_setpoint(target, args, kwargs, robot_target)
        fresh_robot_target = True
        commands_seen += 1
        return result

    def record_tripod_target(target, *args, **kwargs):
        nonlocal tripod_target, fresh_tripod_target
        result = original_tripod_target(target, *args, **kwargs)
        tripod_target = capture_setpoint(target, args, kwargs, tripod_target)
        fresh_tripod_target = True
        return result

    def sample_after_update(*args, **kwargs):
        """After the physics step: pair response with the preceding target."""
        nonlocal fresh_robot_target, fresh_tripod_target
        global physics_steps
        result = original_scene_update(*args, **kwargs)
        physics_steps += 1
        age_steps = observation["read_after_physics_step"]
        age_s = ((physics_steps - age_steps) * dt
                 if age_steps is not None else None)
        # These states are post-step, while setpoints are commands for the
        # just-completed physics step. Angles and joint targets use radians.
        physics_pan = float(tripod.data.joint_pos[0, tripod_names.index("pan_joint")])
        physics_tilt = float(tripod.data.joint_pos[0, tripod_names.index("tilt_joint")])
        desired_abs = intervention["last_reference_abs_rad"]
        pan_error_deg = (math.degrees(physics_pan - desired_abs)
                         if desired_abs is not None else None)
        # This records the actual observed simulator feedback. Policy learning
        # currently uses differentiable model rewards; keep both signals.
        if abs(math.degrees(physics_tilt)) > 4.0:
            intervention["enabled"] = False
        record = {
            "physics_step": physics_steps,
            "sim_time_s": round(physics_steps * dt, 7),
            "control_dt_s": dt,
            "robot_joint_pos_rad": get_floats(robot.data.joint_pos[0]),
            "robot_joint_vel_rad_s": get_floats(robot.data.joint_vel[0]),
            "robot_applied_torque_nm": get_floats(robot.data.applied_torque[0])
                if getattr(robot.data, "applied_torque", None) is not None else None,
            "robot_joint_position_targets_rad": list(robot_target),
            "robot_target_updated_this_step": fresh_robot_target,
            "tripod_joint_pos_rad": get_floats(tripod.data.joint_pos[0]),
            "tripod_joint_vel_rad_s": get_floats(tripod.data.joint_vel[0]),
            "tripod_joint_stiffness_nm_rad": get_floats(tripod.data.joint_stiffness[0])
                if getattr(tripod.data, "joint_stiffness", None) is not None else None,
            "tripod_joint_damping_nm_s_rad": get_floats(tripod.data.joint_damping[0])
                if getattr(tripod.data, "joint_damping", None) is not None else None,
            "tripod_joint_position_targets_rad": list(tripod_target),
            "tripod_target_updated_this_step": fresh_tripod_target,
            "robot_root_pose_w_xyz_quat_wxyz": get_floats(robot.data.root_pose_w[0]),
            "torso_pose_w_xyz_quat_wxyz": pose_of(robot, "torso_link"),
            "hand_pose_w_xyz_quat_wxyz": pose_of(robot, "left_base_link"),
            "handle_pose_w_xyz_quat_wxyz": pose_of(tripod, "control_handle_link"),
            "tripod_root_pose_w_xyz_quat_wxyz": get_floats(tripod.data.root_pose_w[0]),
            "rgbd_detection": observation["detected"],
            "rgbd_actor_bearing_deg": observation["bearing_deg"],
            "rgbd_original_bearing_deg": observation["legacy_bearing_deg"],
            "rgbd_pixel_bearing_deg": observation["pixel_bearing_deg"],
            "rgbd_u_px": observation["u_px"],
            "rgbd_v_px": observation["v_px"],
            "rgbd_median_depth_m": observation["median_depth_m"],
            "rgbd_actor_pixels": observation["pixels"],
            "rgbd_status": observation["status"],
            "rgbd_measurement_read_age_s": round(age_s, 7) if age_s is not None else None,
            "rgbd_measurement_read_this_step": age_steps == physics_steps - 1,
            "actor_exploration_action": intervention["last_actor_used"],
            "raven_reference_abs_rad": desired_abs,
            "raven_actual_pan_error_deg": pan_error_deg,
            "raven_actual_tilt_deg": math.degrees(physics_tilt),
            "raven_instantaneous_physics_reward": (
                -(pan_error_deg ** 2 + 5.0 * math.degrees(physics_tilt) ** 2)
                if pan_error_deg is not None else None
            ),
        }
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        fresh_robot_target = False
        fresh_tripod_target = False
        if physics_steps % 1200 == 0:
            print(f"[RL CAPTURE] {physics_steps} physics steps recorded"
                  f" ({physics_steps * dt:.1f}s simulated)")
        return result

    print("[RL CAPTURE] 120 Hz full joint state/action logging enabled")
    print("[RL CAPTURE] Original source untouched; learned actor blended into tracking phase")
    print(f"[RL CAPTURE] Writing: {data_path}")
    with data_path.open("w", encoding="utf-8", buffering=1024 * 1024) as stream:
        try:
            robot.set_joint_position_target = record_robot_target
            tripod.set_joint_position_target = record_tripod_target
            scene.update = sample_after_update
            return original_run(sim, scene, actor_root)
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Restore original methods even after a safety abort / exception.
            scene.update = original_scene_update
            robot.set_joint_position_target = original_robot_target
            tripod.set_joint_position_target = original_tripod_target
            stream.flush()
            summary = {
                "experiment": "RAVEN 120Hz actor-guided online experience capture",
                "status": "ERROR" if error_message else "RECORDED",
                "error": error_message,
                "physics_hz": 1.0 / dt,
                "time_basis": "Post-simulation physics step; RGB-D age measured since read, not exposure",
                "samples": physics_steps,
                "g1_joint_count": n_robot,
                "g1_joint_order": joint_names,
                "tripod_joint_count": n_tripod,
                "tripod_joint_order": tripod_names,
                "g1_setpoint_calls": commands_seen,
                "perception_reads": read_count,
                "command_semantics": "G1 position targets issued for the physics step, not torque/force actions",
                "tripod_semantics": "The tracker physically frees pan and controls tilt; stiffness/damping are logged when Isaac Lab exposes them",
                "actor_position_privileged_to_controller": False,
                "assisted_grasp": True,
                "original_tracker_sha256": hashlib.sha256(TRACKER.read_bytes()).hexdigest(),
                "exploration_policy": str(explore_args.policy.resolve()),
                "motion_profile": explore_args.motion_profile,
                "bearing_source": explore_args.bearing_source,
                "motion_amplitude_m": float(base.args.actor_amplitude_m),
                "motion_period_s": float(base.args.actor_period_s),
                "motion_seed": explore_args.seed,
                "exploration_blend": explore_args.blend,
                "exploration_noise_rad": explore_args.explore_noise_rad,
                "training_usable": intervention["applied"] > 0,
                "exploration_disabled_on_tilt": not intervention["enabled"],
                "policy_steps_applied": intervention["applied"],
                "policy_fallbacks": intervention["fallback"],
                "policy_fallback_reasons": dict(intervention["fallback_reasons"]),
                "transition_jsonl": str(data_path),
                "original_10hz_csv": str(OUT / "raven_actor_tracking.csv"),
            }
            (OUT / "raven_g1_state_action.summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"[RL CAPTURE] Recorded {physics_steps} post-step samples")
            print(f"[RL CAPTURE] Summary: {OUT / 'raven_g1_state_action.summary.json'}")


base.detect_actor = read_camera
base.run = wrap_run

if __name__ == "__main__":
    try:
        base.main()
    finally:
        base.simulation_app.close()
