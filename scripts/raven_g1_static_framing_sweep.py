
"""RAVEN static cinematography sweep using the unchanged G1 tracking controller.

Requires scripts/raven_g1_actor_tracking.py in the SAME scripts directory.
No actor ground-truth coordinates are supplied to the tracking controller.
The new sensor supplies evaluation measurements only.

Isaac Sim 5.1 / Isaac Lab 2.3.2. Run using env_isaaclab.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path

# Keep the original tracker intact: it owns AppLauncher, the physics sequence,
# the assisted grasp, the IK, and its safety checks.
TRACKER = Path(__file__).with_name("raven_g1_actor_tracking.py")
if not TRACKER.is_file():
    raise SystemExit(f"Cannot find the validated tracker: {TRACKER}")

POSITIONS_M = (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45)
HOLD_S = 6.0
DISCARD_INITIAL_S = 2.0
WAIT_AFTER_REFERENCE_S = 3.0
MIN_TRACKING_START_S = 4.0
EXPERIMENT_TIMEOUT_S = 65.0
CAMERA_POS = (0.38, 0.0, 0.03)
CAMERA_QUAT_WXYZ = (
    0.9993191285690909,
    0.000644302332549033,
    -0.021456229334608615,
    0.030008238421435435,
)

OUT = (Path.cwd() / "raven_tracking_results" / "static_framing_sweep" /
       datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT.mkdir(parents=True, exist_ok=True)

# Original CLI options (--headless, --device, etc.) are passed to its AppLauncher.
# Its normal 35-second run is extended. It will be shortened automatically as
# soon as a valid detection establishes the sweep start time.
sys.argv = [str(TRACKER), *sys.argv[1:],
            "--tracking-duration-s", str(EXPERIMENT_TIMEOUT_S),
            "--output-dir", str(OUT)]
spec = importlib.util.spec_from_file_location("raven_validated_tracker", TRACKER)
if spec is None or spec.loader is None:
    raise SystemExit("Cannot load original tracking module")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)  # starts Isaac Sim via the validated AppLauncher

import torch
from isaaclab.utils import configclass


@configclass
class SweepSceneCfg(base.SceneCfg):
    # Optical mount orientation from the successful +3.44 deg yaw / +2.46 deg
    # pitch experiment. 'world' camera convention means +X optical forward.
    tripod_rgb = base.CameraCfg(
        prim_path="{ENV_REGEX_NS}/Tripod/camera_link/raven_sweep_rgb",
        update_period=0.1,
        height=360,
        width=640,
        data_types=["rgb"],
        spawn=base.sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
            focus_distance=3.0,
        ),
        offset=base.CameraCfg.OffsetCfg(
            pos=CAMERA_POS,
            rot=CAMERA_QUAT_WXYZ,
            convention="world",
        ),
    )


base.SceneCfg = SweepSceneCfg
old_move_actor = base.move_actor
old_detect_actor = base.detect_actor
old_run = base.run

state = {
    "move_call": -1, "time_s": 0.0, "start_rel_s": None,
    "reference_time_s": None, "reference_bearing_deg": None,
    "last_stage": None, "tripod_camera": None,
    "initial_pan_rad": None, "last_pan": None, "last_t": None,
    "grip_relative": None, "rows": [], "saved": set(),
    "vision_calls": 0, "finished": False,
}


def phase_at(t):
    """Returns (index, seconds_at_position) or (None, None)."""
    start = state["start_rel_s"]
    if start is None or t - base.PAN_START_S < start:
        return None, None
    elapsed = t - base.PAN_START_S - start
    index = min(int(max(0.0, elapsed) / HOLD_S), len(POSITIONS_M) - 1)
    return index, elapsed - index * HOLD_S


def scheduled_actor_move(root, ignored_original_lateral):
    # One call from original main() positions the actor before the simulation.
    # Every subsequent call corresponds to the beginning of a physics frame.
    state["move_call"] += 1
    state["time_s"] = max(0, state["move_call"] - 1) * base.DT
    t = state["time_s"]
    if state["start_rel_s"] is None and state["reference_time_s"] is not None:
        state["start_rel_s"] = max(
            MIN_TRACKING_START_S,
            state["reference_time_s"] - base.PAN_START_S + WAIT_AFTER_REFERENCE_S,
        )
        # End a second after the last six-second hold.
        base.args.tracking_duration_s = state["start_rel_s"] + len(POSITIONS_M) * HOLD_S + 1.0
        print(f"[SWEEP] Start at tracking t={state['start_rel_s']:.2f}s; "
              f"seven positions, {HOLD_S:.1f}s each")
    index, at_position = phase_at(t)
    lateral = 0.0 if index is None else POSITIONS_M[index]
    if index is not None and state["last_stage"] != index:
        print(f"[SWEEP] Position {index + 1}/{len(POSITIONS_M)} "
              f"at y={lateral:+.2f}m; disregard first {DISCARD_INITIAL_S:.1f}s")
        state["last_stage"] = index
    old_move_actor(root, lateral)


def detect_tripod_target():
    sensor = state["tripod_camera"]
    data = sensor.data
    rgb = data.output["rgb"][0, ..., :3]
    if rgb.dtype != torch.uint8:
        rgb = (rgb * 255.0).clamp(0, 255).to(torch.uint8) if (
            float(torch.nan_to_num(rgb).max()) <= 1.0
        ) else rgb.clamp(0, 255).to(torch.uint8)
    r, g, b = (rgb[..., i].float() for i in range(3))
    mask = (r > 90) & (b > 65) & (r > g * 1.5) & (b > g * 1.3)
    pixels = torch.nonzero(mask, as_tuple=False)
    if len(pixels) < 80:
        return None, rgb
    K = data.intrinsic_matrices[0]
    if not bool(torch.isfinite(K).all()) or float(K[0, 0]) <= 1 or float(K[1, 1]) <= 1:
        return None, rgb
    u = float(pixels[:, 1].float().mean())
    v = float(pixels[:, 0].float().mean())
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return {
        "horizontal_error_deg": math.degrees(math.atan2(u - cx, fx)),
        "vertical_error_deg": math.degrees(math.atan2(v - cy, fy)),
        "u_px": u, "v_px": v, "cx_px": cx, "cy_px": cy,
        "pixels": len(pixels),
        "bbox": (int(pixels[:, 1].min()), int(pixels[:, 0].min()),
                 int(pixels[:, 1].max()), int(pixels[:, 0].max())),
    }, rgb


def save_example(index, rgb, measurement):
    if index in state["saved"] or not base.PIL_AVAILABLE:
        return
    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb.detach().cpu().numpy())
    draw = ImageDraw.Draw(im)
    width, height = im.size
    cx, cy = measurement["cx_px"], measurement["cy_px"]
    u, v = measurement["u_px"], measurement["v_px"]
    draw.line([(cx, 0), (cx, height)], fill=(255, 255, 255), width=2)
    draw.line([(0, cy), (width, cy)], fill=(255, 255, 255), width=2)
    draw.rectangle(measurement["bbox"], outline=(50, 255, 100), width=3)
    draw.ellipse([(u - 5, v - 5), (u + 5, v + 5)], outline=(50, 255, 100), width=2)
    draw.text((10, 10), f"STATIC {POSITIONS_M[index]:+.2f}m  "
              f"H={measurement['horizontal_error_deg']:+.3f}deg", fill=(255, 255, 255))
    im.save(OUT / f"static_actor_{index + 1:02d}.png")
    state["saved"].add(index)


def logged_detection(g1_cam, tripod, robot, torso_body):
    detected, rgb, status = old_detect_actor(g1_cam, tripod, robot, torso_body)
    t = state["time_s"]
    if detected is not None and state["reference_bearing_deg"] is None:
        state["reference_bearing_deg"] = float(detected["bearing_deg"])
        state["reference_time_s"] = t
    if t < base.PAN_START_S:
        return detected, rgb, status

    state["vision_calls"] += 1
    stage, elapsed = phase_at(t)
    tripod_detection, tripod_rgb = detect_tripod_target()
    pan_id = tripod.data.joint_names.index("pan_joint")
    tilt_id = tripod.data.joint_names.index("tilt_joint")
    now_pan = float(tripod.data.joint_pos[0, pan_id])
    if state["initial_pan_rad"] is None:
        state["initial_pan_rad"] = now_pan
    actual_pan = math.degrees(now_pan - state["initial_pan_rad"])
    rate = None
    if state["last_pan"] is not None and t > state["last_t"]:
        rate = math.degrees((now_pan - state["last_pan"]) / (t - state["last_t"]))
    state["last_pan"], state["last_t"] = now_pan, t

    joints = {n: i for i, n in enumerate(robot.data.joint_names)}
    bodies = {n: i for i, n in enumerate(robot.data.body_names)}
    handle = tripod.data.body_pose_w[:, tripod.data.body_names.index("control_handle_link")]
    hand = robot.data.body_pose_w[:, bodies["left_base_link"]]
    if state["grip_relative"] is None:
        state["grip_relative"] = base.subtract_frame_transforms(
            handle[:, :3], handle[:, 3:7], hand[:, :3], hand[:, 3:7])
    rel_pos, rel_quat = state["grip_relative"]
    expected_pos, _ = base.combine_frame_transforms(
        handle[:, :3], handle[:, 3:7], rel_pos, rel_quat)
    grasp_drift = float(torch.linalg.vector_norm(hand[:, :3] - expected_pos, dim=-1)[0])

    evaluate = stage is not None and DISCARD_INITIAL_S <= elapsed < HOLD_S
    row = {
        "tracking_time_s": round(t - base.PAN_START_S, 5),
        "stage_index": stage + 1 if stage is not None else "",
        "actor_y_ground_truth_m": POSITIONS_M[stage] if stage is not None else 0.0,
        "time_at_position_s": round(elapsed, 4) if elapsed is not None else "",
        "evaluation_sample": evaluate,
        "g1_detected": detected is not None,
        "tripod_detected": tripod_detection is not None,
        "g1_target_bearing_deg": (
            round(max(-base.args.sweep_deg, min(base.args.sweep_deg,
                detected["bearing_deg"] - state["reference_bearing_deg"])), 5)
            if detected is not None and state["reference_bearing_deg"] is not None else ""
        ),
        "actual_pan_deg": round(actual_pan, 5),
        "pan_velocity_deg_s": round(rate, 5) if rate is not None else "",
        "tilt_deg": round(math.degrees(float(tripod.data.joint_pos[0, tilt_id])), 5),
        "waist_yaw_deg": round(math.degrees(float(robot.data.joint_pos[0, joints["waist_yaw_joint"]])), 5),
        "left_elbow_deg": round(math.degrees(float(robot.data.joint_pos[0, joints["left_elbow_joint"]])), 5),
        "assisted_grasp_relative_drift_m": round(grasp_drift, 7),
        "horizontal_error_deg": (
            round(tripod_detection["horizontal_error_deg"], 5) if tripod_detection else ""),
        "vertical_error_deg": (
            round(tripod_detection["vertical_error_deg"], 5) if tripod_detection else ""),
        "actor_pixels": tripod_detection["pixels"] if tripod_detection else "",
    }
    state["rows"].append(row)
    if evaluate and elapsed >= 3.0 and tripod_detection is not None:
        save_example(stage, tripod_rgb, tripod_detection)
    return detected, rgb, status


def hooked_run(sim, scene, actor_root):
    state["tripod_camera"] = scene["tripod_rgb"]
    print("[SWEEP] Calibrated tripod RGB mounted for EVALUATION ONLY")
    return old_run(sim, scene, actor_root)


base.move_actor = scheduled_actor_move
base.detect_actor = logged_detection
base.run = hooked_run


def make_stats(values):
    v = [float(x) for x in values if isinstance(x, (float, int)) and math.isfinite(x)]
    if not v:
        return None
    errors = sorted(abs(x) for x in v)
    k = 0.95 * (len(errors) - 1)
    p95 = errors[int(k)] + (errors[min(int(k) + 1, len(errors) - 1)] - errors[int(k)]) * (k - int(k))
    return {
        "n": len(v), "signed_mean_deg": statistics.mean(v),
        "mae_deg": statistics.mean(errors),
        "rmse_deg": math.sqrt(statistics.mean(x*x for x in v)),
        "p95_abs_deg": p95, "max_abs_deg": errors[-1],
    }


def summarise():
    rows = state["rows"]
    headers = [
        "tracking_time_s", "stage_index", "actor_y_ground_truth_m", "time_at_position_s",
        "evaluation_sample", "g1_detected", "tripod_detected", "g1_target_bearing_deg",
        "actual_pan_deg", "pan_velocity_deg_s", "tilt_deg", "waist_yaw_deg",
        "left_elbow_deg", "assisted_grasp_relative_drift_m", "horizontal_error_deg",
        "vertical_error_deg", "actor_pixels",
    ]
    with (OUT / "raven_static_framing.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    stages = []
    for i, y in enumerate(POSITIONS_M, start=1):
        subset = [r for r in rows if r["stage_index"] == i and r["evaluation_sample"]]
        detected = [r for r in subset if r["tripod_detected"]]
        values = [r["horizontal_error_deg"] for r in detected]
        result = {
            "stage": i, "actor_y_m": y, "evaluation_samples": len(subset),
            "tripod_detected_samples": len(detected),
            "detection_rate": len(detected)/len(subset) if subset else None,
            "horizontal": make_stats(values),
            "vertical": make_stats([r["vertical_error_deg"] for r in detected]),
            "mean_abs_pan_velocity_deg_s": (
                statistics.mean(abs(r["pan_velocity_deg_s"]) for r in subset
                                if r["pan_velocity_deg_s"] != "") if any(
                                    r["pan_velocity_deg_s"] != "" for r in subset) else None),
            "elbow_range_deg": (
                [min(r["left_elbow_deg"] for r in subset),
                 max(r["left_elbow_deg"] for r in subset)] if subset else None),
            "max_grasp_relative_drift_mm": (
                1000 * max(r["assisted_grasp_relative_drift_m"] for r in subset) if subset else None),
        }
        stages.append(result)
        stats = result["horizontal"]
        if stats is None:
            print(f"[SWEEP] y={y:+.2f}m: NO EVALUATION DETECTIONS")
        else:
            print(f"[SWEEP] y={y:+.2f}m: n={stats['n']:2d} "
                  f"bias={stats['signed_mean_deg']:+.3f}deg "
                  f"MAE={stats['mae_deg']:.3f}deg "
                  f"pan-rate={result['mean_abs_pan_velocity_deg_s']:.3f}deg/s")

    samples = [r["horizontal_error_deg"] for r in rows
               if r["evaluation_sample"] and r["tripod_detected"]]
    report = {
        "experiment": "RAVEN stationary actor sweep with independent tripod RGB",
        "completed_all_positions": all(s["horizontal"] is not None for s in stages),
        "tracking_controller_unchanged": True,
        "assisted_grasp": True,
        "actor_ground_truth_used_for_control": False,
        "mount_position_local_m": CAMERA_POS,
        "mount_quaternion_wxyz": CAMERA_QUAT_WXYZ,
        "per_position": stages,
        "aggregate_horizontal": make_stats(samples),
        "csv": str(OUT / "raven_static_framing.csv"),
        "notes": ["These are stationary-position measurements, not latency estimates.",
                  "Error is image-space target-centroid angle (positive toward image-right).",
                  "A 2-second settling interval is excluded after every move.",
                  "The baseline tracker CSV actor_y_ground_truth_m column reflects its unused sinusoidal plan; use the new sweep CSV for actual actor positions.",
                  "Check the centre-position result against the previous ~0.144 deg static bias before comparing runs.",
                  "The original tracker is imported but not rewritten."],
    }
    (OUT / "raven_static_framing.summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("[SWEEP] Outputs:", OUT)
    print("[SWEEP] Use raven_static_framing.csv for actual actor positions; "
          "the baseline tracker CSV still logs its unused sinusoidal actor path.")
    if not report["completed_all_positions"]:
        print("[SWEEP] WARNING: Missing samples from some positions; do not treat as calibrated data.")


if __name__ == "__main__":
    try:
        base.main()
    finally:
        try:
            summarise()
        finally:
            base.simulation_app.close()
