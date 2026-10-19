
"""RAVEN - instrumented reverse stationary sweep (no controller modifications).

Requires these UNCHANGED scripts in the same directory:
  raven_g1_actor_tracking.py
  raven_g1_static_framing_sweep.py

Records G1 RGB-D centroid/depth, reconstructed target bearing, robot torso
pose, tripod pan/tilt and framing measurements. Designed to diagnose the
oscillation observed late in the reverse sweep. The original controller still
uses only its existing perception output; additional data is for logging.
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

HERE = Path(__file__).resolve().parent
SWEEP_PATH = HERE / "raven_g1_static_framing_sweep.py"
TRACKER_PATH = HERE / "raven_g1_actor_tracking.py"
for path in (SWEEP_PATH, TRACKER_PATH):
    if not path.is_file():
        raise SystemExit(f"Required validated script is missing: {path}")

# Importing the sweep starts Isaac Sim (AppLauncher), but not its run loop.
spec = importlib.util.spec_from_file_location("raven_static_sweep_diagnostic", SWEEP_PATH)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot load sweep module: {SWEEP_PATH}")
sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sweep
spec.loader.exec_module(sweep)

# Exact reverse sequence that reproduced the oscillation.
sweep.POSITIONS_M = (+0.45, +0.30, +0.15, 0.00, -0.15, -0.30, -0.45, 0.00)
sweep.OUT = (Path.cwd() / "raven_tracking_results" / "reverse_vision_diagnostics"
             / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
sweep.OUT.mkdir(parents=True, exist_ok=False)
sweep.base.args.output_dir = str(sweep.OUT)

# The sweep has already wrapped the ORIGINAL detector. We wrap that wrapper,
# so existing camera evaluation and controller outputs remain intact.
original_sweep_detector = sweep.base.detect_actor
telemetry = []


def scalar(value):
    """Safely convert a scalar tensor / Python number to a finite float."""
    if value is None:
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def pose_euler_deg(quat):
    """World-frame roll, pitch, yaw for a WXYZ quaternion."""
    w, x, y, z = [float(v) for v in quat]
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    sinp = max(-1.0, min(1.0, 2 * (w*y - z*x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def instrumented_detector(g1_camera, tripod, robot, torso_body):
    detection, rgb, reason = original_sweep_detector(
        g1_camera, tripod, robot, torso_body
    )
    t = sweep.state["time_s"] - sweep.base.PAN_START_S
    if t < 0:
        return detection, rgb, reason

    # The sweep wrapper logs its own synchronized evaluation measurements.
    last = sweep.state["rows"][-1] if sweep.state["rows"] else {}
    torso = robot.data.body_pose_w[0, torso_body]
    roll, pitch, yaw = pose_euler_deg(torso[3:7])
    k = g1_camera.data.intrinsic_matrices[0]
    fx = scalar(k[0, 0])
    cx = scalar(k[0, 2])
    u = scalar(detection["u"]) if detection else None
    image_angle = (math.degrees(math.atan2(u - cx, fx)) if
                   u is not None and fx is not None and cx is not None and fx > 0 else None)

    reference = sweep.state["reference_bearing_deg"]
    relative_bearing = (float(detection["bearing_deg"]) - reference
                        if detection is not None and reference is not None else None)

    row = {
        "tracking_time_s": round(t, 5),
        "stage_index": last.get("stage_index", ""),
        "actor_y_evaluation_only_m": last.get("actor_y_ground_truth_m", ""),
        "time_at_position_s": last.get("time_at_position_s", ""),
        "g1_detected": detection is not None,
        "g1_status": reason,
        "g1_u_px": u,
        "g1_v_px": scalar(detection["v"]) if detection else None,
        "g1_image_horizontal_angle_deg": image_angle,
        "g1_median_depth_m": scalar(detection["depth_m"]) if detection else None,
        "g1_colour_pixel_count": detection["pixels"] if detection else None,
        "g1_raw_bearing_deg": scalar(detection["bearing_deg"]) if detection else None,
        "g1_relative_bearing_deg": relative_bearing,
        "torso_x_m": scalar(torso[0]),
        "torso_y_m": scalar(torso[1]),
        "torso_z_m": scalar(torso[2]),
        "torso_roll_deg": roll,
        "torso_pitch_deg": pitch,
        "torso_yaw_deg": yaw,
        "actual_tripod_pan_deg": last.get("actual_pan_deg", ""),
        "actual_tripod_tilt_deg": last.get("tilt_deg", ""),
        "pan_velocity_deg_s": last.get("pan_velocity_deg_s", ""),
        "waist_yaw_deg": last.get("waist_yaw_deg", ""),
        "left_elbow_deg": last.get("left_elbow_deg", ""),
        "cinema_horizontal_error_deg": last.get("horizontal_error_deg", ""),
        "cinema_vertical_error_deg": last.get("vertical_error_deg", ""),
    }
    telemetry.append(row)
    return detection, rgb, reason


sweep.base.detect_actor = instrumented_detector


def finite_field(rows, field):
    return [v for row in rows if (v := scalar(row.get(field))) is not None]


def field_spread(rows, field):
    v = finite_field(rows, field)
    return {
        "count": len(v),
        "mean": statistics.mean(v) if v else None,
        "std_dev": statistics.stdev(v) if len(v) > 1 else None,
        "min": min(v) if v else None,
        "max": max(v) if v else None,
    }


def write_diagnostics():
    csv_path = sweep.OUT / "raven_g1_vision_pose.csv"
    fields = [
        "tracking_time_s", "stage_index", "actor_y_evaluation_only_m",
        "time_at_position_s", "g1_detected", "g1_status", "g1_u_px", "g1_v_px",
        "g1_image_horizontal_angle_deg", "g1_median_depth_m",
        "g1_colour_pixel_count", "g1_raw_bearing_deg", "g1_relative_bearing_deg",
        "torso_x_m", "torso_y_m", "torso_z_m", "torso_roll_deg",
        "torso_pitch_deg", "torso_yaw_deg", "actual_tripod_pan_deg",
        "actual_tripod_tilt_deg", "pan_velocity_deg_s", "waist_yaw_deg",
        "left_elbow_deg", "cinema_horizontal_error_deg", "cinema_vertical_error_deg",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(telemetry)

    stage_summaries = []
    for stage, y in enumerate(sweep.POSITIONS_M, start=1):
        # Compare the SAME 2-6s evaluation window as previous experiments.
        rows = [r for r in telemetry if r["stage_index"] == stage
                and 2.0 <= (scalar(r["time_at_position_s"]) or -1.0) < 6.0]
        result = {
            "stage": stage, "actor_y_m": y, "samples": len(rows),
            "g1_bearing": field_spread(rows, "g1_relative_bearing_deg"),
            "g1_image_angle": field_spread(rows, "g1_image_horizontal_angle_deg"),
            "g1_depth": field_spread(rows, "g1_median_depth_m"),
            "torso_yaw": field_spread(rows, "torso_yaw_deg"),
            "torso_pitch": field_spread(rows, "torso_pitch_deg"),
            "pan": field_spread(rows, "actual_tripod_pan_deg"),
            "tilt": field_spread(rows, "actual_tripod_tilt_deg"),
            "cinema_horizontal_error": field_spread(rows, "cinema_horizontal_error_deg"),
        }
        stage_summaries.append(result)
        bearing_std = result["g1_bearing"]["std_dev"]
        image_std = result["g1_image_angle"]["std_dev"]
        depth_std = result["g1_depth"]["std_dev"]
        print(f"[VISION DIAG] stage={stage} y={y:+.2f}m "
              f"bearing_sd={bearing_std if bearing_std is not None else float('nan'):.3f}deg "
              f"pixel_angle_sd={image_std if image_std is not None else float('nan'):.3f}deg "
              f"depth_sd={depth_std if depth_std is not None else float('nan'):.3f}m")

    report = {
        "experiment": "RAVEN reverse sweep G1 perception and torso diagnosis",
        "controller_unchanged": True,
        "camera_mount_unchanged": True,
        "actor_ground_truth_used_for_control": False,
        "stages": stage_summaries,
        "vision_csv": str(csv_path),
        "original_sweep_csv": str(sweep.OUT / "raven_static_framing.csv"),
        "analysis_notes": [
            "High bearing variation with high image-angle variation suggests optical pixel motion or changing camera viewpoint.",
            "High bearing variation with stable pixels suggests pose/depth reconstruction or transform sensitivity.",
            "Stage comparisons are diagnostic correlations; they do not prove causation.",
            "The RGB-D values are sampled during the existing 10 Hz perception cycle.",
        ],
    }
    summary_path = sweep.OUT / "raven_g1_vision_pose.summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[VISION DIAG] CSV: {csv_path}")
    print(f"[VISION DIAG] Summary: {summary_path}")


if __name__ == "__main__":
    print("[RAVEN] Instrumented reverse sweep: controller/grasp/IK unchanged")
    print(f"[RAVEN] Output: {sweep.OUT}")
    try:
        sweep.base.main()
    finally:
        try:
            sweep.summarise()
            write_diagnostics()
        finally:
            sweep.base.simulation_app.close()
