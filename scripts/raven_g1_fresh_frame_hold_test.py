
"""RAVEN: experimental stale-image / torso-pose feedback ablation.

REQUIRES existing files in the same scripts directory:
  raven_g1_actor_tracking.py
  raven_g1_static_framing_sweep.py
  raven_g1_reverse_vision_diagnostics.py

Do not use this as the permanent sensor-synchronization fix. It checks whether
holding the last bearing whenever the RGB-D actor measurement is identical
reduces the negative-pan oscillation. It never uses actor ground truth for control.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIAGNOSTICS = HERE / "raven_g1_reverse_vision_diagnostics.py"
for required in (
    "raven_g1_actor_tracking.py",
    "raven_g1_static_framing_sweep.py",
    "raven_g1_reverse_vision_diagnostics.py",
):
    if not (HERE / required).is_file():
        raise SystemExit(f"Missing prerequisite: {HERE / required}")

# Import creates Isaac Sim and installs the SAME diagnostics/sweep hooks.
# The underlying validated controller and calibrated tripod RGB remain unchanged.
spec = importlib.util.spec_from_file_location("raven_vision_diagnostics_ablation", DIAGNOSTICS)
if spec is None or spec.loader is None:
    raise SystemExit(f"Unable to load {DIAGNOSTICS}")
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)
sweep = diag.sweep

sweep.OUT = (
    Path.cwd() / "raven_tracking_results" / "fresh_frame_hold_test"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
sweep.OUT.mkdir(parents=True, exist_ok=False)
sweep.base.args.output_dir = str(sweep.OUT)

# Replace ONLY the original detector function called inside the existing
# static-sweep wrapper. This leaves all control, IK and physics code intact.
validated_detector = sweep.old_detect_actor
last_signature = None
last_bearing = None
rows = []


def identical_measurement(detection):
    """Conservative image-derived signature; NOT a camera frame identifier."""
    return (
        float(detection["u"]),
        float(detection["v"]),
        float(detection["depth_m"]),
        int(detection["pixels"]),
        tuple(detection["bbox"]),
    )


def hold_if_repeated(camera, tripod, robot, torso_body):
    global last_signature, last_bearing
    detection, image, status = validated_detector(camera, tripod, robot, torso_body)
    t = sweep.state["time_s"] - sweep.base.PAN_START_S
    stage, time_in_stage = sweep.phase_at(sweep.state["time_s"])
    held = False
    raw_bearing = None
    used_bearing = None
    if detection is None:
        # No valid image-derived target measurement: the original controller's
        # detection-loss behaviour must remain unchanged.
        last_signature = None
        last_bearing = None
    else:
        raw_bearing = float(detection["bearing_deg"])
        signature = identical_measurement(detection)
        if signature == last_signature and last_bearing is not None:
            held = True
            # A diagnostic estimate-hold, NOT a physical robot-pose freeze.
            detection = dict(detection)
            detection["bearing_deg"] = last_bearing
        else:
            last_bearing = raw_bearing
        last_signature = signature
        used_bearing = float(detection["bearing_deg"])

    if t >= 0:
        rows.append({
            "tracking_time_s": round(t, 5),
            "stage_index": stage + 1 if stage is not None else "",
            "actor_y_evaluation_only_m": sweep.POSITIONS_M[stage] if stage is not None else "",
            "time_at_position_s": round(time_in_stage, 4) if time_in_stage is not None else "",
            "duplicate_measurement_held": held,
            "raw_bearing_deg": raw_bearing if raw_bearing is not None else "",
            "used_bearing_deg": used_bearing if used_bearing is not None else "",
            "held_difference_deg": (
                used_bearing - raw_bearing if used_bearing is not None else ""
            ),
        })
    return detection, image, status


sweep.old_detect_actor = hold_if_repeated


def write_hold_report():
    output_csv = sweep.OUT / "fresh_frame_hold_telemetry.csv"
    fields = [
        "tracking_time_s", "stage_index", "actor_y_evaluation_only_m",
        "time_at_position_s", "duplicate_measurement_held",
        "raw_bearing_deg", "used_bearing_deg", "held_difference_deg",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_stage = []
    for index, pos in enumerate(sweep.POSITIONS_M, start=1):
        valid = [
            row for row in rows
            if row["stage_index"] == index
            and isinstance(row["time_at_position_s"], (int, float))
            and 2.0 <= row["time_at_position_s"] < 6.0
        ]
        count = sum(row["duplicate_measurement_held"] for row in valid)
        item = {
            "stage": index,
            "actor_y_evaluation_only_m": pos,
            "evaluated_samples": len(valid),
            "held_samples": count,
            "held_fraction": count / len(valid) if valid else None,
        }
        per_stage.append(item)
        print(
            f"[FRAME HOLD] stage={index} actor_y={pos:+.2f}m "
            f"held={count}/{len(valid)}"
        )

    summary = {
        "experiment": "RGB-D identical-measurement bearing-hold ablation",
        "purpose": "Test whether repeated image/updated torso-pose mismatch contributes to oscillation",
        "controller_source_unchanged": True,
        "calibrated_camera_unchanged": True,
        "grasp_and_ik_unchanged": True,
        "ground_truth_for_control": False,
        "caveats": [
            "An identical centroid/depth/pixel-count/bbox is not a true camera frame ID.",
            "This experimental hold introduces its own latency and may make performance worse.",
            "An improvement would support a feedback/timing contribution, not prove the sole root cause.",
        ],
        "stages": per_stage,
        "csv": str(output_csv),
    }
    (sweep.OUT / "fresh_frame_hold.summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("[FRAME HOLD] Results:", sweep.OUT)


if __name__ == "__main__":
    print("[RAVEN] Comparing against previous unmodified reverse vision diagnostic run")
    print("[RAVEN] Holding G1 bearing on identical RGB-D actor measurements only")
    print("[RAVEN] Controller/grasp/IK remain unchanged:", sweep.OUT)
    try:
        sweep.base.main()
    finally:
        try:
            sweep.summarise()
            diag.write_diagnostics()
            write_hold_report()
        finally:
            sweep.base.simulation_app.close()
