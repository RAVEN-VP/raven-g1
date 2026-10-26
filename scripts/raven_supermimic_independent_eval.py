
"""Evaluate a FIXED RAVEN world-model checkpoint on a NEW 120 Hz run.

No training, Isaac Sim, or changes to the robot controller. Reports error at
8 steps (66.7 ms) against angle persistence and constant-velocity baselines.

Run from repository root:
    python scripts/raven_supermimic_independent_eval.py --model PATH_TO_WORLD_MODEL_PT --data PATH_TO_NEW_JSONL
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from raven_supermimic_120hz_world_model import (
    Dynamics, HORIZON, JOINT_NAMES, build_arrays, load_records, make_windows,
    predict,
)

RAD2DEG = 180.0 / math.pi


def stats_deg(prediction: np.ndarray, truth: np.ndarray) -> dict:
    error = np.abs(prediction - truth) * RAD2DEG
    return {
        "mae_deg": round(float(error.mean()), 6),
        "p95_abs_error_deg": round(float(np.percentile(error, 95)), 6),
        "max_abs_error_deg": round(float(error.max()), 6),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Previously trained world_model.pt")
    parser.add_argument("--data", type=Path, required=True, help="New, independent 120 Hz JSONL capture")
    parser.add_argument("--min-time-s", type=float, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    model_path, data_path = args.model.resolve(), args.data.resolve()
    for path in (model_path, data_path):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    trained_on = checkpoint.get("metrics", {}).get("source_trajectory")
    if trained_on and Path(trained_on).resolve() == data_path:
        parser.error("This is the original training recording, not an independent test run.")
    if tuple(checkpoint["joint_names"]) != tuple(JOINT_NAMES):
        parser.error("Checkpoint joint selection does not match the installed trainer script.")
    if int(checkpoint["horizon"]) != HORIZON:
        parser.error("Checkpoint rollout horizon does not match the trainer script.")

    info_path = data_path.parent / "raven_g1_state_action.summary.json"
    track_path = data_path.parent / "raven_actor_tracking.summary.json"
    if not info_path.is_file():
        parser.error(f"Missing capture metadata: {info_path}")
    metadata = json.loads(info_path.read_text(encoding="utf-8"))
    if abs(float(metadata["physics_hz"]) - 120.0) > 0.01:
        parser.error("Expected 120 Hz capture.")
    if metadata.get("status") != "RECORDED":
        parser.error(f"Capture did not finish cleanly: {metadata.get('status')}")
    tracking = json.loads(track_path.read_text(encoding="utf-8")) if track_path.is_file() else {}
    if tracking.get("status") not in (None, "COMPLETED"):
        parser.error(f"Original tracking run failed: {tracking.get('status')}")

    min_time = args.min_time_s
    if min_time is None:
        min_time = float(checkpoint.get("metrics", {}).get("training_phase_starts_at_s", 27.5))
    records = load_records(data_path, min_time)
    states, commands, selected_indices = build_arrays(records, metadata["g1_joint_order"])
    if tuple(selected_indices) != tuple(checkpoint["joint_indices"]):
        parser.error("G1 joint order has changed since model training.")
    initial, actions, actual = make_windows(states, commands, 0, len(states), stride=4)

    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        parser.error("CUDA requested but not available.")
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Dynamics(states.shape[1], commands.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    norms = tuple(checkpoint[name].to(device) for name in (
        "state_mean", "state_std", "action_mean", "action_std"
    ))

    print(f"[RAVEN] Independent test: {len(records)} states, {len(initial)} 8-step windows")
    print(f"[RAVEN] Loaded frozen checkpoint: {model_path.name} on {device}")
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(initial), 256):
            finish = start + 256
            p = predict(model, initial[start:finish].to(device),
                        actions[start:finish].to(device), *norms)
            outputs.append(p.cpu().numpy())
    predicted = np.concatenate(outputs, axis=0)
    observed = actual.numpy()
    starting = initial.numpy()
    # State layout: 8 q, 8 dq, tripod pan, tilt, pan velocity, tilt velocity.
    angles = {"pan": (16, 18), "tilt": (17, 19)}
    horizons_s = np.arange(1, HORIZON + 1, dtype=np.float32)[None, :] / 120.0
    comparisons = {}
    for name, (position, velocity) in angles.items():
        truth = observed[:, :, position]
        ml = predicted[:, :, position]
        still = np.broadcast_to(starting[:, None, position], truth.shape)
        linear = starting[:, None, position] + starting[:, None, velocity] * horizons_s
        comparisons[name] = {
            "model_step8": stats_deg(ml[:, -1], truth[:, -1]),
            "persistence_step8": stats_deg(still[:, -1], truth[:, -1]),
            "constant_velocity_step8": stats_deg(linear[:, -1], truth[:, -1]),
            "model_steps1_to8": stats_deg(ml, truth),
        }

    model_pan = comparisons["pan"]["model_step8"]["mae_deg"]
    still_pan = comparisons["pan"]["persistence_step8"]["mae_deg"]
    linear_pan = comparisons["pan"]["constant_velocity_step8"]["mae_deg"]
    metrics = {
        "experiment": "RAVEN independent trajectory generalisation test",
        "training_disabled": True,
        "source_model": str(model_path),
        "new_test_data": str(data_path),
        "model_original_trajectory": trained_on,
        "capture_status": metadata.get("status"),
        "tracking_status": tracking.get("status"),
        "camera_detection_rate": tracking.get("detection_rate"),
        "tracking_controller_mae_deg": tracking.get("mean_absolute_tracking_error_deg"),
        "tracking_phase_start_s": min_time,
        "state_count": len(records),
        "evaluation_windows": len(initial),
        "horizon_steps": HORIZON,
        "horizon_ms": round(HORIZON / 120.0 * 1000, 2),
        "device": str(device),
        "pan_model_vs_persistence_improvement_percent": round(100 * (still_pan - model_pan) / still_pan, 2) if still_pan else None,
        "pan_model_vs_constant_velocity_improvement_percent": round(100 * (linear_pan - model_pan) / linear_pan, 2) if linear_pan else None,
        "results": comparisons,
        "limitations": [
            "Evaluates new recorded controller trajectories, not learned policy control.",
            "Recorded actions are known for each eight-step prediction; no action planning is tested.",
            "One different sinusoidal trajectory alone does not establish broad generalisation.",
        ],
    }
    destination = data_path.parent / "raven_independent_world_model_eval.json"
    destination.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[RAVEN] Test pan 8-step MAE: model {model_pan:.4f} deg; "
          f"still {still_pan:.4f} deg; constant velocity {linear_pan:.4f} deg")
    print(f"[RAVEN] Test tilt 8-step MAE: {comparisons['tilt']['model_step8']['mae_deg']:.4f} deg")
    print(f"[RAVEN] Saved: {destination}")
    if model_pan >= min(still_pan, linear_pan):
        print("[RAVEN] Note: model did NOT beat both simple baselines on this new run.")
    else:
        print("[RAVEN] Model beat both simple baselines on this new run.")


if __name__ == "__main__":
    main()
