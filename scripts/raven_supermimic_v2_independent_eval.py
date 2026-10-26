
"""RAVEN: independent, frozen-checkpoint evaluation of World Model V2 and V1.

Offline PyTorch evaluation; does not launch Isaac Sim or retrain anything.
Evaluation uses the entire post-grasp tracking phase of a THIRD, unseen capture.
All methods use identical start states and 8-step (66.7 ms) targets.

Run from repository root (see ChatGPT instructions for full command).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from raven_supermimic_world_model_v2 import (
    ResidualDynamics, read_run, make_windows, HORIZON, DT, JOINTS
)
from raven_supermimic_120hz_world_model import Dynamics, predict as predict_v1

DEG = 180.0 / math.pi


def same_path(a, b):
    return Path(a).resolve() == Path(b).resolve()


def error_stats(estimate, truth):
    error = np.abs(estimate - truth) * DEG
    return {"mae_deg": round(float(error.mean()), 6),
            "p95_deg": round(float(np.percentile(error, 95)), 6),
            "max_deg": round(float(error.max()), 6)}


def check_not_trained(checkpoint, new_capture, version):
    info = checkpoint.get("metrics", {})
    paths = (info.get("training_recordings", []) if version == "V2"
             else [info.get("source_trajectory")])
    for path in paths:
        if path and same_path(path, new_capture):
            raise ValueError(f"{version} model was trained on this recording; "
                             "it is NOT an independent test.")


def run_batched(model, starts, actions, device, v1_norms=None):
    prediction = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(starts), 256):
            x = starts[i:i + 256].to(device)
            u = actions[i:i + 256].to(device)
            if v1_norms is None:
                y = model.rollout(x, u)
            else:
                y = predict_v1(model, x, u, *v1_norms)
            prediction.append(y.cpu().numpy())
    return np.concatenate(prediction, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Third unseen 120 Hz capture (JSONL)")
    parser.add_argument("--v2", type=Path, required=True,
                        help="Frozen V2 world_model_v2.pt")
    parser.add_argument("--v1", type=Path, required=True,
                        help="Frozen V1 world_model.pt")
    parser.add_argument("--min-time-s", type=float, default=27.5)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    for path in (args.data, args.v2, args.v1):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.stride < 1:
        parser.error("--stride must be positive")

    data = args.data.resolve()
    meta_path = data.parent / "raven_g1_state_action.summary.json"
    track_path = data.parent / "raven_actor_tracking.summary.json"
    if not meta_path.is_file() or not track_path.is_file():
        parser.error("Missing capture and/or tracking summary JSON")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    track = json.loads(track_path.read_text(encoding="utf-8"))
    if meta.get("status") != "RECORDED" or track.get("status") != "COMPLETED":
        parser.error("The third capture did not finish successfully")
    if abs(float(meta.get("physics_hz", 0)) - 120.0) > 0.01:
        parser.error("Expected 120 Hz recording")

    v2_ckpt = torch.load(args.v2, map_location="cpu", weights_only=True)
    v1_ckpt = torch.load(args.v1, map_location="cpu", weights_only=True)
    check_not_trained(v2_ckpt, data, "V2")
    check_not_trained(v1_ckpt, data, "V1")
    if tuple(v2_ckpt["joint_names"]) != JOINTS:
        parser.error("V2 joint names do not match evaluator")
    if tuple(v1_ckpt["joint_names"]) != JOINTS:
        parser.error("V1 joint names do not match evaluator")
    if int(v2_ckpt["horizon"]) != HORIZON or int(v1_ckpt["horizon"]) != HORIZON:
        parser.error("Checkpoint rollout horizon mismatch")
    joint_indices = [meta["g1_joint_order"].index(j) for j in JOINTS]
    if list(v1_ckpt["joint_indices"]) != joint_indices:
        parser.error("Joint ordering differs from V1 checkpoint")

    states, targets = read_run(data, args.min_time_s)
    starts, actions, truth = make_windows(states, targets, 0, len(states), args.stride)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")

    v2 = ResidualDynamics(*v2_ckpt["normalization"]).to(device)
    v2.load_state_dict(v2_ckpt["state_dict"], strict=True)
    v1 = Dynamics(20, 8).to(device)
    v1.load_state_dict(v1_ckpt["state_dict"], strict=True)
    v1_norms = tuple(v1_ckpt[k].to(device) for k in (
        "state_mean", "state_std", "action_mean", "action_std"))

    prediction_v2 = run_batched(v2, starts, actions, device)
    prediction_v1 = run_batched(v1, starts, actions, device, v1_norms)
    first = starts.numpy()
    actual = truth.numpy()

    comparisons = {}
    for axis, (pos_idx, vel_idx) in {
        "pan": (16, 18), "tilt": (17, 19)
    }.items():
        y = actual[:, -1, pos_idx]
        fixed = first[:, pos_idx]
        extrap = fixed + (HORIZON * DT) * first[:, vel_idx]
        comparisons[axis] = {
            "V2": error_stats(prediction_v2[:, -1, pos_idx], y),
            "V1": error_stats(prediction_v1[:, -1, pos_idx], y),
            "constant_velocity": error_stats(extrap, y),
            "stationary": error_stats(fixed, y),
        }

    def improvement_against(base, model):
        return round(100.0 * (base - model) / base, 2) if base else None

    pan = comparisons["pan"]
    result = {
        "experiment": "RAVEN third-trajectory independent test: frozen V1 vs V2",
        "third_recording": str(data),
        "V2_checkpoint": str(args.v2.resolve()),
        "V1_checkpoint": str(args.v1.resolve()),
        "retrained": False,
        "model_controls_robot": False,
        "physics_hz": 120,
        "states": len(states),
        "evaluation_windows": len(starts),
        "window_stride": args.stride,
        "horizon_steps": HORIZON,
        "horizon_ms": round(HORIZON * DT * 1000, 2),
        "tracking_mae_deg": track.get("mean_absolute_tracking_error_deg"),
        "detection_rate": track.get("detection_rate"),
        "results": comparisons,
        "V2_improvement_over_constant_velocity_percent": improvement_against(
            pan["constant_velocity"]["mae_deg"], pan["V2"]["mae_deg"]),
        "limitations": [
            "One independent test run; repeated starts yield correlated test windows.",
            "Rollouts condition on recorded future actions; no policy has been trained.",
            "No controlled camera/actuator latency has been introduced yet."
        ]
    }

    output = data.parent / "raven_world_model_v2_independent_eval.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[RAVEN] {len(states)} independent states, {len(starts)} windows, "
          f"{HORIZON * DT * 1000:.1f} ms horizon, device={device}")
    print("[RAVEN] Pan prediction MAE (degrees; lower is better):")
    for name in ("V2", "V1", "constant_velocity", "stationary"):
        print(f"  {name:20s} {pan[name]['mae_deg']:.4f} "
              f"(p95 {pan[name]['p95_deg']:.4f})")
    print("[RAVEN] V2 improvement over constant velocity: "
          f"{result['V2_improvement_over_constant_velocity_percent']:+.2f}%")
    print(f"[RAVEN] Saved: {output}")


if __name__ == "__main__":
    main()
