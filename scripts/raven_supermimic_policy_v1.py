"""RAVEN Policy V1.1: frame-aligned offline action planning through frozen World Model V2.

This is differentiable model-predictive action optimisation, NOT yet a trained
reinforcement-learning policy. It does not launch Isaac Sim or command a robot.
It uses ONLY the current tracking reference (sample-and-hold from the RGB-D
tracker CSV), current robot state, and current G1 command. Future commands
recorded in the dataset are NOT provided to the planner.

Execute from the RAVEN repository root, with the existing env_isaaclab active.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from raven_supermimic_world_model_v2 import DT, HORIZON, JOINTS, ResidualDynamics, read_run

DEG = 180.0 / math.pi
# Strict experimental trust region (rad) around current held joint targets.
# Waist has a tighter bound; commands remain unconstrained by actual limits
# because the recording does not contain hardware joint-limit metadata.
MAX_DELTA_RAD = (0.006, 0.018, 0.012, 0.012, 0.012, 0.010, 0.010, 0.010)


def current_reference(csv_path: Path, times: np.ndarray) -> np.ndarray:
    """Latest ALREADY MEASURED desired-pan sample; never future samples."""
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"time_s", "desired_pan_deg"}.issubset(rows[0]):
        raise ValueError(f"Tracking CSV missing time_s / desired_pan_deg: {csv_path}")
    timestamps = np.array([float(row["time_s"]) for row in rows], dtype=np.float64)
    references = np.array([float(row["desired_pan_deg"]) for row in rows], dtype=np.float64)
    if (not np.isfinite(timestamps).all() or not np.isfinite(references).all()
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("Tracking CSV must have finite, strictly increasing samples")
    chosen = np.searchsorted(timestamps, times, side="right") - 1
    if np.any(chosen < 0):
        raise ValueError("Cannot choose a camera reference before first RGB-D sample")
    return np.deg2rad(references[chosen]).astype(np.float32)


def planned_actions(raw: torch.Tensor, held: torch.Tensor, max_delta: torch.Tensor):
    """The zero vector reproduces the unchanged-current-command baseline."""
    return held[:, None, :] + max_delta[None, None, :] * torch.tanh(raw)


def objective(pred: torch.Tensor, action: torch.Tensor, held: torch.Tensor,
              desired_pan: torch.Tensor, start: torch.Tensor, max_delta: torch.Tensor):
    # Angle terms in degrees. This is an EXPERIMENTAL surrogate reward, not a
    # proven cinematographic quality metric or a simulation safety constraint.
    pan_error_deg = (pred[:, :, 16] - desired_pan[:, None]) * DEG
    tilt_change_deg = (pred[:, :, 17] - start[:, None, 17]) * DEG
    # Penalise only the commanded changes; model does not simulate hand contact.
    normalized_change = (action - held[:, None, :]) / max_delta[None, None, :]
    smooth = ((action[:, 1:] - action[:, :-1]) /
              max_delta[None, None, :]).square().mean(dim=(1, 2))
    track = pan_error_deg[:, -1].square() + 0.25 * pan_error_deg.square().mean(dim=1)
    return (track + 0.3 * tilt_change_deg.square().mean(dim=1)
            + 0.025 * normalized_change.square().mean(dim=(1, 2))
            + 0.04 * smooth)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True,
                   help="Frozen World Model V2 checkpoint")
    p.add_argument("--data", type=Path, required=True,
                   help="120 Hz capture not used to train V2")
    p.add_argument("--csv", type=Path, default=None,
                   help="Corresponding 10 Hz tracker CSV; defaults to same directory")
    p.add_argument("--out", type=Path, default=Path("raven_training_results/policy_v1"))
    p.add_argument("--cases", type=int, default=20)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--min-time-s", type=float, default=27.5)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if args.cases < 1 or args.iterations < 1:
        p.error("--cases and --iterations must be positive")
    for name, path in (("model", args.model), ("data", args.data)):
        if not path.is_file():
            p.error(f"Missing {name}: {path}")
    csv_file = args.csv or args.data.parent / "raven_actor_tracking.csv"
    if not csv_file.is_file():
        p.error(f"Tracking CSV not found: {csv_file}")

    device = torch.device("cuda" if (args.device == "cuda" or
                         (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA was requested but unavailable")
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    if ckpt.get("model_class") != "ResidualDynamics" or tuple(ckpt["joint_names"]) != JOINTS:
        p.error("Checkpoint is not the expected RAVEN World Model V2")
    if int(ckpt["horizon"]) != HORIZON:
        p.error("Checkpoint horizon mismatch")
    for trained_path in ckpt.get("metrics", {}).get("training_recordings", []):
        if Path(trained_path).resolve() == args.data.resolve():
            p.error("Use an UNSEEN recording; this capture was used for model training")

    model = ResidualDynamics(*ckpt["normalization"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)  # We optimise actions, NEVER retrain the model.

    state_np, command_np = read_run(args.data, args.min_time_s)
    if len(state_np) < HORIZON + 400:
        p.error("Capture is too short for the requested analysis")
    # Pick samples across the entire tracking phase, avoiding early grasp settle.
    first = round(2.0 / DT)
    last = len(state_np) - HORIZON - 1
    starts_idx = np.unique(np.linspace(first, last, args.cases).astype(int))
    start_times = starts_idx * DT
    # The RGB-D tracker records desired/actual pan relative to the physical
    # tripod angle at the START of tracking. Physics telemetry stores the
    # absolute pan joint position. Put both into absolute-joint coordinates
    # BEFORE computing the objective (otherwise a ~2.6 deg error is injected).
    pan_zero_rad = float(state_np[0, 16])
    relative_reference_np = current_reference(csv_file, start_times)
    reference_np = relative_reference_np + np.float32(pan_zero_rad)

    # Verify the measured 120 Hz pan corresponds to the tracking CSV after
    # offset removal; fail rather than silently compare unlike coordinates.
    with csv_file.open(newline="", encoding="utf-8-sig") as stream:
        tracking_rows = list(csv.DictReader(stream))
    alignment_residuals = []
    for row in tracking_rows:
        j = round(float(row["time_s"]) / DT)
        if 0 <= j < len(state_np):
            measured_relative_deg = (float(state_np[j, 16]) - pan_zero_rad) * DEG
            alignment_residuals.append(
                measured_relative_deg - float(row["actual_pan_deg"])
            )
    if len(alignment_residuals) < 20:
        raise RuntimeError("Insufficient samples to verify pan reference alignment")
    alignment_mae_deg = float(np.mean(np.abs(alignment_residuals)))
    alignment_p95_deg = float(np.percentile(np.abs(alignment_residuals), 95))
    if alignment_p95_deg > 0.25:
        raise RuntimeError(
            f"Physics/CSV pan mismatch: p95={alignment_p95_deg:.3f} deg; "
            "verify these files come from the same run and time basis"
        )
    print(f"[POLICY V1.1] Pan zero = {pan_zero_rad * DEG:+.4f} deg absolute; "
          f"physics/CSV alignment MAE = {alignment_mae_deg:.4f} deg")
    start = torch.from_numpy(state_np[starts_idx]).to(device)
    held = torch.from_numpy(command_np[starts_idx]).to(device)
    refs = torch.from_numpy(reference_np).to(device)
    max_delta = torch.tensor(MAX_DELTA_RAD, dtype=torch.float32, device=device)
    reference_actions = held[:, None, :].expand(-1, HORIZON, -1)

    with torch.no_grad():
        baseline_pred = model.rollout(start, reference_actions)
        baseline_loss = objective(baseline_pred, reference_actions, held,
                                  refs, start, max_delta)

    # Per-window candidate plan. This is NOT a deployable feedback policy yet.
    raw = torch.nn.Parameter(torch.zeros((len(starts_idx), HORIZON, len(JOINTS)), device=device))
    optimizer = torch.optim.Adam([raw], lr=0.08)
    best_raw = raw.detach().clone()
    best_loss = baseline_loss.clone()
    for iteration in range(args.iterations):
        actions = planned_actions(raw, held, max_delta)
        predicted = model.rollout(start, actions)
        losses = objective(predicted, actions, held, refs, start, max_delta)
        total = losses.mean()
        if not torch.isfinite(total):
            raise RuntimeError("Non-finite policy objective; no results saved")
        with torch.no_grad():
            # Save the parameters that produced THESE losses, before stepping.
            improved = losses.detach() < best_loss
            best_loss = torch.minimum(losses.detach(), best_loss)
            best_raw[improved] = raw.detach()[improved]
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        optimizer.step()
        if iteration == 0 or (iteration + 1) % 20 == 0 or iteration + 1 == args.iterations:
            print(f"[POLICY V1] iteration {iteration+1:3d}/{args.iterations}: "
                  f"predicted objective={float(total.detach()):.5f}")

    with torch.no_grad():
        candidate = planned_actions(best_raw, held, max_delta)
        planned_pred = model.rollout(start, candidate)
        before = (baseline_pred[:, -1, 16] - refs).abs() * DEG
        after = (planned_pred[:, -1, 16] - refs).abs() * DEG
        shifts = (candidate - reference_actions).abs().amax(dim=(1, 2)) * DEG
        results = []
        for n, idx in enumerate(starts_idx):
            results.append({
                "tracking_time_s": round(float(start_times[n]), 3),
                "reference_pan_deg": round(float((refs[n] - pan_zero_rad) * DEG), 6),
                "current_pan_deg": round(float((start[n, 16] - pan_zero_rad) * DEG), 6),
                "held_command_predicted_final_pan_deg": round(float((baseline_pred[n, -1, 16] - pan_zero_rad) * DEG), 6),
                "planned_predicted_final_pan_deg": round(float((planned_pred[n, -1, 16] - pan_zero_rad) * DEG), 6),
                "baseline_predicted_error_deg": round(float(before[n]), 6),
                "planned_predicted_error_deg": round(float(after[n]), 6),
                "max_joint_target_change_deg": round(float(shifts[n]), 6),
                "planned_first_joint_targets_rad": [round(float(x), 7) for x in candidate[n, 0]],
            })
        avg_before = float(before.mean())
        avg_after = float(after.mean())

    summary = {
        "experiment": "RAVEN Policy V1.1: coordinate-aligned offline differentiable action planning",
        "pan_zero_absolute_deg": pan_zero_rad * DEG,
        "pan_csv_alignment_mae_deg": alignment_mae_deg,
        "pan_csv_alignment_p95_deg": alignment_p95_deg,
        "frozen_world_model_v2": str(args.model.resolve()),
        "test_recording": str(args.data.resolve()),
        "tracking_csv": str(csv_file.resolve()),
        "physics_hz": 120, "horizon_steps": HORIZON,
        "horizon_ms": round(HORIZON * DT * 1000, 3),
        "cases": len(results), "iterations": args.iterations,
        "joint_names": JOINTS, "max_command_delta_rad": MAX_DELTA_RAD,
        "mean_held_command_model_predicted_terminal_pan_error_deg": avg_before,
        "mean_optimised_model_predicted_terminal_pan_error_deg": avg_after,
        "predicted_error_change_percent": (100.0 * (avg_before - avg_after) / avg_before
                                           if avg_before > 0 else None),
        "cases_detail": results,
        "caveats": [
            "Predicted counterfactual improvement only; untested in Isaac Sim or on a robot.",
            "RGB-D pan reference corrected from relative to absolute joint coordinates; outputs shown relative to initial pan.",
            "This is MPC-style action optimisation, not a trained RL feedback policy.",
            "No grasp/contact constraints, joint-limit checks or physical collision guarantees.",
            "Uses current controller desired_pan_deg as a reference proxy, not direct actor pixel observations.",
            "Current joint commands are held for the eight-step baseline; no future commands leak in.",
            "Model predictions can be exploited outside the training distribution."
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "policy_v1_frame_corrected_metrics.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[POLICY V1.1] {len(results)} cases on {device}; frozen V2; horizon 66.7 ms")
    print(f"[POLICY V1] Mean predicted final pan error (held command): {avg_before:.5f} deg")
    print(f"[POLICY V1] Mean predicted final pan error (optimised):    {avg_after:.5f} deg")
    print("[POLICY V1] NO simulator rollout or real robot action was executed.")
    print(f"[POLICY V1] Saved: {destination}")


if __name__ == "__main__":
    main()
