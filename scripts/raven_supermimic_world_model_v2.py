
"""RAVEN World Model V2: constant-velocity + learned residual dynamics.

Offline, SuperMimic-inspired research prototype; NOT official SuperMimic or RL.
Train on TWO independent 120 Hz RAVEN captures (no Isaac Sim needed).
Validation uses the final 20% of EACH run; no time-overlapping windows.
Reserve a third, unseen run for later independent evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

JOINTS = (
    "waist_yaw_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
)
DT = 1.0 / 120.0
HORIZON = 8
DEG = 180.0 / math.pi


def read_run(path: Path, min_time: float):
    sidecar = path.parent / "raven_g1_state_action.summary.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"Capture metadata not found: {sidecar}")
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    if abs(float(info.get("physics_hz", 0)) - 120.0) > 0.01:
        raise ValueError(f"Expected 120 Hz: {path}")
    indices = [info["g1_joint_order"].index(j) for j in JOINTS]
    if list(info["tripod_joint_order"]) != ["pan_joint", "tilt_joint"]:
        raise ValueError("Unexpected tripod joint ordering")
    states, actions, steps = [], [], []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if float(r["sim_time_s"]) < min_time:
                continue
            pos = [r["robot_joint_pos_rad"][i] for i in indices]
            vel = [r["robot_joint_vel_rad_s"][i] for i in indices]
            cmd = [r["robot_joint_position_targets_rad"][i] for i in indices]
            states.append(pos + vel + r["tripod_joint_pos_rad"]
                          + r["tripod_joint_vel_rad_s"])
            actions.append(cmd)
            steps.append(int(r["physics_step"]))
    if len(states) < 300:
        raise ValueError(f"Too few tracking-phase samples: {path}")
    if np.any(np.diff(steps) != 1):
        raise ValueError(f"Discontinuous physics steps: {path}")
    s, a = np.asarray(states, np.float32), np.asarray(actions, np.float32)
    if s.shape[1] != 20 or a.shape[1] != 8:
        raise ValueError("Wrong state/action dimensions")
    if not (np.isfinite(s).all() and np.isfinite(a).all()):
        raise ValueError(f"Non-finite telemetry: {path}")
    return s, a


def baseline(s: torch.Tensor):
    """Constant joint and tripod velocities, over one physics step."""
    b = s.clone()
    b[:, :8] = s[:, :8] + DT * s[:, 8:16]
    b[:, 16:18] = s[:, 16:18] + DT * s[:, 18:20]
    return b


def make_windows(s, a, first, last, stride):
    # Command at t+1 was applied before state t+1 was captured.
    indices = np.arange(first, last - HORIZON, stride)
    if len(indices) == 0:
        raise ValueError("Partition too short for rollout")
    x = np.stack([s[i] for i in indices]).astype(np.float32)
    u = np.stack([a[i + 1:i + 1 + HORIZON] for i in indices]).astype(np.float32)
    y = np.stack([s[i + 1:i + 1 + HORIZON] for i in indices]).astype(np.float32)
    return tuple(torch.from_numpy(v) for v in (x, u, y))


class ResidualDynamics(nn.Module):
    def __init__(self, state_mean, state_scale, action_mean, action_scale, residual_scale):
        super().__init__()
        for name, value in (
            ("state_mean", state_mean), ("state_scale", state_scale),
            ("action_mean", action_mean), ("action_scale", action_scale),
            ("residual_scale", residual_scale),
        ):
            self.register_buffer(name, value.clone().float())
        self.net = nn.Sequential(
            nn.Linear(28, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 20),
        )
        # Start with EXACT constant-velocity predictions.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def step(self, state, target):
        position_error = target - state[:, :8]
        features = torch.cat((
            (state - self.state_mean) / self.state_scale,
            (position_error - self.action_mean) / self.action_scale,
        ), dim=1)
        correction = self.net(features) * self.residual_scale
        return baseline(state) + correction

    def rollout(self, start, targets):
        s = start
        sequence = []
        for t in range(targets.shape[1]):
            s = self.step(s, targets[:, t, :])
            sequence.append(s)
        return torch.stack(sequence, dim=1)


def collect_norms(runs, cutoffs):
    parts = [s[:stop] for (s, _), stop in zip(runs, cutoffs)]
    actions = [a[:stop] - s[:stop, :8] for (s, a), stop in zip(runs, cutoffs)]
    state = np.concatenate(parts)
    act_err = np.concatenate(actions)
    state_floor = np.array([0.02]*8 + [0.04]*8 + [0.02, 0.005, 0.04, 0.02])
    residual_floor = np.array([0.00005]*8 + [0.005]*8
                              + [0.0001, 0.0001, 0.003, 0.003])
    delta = np.concatenate([
        s[1:stop] - np.column_stack((
            s[:stop-1, :8] + DT * s[:stop-1, 8:16],
            s[:stop-1, 8:16],
            s[:stop-1, 16:18] + DT * s[:stop-1, 18:20],
            s[:stop-1, 18:20],
        ))
        for (s, _), stop in zip(runs, cutoffs)
    ])
    return tuple(torch.from_numpy(v.astype(np.float32)) for v in (
        state.mean(0), np.maximum(state.std(0), state_floor),
        act_err.mean(0), np.maximum(act_err.std(0), 0.002),
        np.maximum(delta.std(0), residual_floor),
    ))


def metrics(pred: torch.Tensor, initial: torch.Tensor, truth: torch.Tensor):
    model = pred[:, -1, 16:18].numpy()
    actual = truth[:, -1, 16:18].numpy()
    s = initial.numpy()
    persist = s[:, 16:18]
    cv = s[:, 16:18] + s[:, 18:20] * (HORIZON * DT)
    out = {}
    for idx, axis in enumerate(("pan", "tilt")):
        errors = {}
        for label, guess in (("model", model[:, idx]),
                             ("constant_velocity", cv[:, idx]),
                             ("stationary", persist[:, idx])):
            e = np.abs(guess - actual[:, idx]) * DEG
            errors[label] = {"mae_deg": float(e.mean()),
                             "p95_deg": float(np.percentile(e, 95))}
        out[axis] = errors
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, nargs=2, required=True,
                    help="Exactly two separate 120 Hz tracking captures")
    ap.add_argument("--out", type=Path, default=Path("raven_training_results/world_model_v2"))
    ap.add_argument("--min-time-s", type=float, default=27.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.epochs < 1 or args.stride < 1 or args.batch_size < 1:
        ap.error("epochs, stride and batch size must be positive")
    paths = [p.resolve() for p in args.data]
    if paths[0] == paths[1]:
        ap.error("Provide two DIFFERENT recordings")
    if not all(p.is_file() for p in paths):
        ap.error("At least one recording path does not exist")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = [read_run(path, args.min_time_s) for path in paths]
    cutoffs = [int(len(s) * 0.80) for s, _ in runs]
    norms = collect_norms(runs, cutoffs)
    model = ResidualDynamics(*norms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0001)
    # Emphasise pan, while still learning all G1 + tripod states.
    weights = torch.tensor([0.25]*8 + [0.20]*8 + [12., 6., 2., 1.], device=device)
    train_parts = [make_windows(s, a, 0, end, args.stride)
                   for (s, a), end in zip(runs, cutoffs)]
    val_parts = [make_windows(s, a, end, len(s), args.stride)
                 for (s, a), end in zip(runs, cutoffs)]
    train = [torch.cat([p[i] for p in train_parts]).to(device) for i in range(3)]
    print(f"[RAVEN V2] {len(runs)} trajectories, {sum(len(s) for s, _ in runs)} states")
    print(f"[RAVEN V2] {len(train[0])} training windows; "
          f"{sum(len(p[0]) for p in val_parts)} validation windows; GPU: {device}")
    best_val = float("inf")
    best = None
    history = []

    def loss_fn(pred, actual):
        err = (pred - actual) / model.state_scale
        return (err.square() * weights).mean()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train[0]), device=device)
        train_loss = 0.0
        for chunk in order.split(args.batch_size):
            p = model.rollout(train[0][chunk], train[1][chunk])
            loss = loss_fn(p, train[2][chunk])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(chunk)
        model.eval()
        val_losses = []
        with torch.no_grad():
            for part in val_parts:
                x, a, y = [t.to(device) for t in part]
                val_losses.append((loss_fn(model.rollout(x, a), y).item(), len(x)))
        val_loss = sum(v * n for v, n in val_losses) / sum(n for _, n in val_losses)
        history.append({"epoch": epoch, "train_loss": train_loss/len(train[0]),
                        "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[Epoch {epoch:3d}/{args.epochs}] train={history[-1]['train_loss']:.6f} "
                  f"validation={val_loss:.6f}")

    model.load_state_dict(best)
    model.eval()
    reports = []
    with torch.no_grad():
        for path, part in zip(paths, val_parts):
            x, u, y = part
            preds = []
            for start in range(0, len(x), 256):
                preds.append(model.rollout(x[start:start+256].to(device),
                                           u[start:start+256].to(device)).cpu())
            result = metrics(torch.cat(preds), x, y)
            reports.append({"recording": str(path), "windows": len(x),
                            "validation": result})
            pan = result["pan"]
            print(f"[VALIDATION] {path.parent.name}: "
                  f"model={pan['model']['mae_deg']:.4f}°, "
                  f"CV={pan['constant_velocity']['mae_deg']:.4f}°, "
                  f"still={pan['stationary']['mae_deg']:.4f}°")
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"experiment": "RAVEN World Model V2: constant-velocity residual",
              "is_official_supermimic": False, "rl_policy_trained": False,
              "evaluated_on_independent_third_run": False,
              "device": str(device), "physics_hz": 120, "horizon_steps": HORIZON,
              "prediction_horizon_ms": 1000*HORIZON*DT,
              "joint_names": JOINTS, "training_recordings": [str(p) for p in paths],
              "tracking_phase_starts_s": args.min_time_s,
              "split": "first 80% train, last 20% validation for EACH trajectory; no overlapping windows",
              "best_validation_loss": best_val, "epochs": args.epochs,
              "training_history": history, "per_trajectory": reports,
              "important_limitation": "Validation segments are from the same two trajectories used for training; hold out an entirely third trajectory for independent evaluation."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save({"state_dict": best, "joint_names": JOINTS,
                "model_class": "ResidualDynamics", "horizon": HORIZON,
                "normalization": [n.cpu() for n in norms], "metrics": report},
               output / "world_model_v2.pt")
    print(f"[RAVEN V2] Saved metrics: {output / 'metrics.json'}")
    print(f"[RAVEN V2] Saved checkpoint: {output / 'world_model_v2.pt'}")
    print("[RAVEN V2] These are within-trajectory validation results; a third run remains essential.")


if __name__ == "__main__":
    main()
