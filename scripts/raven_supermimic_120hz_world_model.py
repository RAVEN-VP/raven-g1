
"""RAVEN 120 Hz offline dynamics bootstrap, inspired by SuperMimic (NOT the official implementation).

Learns f(s_t, a_t) -> s_(t+1) from the validated 120 Hz G1-tripod
telemetry, using eight-step differentiable rollouts. Does not import Isaac Sim
or alter the existing G1 controller. Targets are 8 selected G1 joint position
commands; states include the 8 joint positions, velocities, tripod pan/tilt
positions and velocities.

Usage:
    python scripts/raven_supermimic_120hz_world_model.py \
        --data raven_training_results/supermimic_capture/<run>/raven_g1_state_action_120hz.jsonl

Evaluation: chronological train/validation/test split of a SINGLE trajectory,
no overlapping rollout windows across partitions. Held-out results should NOT
be taken as evidence of generalization to new motions or grasp configurations.
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

JOINT_NAMES = (
    "waist_yaw_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
)
HORIZON = 8


def load_records(path: Path, min_time_s: float):
    records = []
    for line in path.open("r", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["sim_time_s"] >= min_time_s:
                records.append(r)
    if len(records) < 3 * (HORIZON + 50):
        raise ValueError("Not enough tracking-phase records for split and rollout")
    for a, b in zip(records, records[1:]):
        if b["physics_step"] != a["physics_step"] + 1:
            raise ValueError("Missing physics step in capture; need contiguous samples")
        if abs((b["sim_time_s"] - a["sim_time_s"]) - (1 / 120)) > 0.0001:
            raise ValueError("Capture is not continuous 120 Hz")
    return records


def build_arrays(records, joint_names):
    indices = [joint_names.index(name) for name in JOINT_NAMES]
    states = []
    commands = []
    for r in records:
        q = [r["robot_joint_pos_rad"][k] for k in indices]
        dq = [r["robot_joint_vel_rad_s"][k] for k in indices]
        states.append(q + dq + r["tripod_joint_pos_rad"] + r["tripod_joint_vel_rad_s"])
        commands.append([r["robot_joint_position_targets_rad"][k] for k in indices])
    states = np.asarray(states, dtype=np.float32)
    commands = np.asarray(commands, dtype=np.float32)
    if not (np.isfinite(states).all() and np.isfinite(commands).all()):
        raise ValueError("Non-finite state or command in recording")
    return states, commands, indices


def partitions(n):
    # Windows never cross a partition boundary. Any early grasp phase was
    # excluded at the record-loading step.
    a, b = int(n * 0.70), int(n * 0.85)
    return {"train": (0, a), "validation": (a, b), "test": (b, n)}


def make_windows(states, commands, start, stop, stride=4):
    xs, us, ys = [], [], []
    for i in range(start, stop - HORIZON, stride):
        xs.append(states[i])
        # Each logged command was applied BEFORE its corresponding post-step
        # state. So use command at i+1 to predict state at i+1 from state at i.
        us.append(commands[i+1:i+HORIZON+1])
        ys.append(states[i+1:i+HORIZON+1])
    if not xs:
        raise ValueError("Partition too short for rollout")
    return tuple(torch.tensor(np.asarray(x), dtype=torch.float32)
                 for x in (xs, us, ys))


class Dynamics(nn.Module):
    """Residual normalized next-state predictor, differentiable through steps."""
    def __init__(self, ns, na):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ns + na, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, ns))
        # Begin close to constant-state prediction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, s, a):
        return s + self.net(torch.cat((s, a), dim=-1))


def predict(model, start, actions, mean_s, std_s, mean_a, std_a):
    normalized = (start - mean_s) / std_s
    predicted = []
    for j in range(actions.shape[1]):
        normalized_action = (actions[:, j] - mean_a) / std_a
        normalized = model(normalized, normalized_action)
        predicted.append(normalized * std_s + mean_s)
    return torch.stack(predicted, dim=1)


def score(model, window, norms, device):
    start, actions, truth = (tensor.to(device) for tensor in window)
    with torch.no_grad():
        pred = predict(model, start, actions, *norms)
        persistence = start.unsqueeze(1).expand_as(truth)
        errors = (pred - truth).abs()
        unchanged = (persistence - truth).abs()
    # Final pan is index 16; tilt is index 17.
    return {
        "rollout_horizon_steps": HORIZON,
        "rollout_duration_ms": round(HORIZON / 120 * 1000, 2),
        "windows": len(start),
        "pan_mean_absolute_error_deg": float(errors[:, :, 16].mean().item() * 180 / math.pi),
        "pan_persistence_baseline_mae_deg": float(unchanged[:, :, 16].mean().item() * 180 / math.pi),
        "pan_step8_mae_deg": float(errors[:, -1, 16].mean().item() * 180 / math.pi),
        "pan_persistence_step8_mae_deg": float(unchanged[:, -1, 16].mean().item() * 180 / math.pi),
        "tilt_step8_mae_deg": float(errors[:, -1, 17].mean().item() * 180 / math.pi),
        "joint_position_step8_mae_deg": float(errors[:, -1, :8].mean().item() * 180 / math.pi),
        "joint_velocity_step8_mae_rad_s": float(errors[:, -1, 8:16].mean().item()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--min-time-s", type=float, default=27.5,
                    help="Exclude staged approach/grasp; baseline tracking starts at ~27.5 s")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA but PyTorch reports no CUDA device")
    device = torch.device("cuda" if (args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()))
                          else "cpu")
    data_file = args.data.resolve()
    out = args.out or (data_file.parent / "world_model_bootstrap")
    out.mkdir(parents=True, exist_ok=True)

    summary_file = data_file.parent / "raven_g1_state_action.summary.json"
    if not summary_file.exists():
        raise SystemExit(f"Missing joint-order metadata: {summary_file}")
    metadata = json.loads(summary_file.read_text(encoding="utf-8"))
    if metadata.get("physics_hz") != 120.0:
        raise SystemExit("Expected 120 Hz capture")
    records = load_records(data_file, args.min_time_s)
    states, commands, joint_indices = build_arrays(records, metadata["g1_joint_order"])
    cuts = partitions(len(states))
    mean_s = torch.tensor(states[:cuts["train"][1]].mean(0), device=device)
    std_s = torch.tensor(np.maximum(states[:cuts["train"][1]].std(0), 0.01), device=device)
    mean_a = torch.tensor(commands[:cuts["train"][1]].mean(0), device=device)
    std_a = torch.tensor(np.maximum(commands[:cuts["train"][1]].std(0), 0.01), device=device)
    norms = (mean_s, std_s, mean_a, std_a)
    windows = {k: make_windows(states, commands, *v) for k, v in cuts.items()}
    train_x, train_a, train_y = (x.to(device) for x in windows["train"])
    model = Dynamics(states.shape[1], commands.shape[1]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = float("inf")
    best_weights = None

    print(f"[RAVEN WORLD MODEL] {len(records)} tracking-phase states, 120 Hz")
    print(f"[RAVEN WORLD MODEL] {len(windows['train'][0])} train / "
          f"{len(windows['validation'][0])} validation / "
          f"{len(windows['test'][0])} test windows")
    print(f"[RAVEN WORLD MODEL] Training on {device}; Isaac Sim NOT required")
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        running = 0.0
        for idx in order.split(args.batch_size):
            pred = predict(model, train_x[idx], train_a[idx], *norms)
            target = train_y[idx]
            loss = (((pred-target) / std_s)**2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Numerical instability in world-model training")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.item()) * len(idx)
        model.eval()
        with torch.no_grad():
            val_x, val_a, val_y = (x.to(device) for x in windows["validation"])
            val_pred = predict(model, val_x, val_a, *norms)
            val_loss = float(((((val_pred - val_y) / std_s)**2).mean()).item())
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 10 == 0 or epoch == 0 or epoch == args.epochs-1:
            print(f"[Epoch {epoch+1:3d}/{args.epochs}] "
                  f"train_mse={running/len(train_x):.6f} val_mse={val_loss:.6f}")

    model.load_state_dict(best_weights)
    model.eval()
    results = {k: score(model, data, norms, device)
               for k, data in windows.items()}
    results.update({"experiment": "SuperMimic-inspired OFFLINE 8-step RAVEN world model",
                    "is_official_supermimic": False,
                    "policy_trained": False, "deployed_to_simulator": False,
                    "source_trajectory": str(data_file),
                    "training_phase_starts_at_s": args.min_time_s,
                    "physics_hz": 120, "selected_joint_names": JOINT_NAMES,
                    "selected_joint_indices": joint_indices,
                    "capture_splits": cuts,
                    "best_validation_normalized_mse": best_loss,
                    "limitation": "One closed-loop actor-tracking trajectory; no independent episode test."})
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    torch.save({"state_dict": best_weights, "state_mean": mean_s.detach().cpu(),
                "state_std": std_s.detach().cpu(), "action_mean": mean_a.detach().cpu(),
                "action_std": std_a.detach().cpu(), "joint_names": JOINT_NAMES,
                "joint_indices": joint_indices, "horizon": HORIZON,
                "metrics": results}, out / "world_model.pt")
    print("[RAVEN WORLD MODEL] Test pan step-8 MAE: "
          f"{results['test']['pan_step8_mae_deg']:.4f} deg "
          "(persistence baseline "
          f"{results['test']['pan_persistence_step8_mae_deg']:.4f} deg)")
    print(f"[RAVEN WORLD MODEL] Saved: {out / 'metrics.json'}")
    print(f"[RAVEN WORLD MODEL] Checkpoint: {out / 'world_model.pt'}")


if __name__ == "__main__":
    main()
