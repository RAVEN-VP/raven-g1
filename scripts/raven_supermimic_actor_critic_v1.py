"""RAVEN Policy V2 / actor-critic bootstrap, inspired by SuperMimic (CoRL 2026).

This is the FIRST offline implementation step, not an official SuperMimic
reproduction and not yet simulator-in-the-loop policy training.

- Reads ONLY two existing capture runs used in V2 model development.
- Loads the frozen, validated differentiable World Model V2.
- Trains an actor by backpropagation through imagined G1-tripod dynamics.
- Trains a critic using bootstrapped TD(lambda) returns from imagined states.
- Uses the latest ALREADY observed 10 Hz desired-pan reference in the SAME
  absolute pan coordinate system as the logged physics state.
- Holds the observed reference constant for each short imagined rollout.
- Preserves the third unseen trajectory for independent evaluation.
- NEVER launches Isaac Sim or sends commands to the robot.

The original SuperMimic alternates NEW simulator data, world-model fitting and
32-step policy/critic updates. This bootstrap freezes the model and starts with
8-step rollouts because V2 has only been validated over eight steps.

Run from C:\\RAVEN-Isaac\\raven-g1 with env_isaaclab active:
    python scripts/raven_supermimic_actor_critic_v1.py --epochs 10
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from raven_supermimic_world_model_v2 import (
    DT, HORIZON, JOINTS, ResidualDynamics, read_run,
)
from raven_supermimic_policy_v1 import (
    MAX_DELTA_RAD, current_reference,
)

ROOT = Path(__file__).resolve().parent.parent
CAPTURE = ROOT / "raven_training_results" / "supermimic_capture"
DEFAULT_RUNS = [
    CAPTURE / "20261025_230816_212515" / "raven_g1_state_action_120hz.jsonl",
    CAPTURE / "20261026_004925_836186" / "raven_g1_state_action_120hz.jsonl",
]
DEFAULT_MODEL = ROOT / "raven_training_results" / "world_model_v2" / "world_model_v2.pt"
DEFAULT_OUTPUT = ROOT / "raven_training_results" / "policy_v2_actor_critic"
DEG = 180.0 / math.pi


class Actor(nn.Module):
    """Amortised G1 joint-command policy; zero final layer starts at held commands."""

    def __init__(self, state_mean: torch.Tensor, state_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("state_mean", state_mean.detach().clone().float())
        self.register_buffer("state_scale", state_scale.detach().clone().float())
        self.net = nn.Sequential(
            nn.Linear(20 + 8 + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 8),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("delta", torch.tensor(MAX_DELTA_RAD, dtype=torch.float32))

    def forward(self, state, held, reference_abs):
        # Use a bounded relative reference: reference is absolute physics pan,
        # NOT the camera CSV's pan-zeroed relative angle.
        pan_error = (reference_abs - state[:, 16:17]) / 0.15
        obs = torch.cat((
            (state - self.state_mean) / self.state_scale,
            (held - state[:, :8]) / 0.05,
            pan_error,
        ), dim=-1)
        return held + self.delta * torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Bootstrap imagined returns beyond the short training horizon."""

    def __init__(self, state_mean: torch.Tensor, state_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("state_mean", state_mean.detach().clone().float())
        self.register_buffer("state_scale", state_scale.detach().clone().float())
        self.net = nn.Sequential(
            nn.Linear(20 + 8 + 1, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, held, reference_abs):
        pan_error = (reference_abs - state[:, 16:17]) / 0.15
        features = torch.cat((
            (state - self.state_mean) / self.state_scale,
            (held - state[:, :8]) / 0.05,
            pan_error,
        ), dim=-1)
        return self.net(features).squeeze(-1)


def load_v2(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_class") != "ResidualDynamics":
        raise ValueError("Expected frozen RAVEN World Model V2")
    if tuple(checkpoint["joint_names"]) != JOINTS or int(checkpoint["horizon"]) != HORIZON:
        raise ValueError("World Model V2 joint order or trained horizon differs")
    model = ResidualDynamics(*checkpoint["normalization"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def build_samples(paths, min_time_s=27.5, stride=12):
    """Only the first 80% of each capture is used for actor learning."""
    train_parts, val_parts, sources = [], [], []
    for path in paths:
        states, held = read_run(path, min_time=min_time_s)
        csv_path = path.parent / "raven_actor_tracking.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Corresponding RGB-D CSV missing: {csv_path}")
        pan_zero = float(states[0, 16])
        split = int(len(states) * 0.80)
        # Do not admit incomplete short-horizon windows across the split.
        for tag, first, stop in (
            ("train", max(round(2.0 / DT), 0), split - HORIZON),
            ("validation", split + HORIZON, len(states) - HORIZON),
        ):
            indices = np.arange(first, stop, stride, dtype=np.int64)
            if len(indices) < 8:
                raise ValueError(f"Too few {tag} samples in {path}")
            # Reference lookup is a sample-and-hold of PAST camera samples.
            reference = current_reference(csv_path, indices.astype(float) * DT)
            reference += np.float32(pan_zero)  # CSV relative -> physics absolute
            part = (
                states[indices].copy(), held[indices].copy(),
                reference[:, None].copy(),
            )
            (train_parts if tag == "train" else val_parts).append(part)
        sources.append({
            "capture": str(path.resolve()), "tracking_csv": str(csv_path.resolve()),
            "tracking_phase_pan_zero_rad": pan_zero,
            "total_tracking_physics_states": int(len(states)),
            "training_fraction": 0.8,
        })

    def concatenate(parts):
        return tuple(torch.from_numpy(np.concatenate([p[i] for p in parts]).astype(np.float32))
                     for i in range(3))
    return concatenate(train_parts), concatenate(val_parts), sources


def imagined_rollout(model, actor, initial, held, reference, horizon):
    """Differentiable closed-loop rollout; gradients go through model to actor."""
    state = initial
    previous_target = None
    rewards, states, actions = [], [], []
    delta = actor.delta
    for index in range(horizon):
        candidate = actor(state, held, reference)
        # The offline log has no physical joint-limit metadata. These are
        # EXPERIMENTAL trust-region bounds, NOT hard robot safety guarantees.
        previous = initial[:, :8] if index == 0 else previous_target
        max_change = torch.full_like(delta, 0.012)
        max_change[0] = 0.007
        lower = torch.maximum(held - delta, previous - max_change)
        upper = torch.minimum(held + delta, previous + max_change)
        if torch.any(lower > upper + 1e-6):
            raise ValueError("Training state has infeasible command interval")
        command = torch.clamp(candidate, min=lower, max=upper)
        next_state = model.step(state, command)

        pan_error_deg = (next_state[:, 16] - reference[:, 0]) * DEG
        tilt_delta_deg = (next_state[:, 17] - initial[:, 17]) * DEG
        normalized_command = (command - held) / delta
        # Penalise command oscillations and preserve tripod tilt.
        smooth_cost = (normalized_command.square()).mean(-1)
        if previous_target is not None:
            smooth_cost = smooth_cost + 0.5 * (((command - previous_target) / delta).square().mean(-1))
        # Proxy reward. There is NO differentiable contact/collision model.
        reward = (
            - pan_error_deg.square()
            - 0.3 * tilt_delta_deg.square()
            - 0.025 * smooth_cost
        )
        rewards.append(reward)
        states.append(state)
        actions.append(command)
        state = next_state
        previous_target = command
    return (
        torch.stack(rewards, dim=1),
        torch.stack(states, dim=1),
        torch.stack(actions, dim=1),
        state,
    )


@torch.no_grad()
def td_lambda_targets(rewards, states, final, held, reference, critic_target,
                      gamma=0.98, lam=0.95):
    """Bootstrap targets: G_t = r_t + γ[(1−λ)V(s_{t+1}) + λG_{t+1}]."""
    batch, horizon = rewards.shape
    # Stop gradients: critic targets must never differentiate into actor/model.
    next_states = torch.cat((states[:, 1:], final[:, None, :]), dim=1)
    h = held[:, None, :].expand(batch, horizon, -1).reshape(-1, 8)
    refs = reference[:, None, :].expand(batch, horizon, -1).reshape(-1, 1)
    values = critic_target(next_states.reshape(-1, 20), h, refs).view(batch, horizon)
    result = torch.empty_like(rewards)
    future = values[:, -1]
    for index in reversed(range(horizon)):
        future = rewards[:, index] + gamma * (
            (1.0 - lam) * values[:, index] + lam * future
        )
        result[:, index] = future
    return result


def evaluate(model, actor, samples, device, horizon, batch_size):
    """Offline model forecasts only; NOT an Isaac Sim performance evaluation."""
    actor.eval()
    metrics = {"held_pan_abs_error_deg": [], "actor_pan_abs_error_deg": [],
               "held_tilt_delta_deg": [], "actor_tilt_delta_deg": []}
    with torch.no_grad():
        for chunk in range(0, len(samples[0]), batch_size):
            s, held, ref = [x[chunk:chunk + batch_size].to(device) for x in samples]
            h = held[:, None, :].expand(-1, horizon, -1)
            baseline = model.rollout(s, h)[:, -1, :]
            _, _, _, predicted = imagined_rollout(model, actor, s, held, ref, horizon)
            metrics["held_pan_abs_error_deg"].extend(((baseline[:, 16] - ref[:, 0]).abs() * DEG).cpu().tolist())
            metrics["actor_pan_abs_error_deg"].extend(((predicted[:, 16] - ref[:, 0]).abs() * DEG).cpu().tolist())
            metrics["held_tilt_delta_deg"].extend(((baseline[:, 17] - s[:, 17]).abs() * DEG).cpu().tolist())
            metrics["actor_tilt_delta_deg"].extend(((predicted[:, 17] - s[:, 17]).abs() * DEG).cpu().tolist())
    actor.train()
    return {key: float(np.mean(values)) for key, values in metrics.items()}


def benchmark_actor(actor, examples, device, repeats=200):
    actor.eval()
    s, held, ref = [t[:1].to(device) for t in examples]
    with torch.no_grad():
        for _ in range(30):
            actor(s, held, ref)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings = []
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            actor(s, held, ref)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000)
    actor.train()
    return {"median_ms": float(statistics.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "repeats": repeats, "device": str(device),
            "excludes": "Isaac Sim, telemetry, Python loop integration, RGB-D, communications"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, nargs=2, default=DEFAULT_RUNS,
                   help="Two V2 development runs (not the independent third run)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--updates-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = p.parse_args()
    if any(x < 1 for x in (args.epochs, args.updates_per_epoch, args.batch_size, args.horizon, args.stride)):
        p.error("All epochs, updates, batch-size, horizon and stride must be positive")
    if args.horizon > HORIZON:
        p.error("Start with --horizon <= 8; World Model V2 has been validated only at 8 physics steps")
    for path in [args.model] + list(args.data):
        if not path.is_file():
            p.error(f"Missing input: {path}")
    if len({str(x.resolve()) for x in args.data}) != 2:
        p.error("Provide TWO distinct model-development captures")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model, ckpt = load_v2(args.model, device)
    # Guard against accidentally training on the third independent V2 evaluation.
    training_recordings = {
        str(Path(t).resolve()) for t in ckpt.get("metrics", {}).get("training_recordings", [])
    }
    if training_recordings:
        for path in args.data:
            if str(path.resolve()) not in training_recordings:
                p.error(f"Policy bootstrap must use the model's development captures, not held-out evaluation: {path}")
    train, validation, sources = build_samples(args.data, stride=args.stride)
    actor = Actor(model.state_mean, model.state_scale).to(device)
    critic = Critic(model.state_mean, model.state_scale).to(device)
    target_critic = copy.deepcopy(critic)
    for param in target_critic.parameters():
        param.requires_grad_(False)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    history = []
    print(f"[RAVEN ACTOR-CRITIC] {len(train[0])} training states, {len(validation[0])} validation states, {device}")
    print(f"[RAVEN ACTOR-CRITIC] Frozen V2, horizon={args.horizon}, gamma=0.98, lambda=0.95; no Isaac Sim")

    for epoch in range(1, args.epochs + 1):
        actor_losses, critic_losses = [], []
        for _ in range(args.updates_per_epoch):
            ix = torch.randint(len(train[0]), (args.batch_size,))
            state, held, ref = [t[ix].to(device) for t in train]
            # World-model weights fixed; retain differentiability wrt its input.
            for parameter in critic.parameters():
                parameter.requires_grad_(False)
            rewards, states, _, final = imagined_rollout(
                model, actor, state, held, ref, args.horizon)
            # Target critic parameters are frozen, but the bootstrap value
            # MUST remain differentiable w.r.t. final imagined state so its
            # gradient can reach the actor through V2 dynamics.
            terminal_value = target_critic(final, held, ref)
            weights = torch.pow(torch.tensor(0.98, device=device),
                                torch.arange(args.horizon, device=device))
            actor_loss = -(
                (rewards * weights[None, :]).sum(dim=1)
                + (0.98 ** args.horizon) * terminal_value
            ).mean()
            if not bool(torch.isfinite(actor_loss)):
                raise ValueError("Nonfinite actor objective; no checkpoint produced")
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()

            # Critic TD(lambda) update on the imagined trajectory.
            for parameter in critic.parameters():
                parameter.requires_grad_(True)
            target = td_lambda_targets(
                rewards.detach(), states.detach(), final.detach(),
                held.detach(), ref.detach(), target_critic)
            s_flat = states.detach().reshape(-1, 20)
            held_flat = held[:, None, :].expand(-1, args.horizon, -1).reshape(-1, 8)
            ref_flat = ref[:, None, :].expand(-1, args.horizon, -1).reshape(-1, 1)
            pred_value = critic(s_flat, held_flat, ref_flat).reshape(-1, args.horizon)
            critic_loss = torch.nn.functional.smooth_l1_loss(pred_value, target)
            if not bool(torch.isfinite(critic_loss)):
                raise ValueError("Nonfinite critic objective; no checkpoint produced")
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            with torch.no_grad():
                for dest, src in zip(target_critic.parameters(), critic.parameters()):
                    dest.lerp_(src, 0.01)
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))

        forecast = evaluate(model, actor, validation, device, args.horizon, args.batch_size)
        record = {"epoch": epoch, "actor_loss": float(np.mean(actor_losses)),
                  "critic_loss": float(np.mean(critic_losses)), **forecast}
        history.append(record)
        print(f"[RAVEN ACTOR-CRITIC] epoch {epoch:02d}/{args.epochs} "
              f"loss_actor={record['actor_loss']:.4f} loss_critic={record['critic_loss']:.4f} "
              f"predicted_pan_held={forecast['held_pan_abs_error_deg']:.4f}deg "
              f"actor={forecast['actor_pan_abs_error_deg']:.4f}deg")

    inference = benchmark_actor(actor, validation, device)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    saved = output / "actor_critic_v1.pt"
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "joint_names": JOINTS,
        "observation_dimensions": 29,
        "state_dimensions": 20,
        "action_dimensions": 8,
        "horizon": args.horizon,
        "model_source": str(args.model.resolve()),
        "model_class": "RAVENOfflineActorCriticBootstrap",
        "config": {k: ([str(p) for p in v] if isinstance(v, list) else str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
    }, saved)
    summary = {
        "experiment": "RAVEN SuperMimic-inspired offline actor/critic bootstrap V1",
        "official_supermimic_reproduction": False,
        "simulator_transitions_collected_during_training": False,
        "world_model_updated_during_training": False,
        "world_model_is_frozen": True,
        "trained_policy": True,
        "policy_validated_in_isaac_sim": False,
        "not_real_time_integrated": True,
        "notes": [
            "This is an offline proof of actor and critic learning through V2's imagined differentiable trajectories.",
            "Validation forecasts are correlated same-run holdouts and are not physical policy evidence.",
            "The official method alternates simulator replay collection, world-model fitting and 32-step actor/critic training.",
            "The dynamics state has NO explicit grasp/contact, collision or real joint-limit representation.",
            "Current desired-pan samples are held constant for each imagined rollout; future observations are not leaked.",
            "End-to-end Isaac Sim closed-loop, dataset shift, and inference latency under simulator load are NOT tested.",
        ],
        "frozen_v2_checkpoint": str(args.model.resolve()),
        "source_runs": sources,
        "epochs": args.epochs, "updates_per_epoch": args.updates_per_epoch,
        "batch_size": args.batch_size, "horizon_steps": args.horizon,
        "physics_hz": 120, "sample_and_hold_vision_hz": 10,
        "discount_gamma": 0.98, "td_lambda": 0.95,
        "history": history,
        "offline_inference_benchmark": inference,
        "saved_checkpoint": str(saved),
    }
    (output / "training_metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[RAVEN ACTOR-CRITIC] actor inference median={inference['median_ms']:.3f}ms p95={inference['p95_ms']:.3f}ms")
    print("[RAVEN ACTOR-CRITIC] OFFLINE MODEL FORECAST ONLY; NO PHYSICAL POLICY VALIDATION")
    print(f"[RAVEN ACTOR-CRITIC] Saved {saved}")
    print(f"[RAVEN ACTOR-CRITIC] Saved {output / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
