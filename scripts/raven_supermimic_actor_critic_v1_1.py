
"""RAVEN / SuperMimic-inspired actor-critic V1.1: conservative offline training.

Uses the UNCHANGED V1 Actor/Critic classes and the FROZEN V2 world model.
Adds: larger tilt/rate penalties, conservative same-run checkpoint selection,
and a one-time independent third-trajectory model-based evaluation.

This is NOT the official SuperMimic implementation and NOT Isaac Sim policy
validation. There is no learned contact, collision, or physical joint-limit model.

PowerShell (from raven-g1 with env_isaaclab active):
  python scripts/raven_supermimic_actor_critic_v1_1.py --epochs 10 --updates-per-epoch 50 --batch-size 64

Do not use the independent third trajectory to select weights or tune the loss.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from raven_supermimic_world_model_v2 import DT, HORIZON, JOINTS, read_run
from raven_supermimic_policy_v1 import current_reference
from raven_supermimic_actor_critic_v1 import (
    Actor, Critic, load_v2, build_samples, td_lambda_targets, benchmark_actor,
    DEFAULT_RUNS, DEFAULT_MODEL, ROOT,
)

DEG = 180.0 / math.pi
DEFAULT_THIRD = (ROOT / "raven_training_results" / "supermimic_capture"
                 / "20261026_012923_523533" / "raven_g1_state_action_120hz.jsonl")
DEFAULT_OUT = ROOT / "raven_training_results" / "policy_v2_actor_critic" / "train_v1_1"


def imagined_rollout(model, actor, initial, held, reference, horizon, cfg):
    """Differentiable eight-step feedback rollout; V2 weights stay frozen."""
    state = initial
    previous_command = held
    actions, rewards, states = [], [], []
    delta = actor.delta
    max_change = torch.full_like(delta, 0.012)
    max_change[0] = 0.007

    for step in range(horizon):
        candidate = actor(state, held, reference)

        # First action is bounded by MEASURED joint positions.
        # Later imagined actions are constrained relative to
        # the preceding planned COMMAND.
        anchor = initial[:, :8] if step == 0 else previous_command

        lower = torch.maximum(held - delta, anchor - max_change)
        upper = torch.minimum(held + delta, anchor + max_change)

        if bool(torch.any(lower > upper + 1e-6)):
            raise ValueError(
                "Infeasible actor trust region in imagined rollout"
            )

        command = torch.minimum(torch.maximum(candidate, lower), upper)
        next_state = model.step(state, command)

        pan_error = (next_state[:, 16] - reference[:, 0]) * DEG
        tilt_change = (next_state[:, 17] - initial[:, 17]) * DEG

        action_magnitude = ((command - held) / delta).square().mean(-1)
        command_rate = ((command - previous_command) / delta).square().mean(-1)

        reward = -(
            pan_error.square()
            + cfg.tilt_weight * tilt_change.square()
            + cfg.action_weight * action_magnitude
            + cfg.rate_weight * command_rate
        )

        rewards.append(reward)
        states.append(state)
        actions.append(command)

        state = next_state
        previous_command = command

    return (
        torch.stack(rewards, 1),
        torch.stack(states, 1),
        torch.stack(actions, 1),
        state,
    )


@torch.no_grad()
def evaluate(model, actor, samples, device, horizon, batch_size, cfg):
    """Predictions under frozen V2 only; never interpreted as real physics."""
    actor.eval()

    result = {k: [] for k in (
        "held_pan_abs_error_deg",
        "actor_pan_abs_error_deg",
        "held_tilt_delta_deg",
        "actor_tilt_delta_deg",
        "actor_action_offset_normalised",
        "actor_command_step_normalised",
        "actor_max_command_step_rad",
    )}

    for start in range(0, len(samples[0]), batch_size):
        state, held, ref = [
            t[start:start + batch_size].to(device) for t in samples
        ]

        targets = held[:, None, :].expand(-1, horizon, -1)
        baseline = model.rollout(state, targets)[:, -1]

        _, _, actions, predicted = imagined_rollout(
            model, actor, state, held, ref, horizon, cfg
        )

        result["held_pan_abs_error_deg"].extend(
            ((baseline[:, 16] - ref[:, 0]).abs() * DEG).cpu().tolist()
        )

        result["actor_pan_abs_error_deg"].extend(
            ((predicted[:, 16] - ref[:, 0]).abs() * DEG).cpu().tolist()
        )

        result["held_tilt_delta_deg"].extend(
            ((baseline[:, 17] - state[:, 17]).abs() * DEG).cpu().tolist()
        )

        result["actor_tilt_delta_deg"].extend(
            ((predicted[:, 17] - state[:, 17]).abs() * DEG).cpu().tolist()
        )

        offsets = ((actions - held[:, None]) / actor.delta).abs().mean((1, 2))

        prior = torch.cat(
            (held[:, None, :], actions[:, :-1]), dim=1
        )

        rate = ((actions - prior) / actor.delta).abs().mean((1, 2))
        max_rate = (actions - prior).abs().amax((1, 2))

        result["actor_action_offset_normalised"].extend(
            offsets.cpu().tolist()
        )

        result["actor_command_step_normalised"].extend(
            rate.cpu().tolist()
        )

        result["actor_max_command_step_rad"].extend(
            max_rate.cpu().tolist()
        )

    actor.train()

    return {
        key: float(np.mean(values))
        for key, values in result.items()
    }


def eligible_and_score(m, cfg):
    """Select on same-run validation ONLY, with a tilt acceptance constraint."""
    tilt_limit = min(
        cfg.tilt_absolute_ceiling_deg,
        m["held_tilt_delta_deg"] + cfg.tilt_margin_deg
    )

    pan_gain = (
        m["held_pan_abs_error_deg"]
        - m["actor_pan_abs_error_deg"]
    )

    eligible = (
        pan_gain >= cfg.min_pan_gain_deg
        and m["actor_tilt_delta_deg"] <= tilt_limit
    )

    score = (
        m["actor_pan_abs_error_deg"]
        + 2.0 * m["actor_tilt_delta_deg"]
        + 0.03 * m["actor_command_step_normalised"]
    )

    return bool(eligible), float(score), float(tilt_limit)


def independent_samples(path, stride):
    """Third capture: NEVER sampled during training or checkpoint selection."""
    state, held = read_run(path, min_time=27.5)
    csv = path.parent / "raven_actor_tracking.csv"

    if not csv.is_file():
        raise FileNotFoundError(
            f"Independent capture RGB-D CSV missing: {csv}"
        )

    idx = np.arange(
        max(round(2 / DT), 0),
        len(state) - HORIZON,
        stride
    )

    if len(idx) < 8:
        raise ValueError(
            "Independent capture contains too few evaluation states"
        )

    relative_ref = current_reference(
        csv, idx.astype(np.float64) * DT
    )

    absolute_ref = relative_ref + np.float32(state[0, 16])

    return (
        torch.from_numpy(state[idx].copy().astype(np.float32)),
        torch.from_numpy(held[idx].copy().astype(np.float32)),
        torch.from_numpy(absolute_ref[:, None].astype(np.float32)),
    )


def checkpoint_dict(actor, critic, model_path, epoch, cfg, selected):
    return {
        "model_class": "RAVENOfflineActorCriticV1_1",
        "official_supermimic_reproduction": False,

        "actor_state_dict": copy.deepcopy({
            k: v.detach().cpu()
            for k, v in actor.state_dict().items()
        }),

        "critic_state_dict": copy.deepcopy({
            k: v.detach().cpu()
            for k, v in critic.state_dict().items()
        }),

        "joint_names": JOINTS,
        "state_dimensions": 20,
        "action_dimensions": 8,
        "observation_dimensions": 29,
        "horizon": cfg.horizon,
        "model_source": str(model_path.resolve()),
        "selected_epoch": int(epoch),
        "passed_validation_gate": bool(selected),

        "reward_weights": {
            "pan": 1.0,
            "tilt": cfg.tilt_weight,
            "action_magnitude": cfg.action_weight,
            "command_rate": cfg.rate_weight,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", type=Path, nargs=2, default=DEFAULT_RUNS)
    p.add_argument("--independent-data", type=Path, default=DEFAULT_THIRD)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--updates-per-epoch", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    p.add_argument("--tilt-weight", type=float, default=5.0)
    p.add_argument("--action-weight", type=float, default=0.08)
    p.add_argument("--rate-weight", type=float, default=0.20)

    p.add_argument("--tilt-margin-deg", type=float, default=0.02)
    p.add_argument("--tilt-absolute-ceiling-deg", type=float, default=0.08)
    p.add_argument("--min-pan-gain-deg", type=float, default=0.002)

    args = p.parse_args()

    if any(v < 1 for v in (
        args.epochs,
        args.updates_per_epoch,
        args.batch_size,
        args.stride,
        args.horizon,
    )):
        p.error(
            "Epochs, updates, batch size, stride and horizon must be positive"
        )

    if args.horizon > HORIZON:
        p.error(
            "V2 is independently validated for at most 8 physics steps"
        )

    if any(v < 0 for v in (
        args.tilt_weight,
        args.action_weight,
        args.rate_weight,
        args.tilt_margin_deg,
        args.tilt_absolute_ceiling_deg,
        args.min_pan_gain_deg,
    )):
        p.error(
            "Reward and acceptance parameters must be non-negative"
        )

    for path in (args.model, args.independent_data, *args.data):
        if not path.is_file():
            p.error(f"Required input missing: {path}")

    if len({str(path.resolve()) for path in args.data}) != 2:
        p.error(
            "Exactly two distinct model-development capture runs are required"
        )

    if args.independent_data.resolve() in {
        path.resolve() for path in args.data
    }:
        p.error(
            "Third independent capture cannot be used to train or select weights"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")

    device = torch.device(
        "cuda"
        if args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        )
        else "cpu"
    )

    model, model_ckpt = load_v2(args.model, device)

    training_recordings = {
        str(Path(path).resolve())
        for path in model_ckpt.get("metrics", {}).get("training_recordings", [])
    }

    if training_recordings and any(
        str(path.resolve()) not in training_recordings
        for path in args.data
    ):
        p.error(
            "Policy training inputs must be V2 development recordings"
        )

    if str(args.independent_data.resolve()) in training_recordings:
        p.error(
            "Independent capture is present in World Model V2 training metadata"
        )

    train, validation, sources = build_samples(
        args.data, stride=args.stride
    )

    actor = Actor(
        model.state_mean, model.state_scale
    ).to(device)

    critic = Critic(
        model.state_mean, model.state_scale
    ).to(device)

    target_critic = copy.deepcopy(critic)

    for param in target_critic.parameters():
        param.requires_grad_(False)

    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=2e-4
    )

    critic_optim = torch.optim.Adam(
        critic.parameters(), lr=3e-4
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    initial_eval = evaluate(
        model, actor, validation, device,
        args.horizon, args.batch_size, args
    )

    history = []
    selected = None
    best_score = float("inf")
    best_epoch = 0

    best_weights = checkpoint_dict(
        actor, critic, args.model, 0, args, False
    )

    gamma, lam = 0.98, 0.95

    print(
        f"[RAVEN V1.1] {len(train[0])} training states, "
        f"{len(validation[0])} same-run validation states"
    )

    print(
        f"[RAVEN V1.1] frozen V2, {device}, "
        f"horizon {args.horizon}, "
        f"tilt weight {args.tilt_weight}, "
        f"rate weight {args.rate_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        losses_actor, losses_critic = [], []

        actor.train()
        critic.train()

        for _ in range(args.updates_per_epoch):
            ids = torch.randint(
                len(train[0]), (args.batch_size,)
            )

            s, held, ref = [
                t[ids].to(device) for t in train
            ]

            for param in critic.parameters():
                param.requires_grad_(False)

            rewards, states, _, final = imagined_rollout(
                model, actor, s, held, ref,
                args.horizon, args
            )

            terminal = target_critic(
                final, held, ref
            )

            weights = (
                gamma ** torch.arange(
                    args.horizon, device=device
                )
            ).to(rewards.dtype)

            actor_loss = -(
                (rewards * weights[None]).sum(1)
                + gamma ** args.horizon * terminal
            ).mean()

            if not bool(torch.isfinite(actor_loss)):
                raise RuntimeError(
                    "Nonfinite actor loss: refusing checkpoint"
                )

            actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()

            nn.utils.clip_grad_norm_(
                actor.parameters(), 1.0
            )

            actor_optim.step()

            for param in critic.parameters():
                param.requires_grad_(True)

            targets = td_lambda_targets(
                rewards.detach(),
                states.detach(),
                final.detach(),
                held.detach(),
                ref.detach(),
                target_critic,
                gamma=gamma,
                lam=lam,
            )

            s_flat = states.detach().reshape(-1, 20)

            held_flat = held[:, None].expand(
                -1, args.horizon, -1
            ).reshape(-1, 8)

            ref_flat = ref[:, None].expand(
                -1, args.horizon, -1
            ).reshape(-1, 1)

            predicted_value = critic(
                s_flat, held_flat, ref_flat
            ).reshape(-1, args.horizon)

            critic_loss = nn.functional.smooth_l1_loss(
                predicted_value, targets
            )

            if not bool(torch.isfinite(critic_loss)):
                raise RuntimeError(
                    "Nonfinite critic loss: refusing checkpoint"
                )

            critic_optim.zero_grad(set_to_none=True)
            critic_loss.backward()

            nn.utils.clip_grad_norm_(
                critic.parameters(), 1.0
            )

            critic_optim.step()

            with torch.no_grad():
                for target, current in zip(
                    target_critic.parameters(),
                    critic.parameters(),
                ):
                    target.lerp_(current, 0.01)

            losses_actor.append(
                float(actor_loss.detach())
            )

            losses_critic.append(
                float(critic_loss.detach())
            )

        val = evaluate(
            model, actor, validation, device,
            args.horizon, args.batch_size, args
        )

        if not all(np.isfinite(v) for v in val.values()):
            raise RuntimeError(
                "Nonfinite validation measurement"
            )

        accepted, score, ceiling = eligible_and_score(
            val, args
        )

        record = {
            "epoch": epoch,
            "actor_loss": float(np.mean(losses_actor)),
            "critic_loss": float(np.mean(losses_critic)),
            **val,
            "validation_score": score,
            "tilt_acceptance_limit_deg": ceiling,
            "accepted_by_selection_gate": accepted,
        }

        history.append(record)

        if accepted and score < best_score:
            best_score = score
            best_epoch = epoch
            selected = record

            best_weights = checkpoint_dict(
                actor, critic,
                args.model, epoch, args, True
            )

            torch.save(
                best_weights,
                out / "actor_critic_v1_1_best.pt"
            )

        print(
            f"[RAVEN V1.1] epoch {epoch:02d}/{args.epochs} "
            f"pan={val['actor_pan_abs_error_deg']:.4f}deg "
            f"tilt={val['actor_tilt_delta_deg']:.4f}deg "
            f"limit={ceiling:.4f}deg "
            f"accepted={accepted} "
            f"{'BEST' if best_epoch == epoch else ''}"
        )

    last_weights = checkpoint_dict(
        actor, critic,
        args.model, args.epochs, args, False
    )

    torch.save(
        last_weights,
        out / "actor_critic_v1_1_last.pt"
    )

    if selected is None:
        torch.save(
            best_weights,
            out / "actor_critic_v1_1_best.pt"
        )

        print(
            "[RAVEN V1.1] WARNING: no epoch met both "
            "pan-gain and tilt constraints; "
            "best=initial fallback"
        )

    actor.load_state_dict(
        best_weights["actor_state_dict"],
        strict=True
    )

    best_same_run = evaluate(
        model, actor, validation, device,
        args.horizon, args.batch_size, args
    )

    independent = independent_samples(
        args.independent_data,
        args.stride
    )

    best_unseen = evaluate(
        model, actor, independent, device,
        args.horizon, args.batch_size, args
    )

    inference = benchmark_actor(
        actor, validation, device, repeats=200
    )

    report = {
        "experiment":
            "RAVEN SuperMimic-inspired actor-critic V1.1, tilt-aware offline bootstrap",

        "official_supermimic_reproduction": False,
        "world_model_frozen": True,
        "policy_validated_in_isaac_sim": False,
        "independent_evaluation_is_only_world_model_forecast": True,
        "third_run_used_in_training_or_checkpoint_selection": False,

        "reward_weights": best_weights["reward_weights"],

        "selection_rule": {
            "minimum_pan_gain_deg": args.min_pan_gain_deg,
            "max_tilt_above_held_deg": args.tilt_margin_deg,
            "absolute_tilt_ceiling_deg": args.tilt_absolute_ceiling_deg,

            "score":
                "mean_pan_abs_deg + 2*mean_tilt_change_abs_deg + 0.03*mean_normalised_command_step",

            "selection_data":
                "last 20% of two development runs; independent third not consulted",
        },

        "epochs": args.epochs,
        "updates_per_epoch": args.updates_per_epoch,
        "batch_size": args.batch_size,
        "horizon_steps": args.horizon,

        "training_capture_sources": sources,
        "initial_epoch_0_validation": initial_eval,
        "history": history,

        "best_epoch": best_epoch,
        "best_epoch_passed_selection_gate": bool(
            selected is not None
        ),

        "best_accepted_validation": selected,
        "best_same_run_validation": best_same_run,

        "unseen_third_capture": str(
            args.independent_data.resolve()
        ),

        "independent_third_model_forecast": best_unseen,

        "best_actor_offline_inference_latency": inference,

        "best_checkpoint": str(
            out / "actor_critic_v1_1_best.pt"
        ),

        "final_epoch_checkpoint": str(
            out / "actor_critic_v1_1_last.pt"
        ),

        "limitations": [
            "Independent third trajectory is independent of training, but still evaluated using frozen V2 predictions, not Isaac Sim physics.",
            "World Model V2 lacks contact/grasp force, collision, hard joint-limit and observation-delay dynamics.",
            "Policy is not trained on new simulator transitions; official SuperMimic algorithm has not been reproduced.",
            "No closed-loop simulated tracking, real-time integrated inference, or long-horizon stability demonstrated.",
            "Repeated hyperparameter tuning based on the third run would invalidate its role as an independent evaluation.",
        ],
    }

    (out / "training_metrics_v1_1.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(
        f"[RAVEN V1.1] BEST epoch={best_epoch} "
        f"passed_gate={selected is not None}"
    )

    print(
        "[RAVEN V1.1] same-run predicted "
        f"pan={best_same_run['actor_pan_abs_error_deg']:.4f}deg "
        f"tilt={best_same_run['actor_tilt_delta_deg']:.4f}deg"
    )

    print(
        "[RAVEN V1.1] unseen-third MODEL prediction "
        f"pan={best_unseen['actor_pan_abs_error_deg']:.4f}deg "
        f"tilt={best_unseen['actor_tilt_delta_deg']:.4f}deg"
    )

    print(
        "[RAVEN V1.1] actor inference "
        f"median={inference['median_ms']:.3f}ms "
        f"p95={inference['p95_ms']:.3f}ms"
    )

    print(
        "[RAVEN V1.1] No physical policy validation; "
        "do NOT deploy best checkpoint to G1"
    )

    print(
        f"[RAVEN V1.1] Metrics saved: "
        f"{out / 'training_metrics_v1_1.json'}"
    )


if __name__ == "__main__":
    main()
