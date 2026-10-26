
"""RAVEN: offline wall-clock benchmark of the V1.4 MPC optimisation.

This DOES NOT run Isaac Sim, change the G1, or train a policy.
It measures the frozen V2 model + 8-step/60-iteration differentiable plan
using real held-out 120 Hz states and last-known 10 Hz framing references.

The capture lacks per-joint physical limits. For TIMING ONLY, bounds use
current target +/-1 rad plus the actual V1.4 per-command trust region.
This is NOT a feasibility or physics validation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from raven_supermimic_world_model_v2 import (
    DT, HORIZON, JOINTS, ResidualDynamics, read_run,
)
from raven_supermimic_policy_v1 import (
    MAX_DELTA_RAD, current_reference, objective,
)

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = (
    ROOT
    / "raven_training_results/world_model_v2/world_model_v2.pt"
)

DEFAULT_DATA = (
    ROOT
    / "raven_training_results/supermimic_capture/"
      "20261026_012923_523533/raven_g1_state_action_120hz.jsonl"
)

DEFAULT_OUT = (
    ROOT
    / "raven_training_results/mpc_latency/benchmark.json"
)


def synchronise(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(path, device):
    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if ckpt.get("model_class") != "ResidualDynamics":
        raise ValueError(
            "Not the validated V2 residual checkpoint"
        )

    if (
        tuple(ckpt["joint_names"]) != JOINTS
        or int(ckpt["horizon"]) != HORIZON
    ):
        raise ValueError(
            "V2 checkpoint joint ordering/horizon mismatch"
        )

    model = ResidualDynamics(
        *ckpt["normalization"]
    ).to(device)

    model.load_state_dict(
        ckpt["state_dict"],
        strict=True,
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    return model


def plan_once(
    model,
    state,
    held,
    reference_abs_rad,
    iterations,
):
    """Match V1.4's bounded differentiable planning workload."""

    device = state.device

    delta = torch.tensor(
        MAX_DELTA_RAD,
        device=device,
        dtype=state.dtype,
    )

    max_step = torch.full_like(delta, 0.012)
    max_step[0] = 0.007

    # Timing-only placeholders for missing physical joint limits.
    lower = held - 1.0
    upper = held + 1.0

    reference = torch.tensor(
        [reference_abs_rad],
        device=device,
        dtype=state.dtype,
    )

    raw = torch.nn.Parameter(
        torch.zeros(
            (1, HORIZON, len(JOINTS)),
            device=device,
        )
    )

    optimiser = torch.optim.Adam(
        [raw],
        lr=0.08,
    )

    fallback = (
        held[:, None, :]
        .expand(-1, HORIZON, -1)
        .clone()
    )

    def rollout_projected():
        s = state
        actions = []
        predictions = []

        for i in range(HORIZON):
            anchor = (
                state[:, :8]
                if i == 0
                else actions[-1]
            )

            lo = torch.maximum(
                lower,
                torch.maximum(
                    held - delta,
                    anchor - max_step,
                ),
            )

            hi = torch.minimum(
                upper,
                torch.minimum(
                    held + delta,
                    anchor + max_step,
                ),
            )

            if bool(torch.any(lo > hi + 1e-7)):
                raise ValueError(
                    "Measured state is incompatible "
                    "with this command trust region"
                )

            suggested = (
                held
                + delta * torch.tanh(raw[:, i])
            )

            command = torch.minimum(
                torch.maximum(suggested, lo),
                hi,
            )

            s = model.step(s, command)

            actions.append(command)
            predictions.append(s)

        return (
            torch.stack(actions, 1),
            torch.stack(predictions, 1),
        )

    with torch.no_grad():
        baseline_forecast = model.rollout(
            state,
            fallback,
        )

        best_loss = float(
            objective(
                baseline_forecast,
                fallback,
                held,
                reference,
                state,
                delta,
            )[0]
        )

    best_action = fallback.detach().clone()

    for _ in range(iterations):
        actions, states = rollout_projected()

        loss = objective(
            states,
            actions,
            held,
            reference,
            state,
            delta,
        ).mean()

        if not bool(torch.isfinite(loss)):
            raise ValueError(
                "Non-finite planning loss"
            )

        with torch.no_grad():
            if float(loss) < best_loss:
                best_loss = float(loss)
                best_action = actions.detach().clone()

        optimiser.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [raw],
            1.0,
        )

        optimiser.step()

    with torch.no_grad():
        actions, states = rollout_projected()

        candidate_loss = float(
            objective(
                states,
                actions,
                held,
                reference,
                state,
                delta,
            )[0]
        )

        if candidate_loss < best_loss:
            best_action = actions.detach().clone()

        model.rollout(state, best_action)

    return best_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )

    p.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
    )

    p.add_argument(
        "--csv",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    p.add_argument(
        "--iterations",
        type=int,
        default=60,
    )

    p.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    p.add_argument(
        "--sample-times",
        type=float,
        nargs="+",
        default=[2.0, 10.0, 20.0, 30.0],
    )

    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
    )

    args = p.parse_args()

    if args.iterations < 1 or args.repeats < 1:
        p.error(
            "Iterations and repeats must be positive"
        )

    if not args.model.is_file() or not args.data.is_file():
        p.error(
            "Model or capture file does not exist"
        )

    csv_path = (
        args.csv
        or args.data.parent / "raven_actor_tracking.csv"
    )

    if not csv_path.is_file():
        p.error(
            f"Tracking CSV not found: {csv_path}"
        )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        p.error(
            "CUDA requested but unavailable"
        )

    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            or (
                args.device == "auto"
                and torch.cuda.is_available()
            )
        )
        else "cpu"
    )

    model = load_model(
        args.model,
        device,
    )

    states_np, actions_np = read_run(
        args.data,
        min_time=27.5,
    )

    pan_zero = float(states_np[0, 16])

    rows = []
    all_ms = []

    for t in args.sample_times:
        idx = round(t / DT)

        if not 0 <= idx < len(states_np):
            p.error(
                f"Sample time {t}s is outside "
                "tracking-phase recording"
            )

        state = torch.from_numpy(
            states_np[idx:idx + 1].copy()
        ).to(device)

        held = torch.from_numpy(
            actions_np[idx:idx + 1].copy()
        ).to(device)

        current_ref_relative = float(
            current_reference(
                csv_path,
                np.array([t]),
            )[0]
        )

        reference_abs = (
            current_ref_relative
            + pan_zero
        )

        # Warm up before measuring wall-clock runtime.
        plan_once(
            model,
            state,
            held,
            reference_abs,
            iterations=args.iterations,
        )

        synchronise(device)

        durations = []

        for _ in range(args.repeats):
            synchronise(device)

            started = time.perf_counter()

            plan_once(
                model,
                state,
                held,
                reference_abs,
                iterations=args.iterations,
            )

            synchronise(device)

            elapsed_ms = (
                time.perf_counter() - started
            ) * 1000.0

            durations.append(elapsed_ms)

        all_ms.extend(durations)

        rows.append({
            "tracking_time_s": t,
            "planner_wall_ms": durations,
            "median_wall_ms": statistics.median(
                durations
            ),
        })

        print(
            f"[MPC TIMING] t={t:5.1f}s: "
            f"median "
            f"{statistics.median(durations):8.2f} ms"
        )

    med = statistics.median(all_ms)
    p95 = float(np.percentile(all_ms, 95))

    result = {
        "experiment":
            "V1.4 MPC offline measured compute-latency test",

        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "device": str(device),
        "iterations": args.iterations,
        "rollout_steps": HORIZON,
        "repeats_per_state": args.repeats,

        "median_ms": med,
        "p95_ms": p95,

        "physics_120hz_deadline_ms": 1000.0 / 120.0,
        "vision_10hz_deadline_ms": 100.0,

        "fits_120hz_by_median":
            med <= 1000.0 / 120.0,

        "fits_10hz_by_p95":
            p95 <= 100.0,

        "cases": rows,

        "limitations": [
            "Offline timing; actual Isaac Sim contention "
            "and camera work are not included.",

            "The measured algorithm matches V1.4 "
            "planning but uses wide joint-limit "
            "placeholders for timing only.",

            "Frame references come from last-known "
            "RGB-D samples; planning is not "
            "physically executed.",

            "Meeting a deadline offline does not "
            "establish deterministic real-time operation.",
        ],
    }

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.out.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(
        f"[MPC TIMING] median={med:.2f}ms "
        f"| p95={p95:.2f}ms "
        f"| 120Hz budget=8.33ms "
        f"| 10Hz budget=100ms"
    )

    print(
        f"[MPC TIMING] Saved: {args.out}"
    )


if __name__ == "__main__":
    main()
