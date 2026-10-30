import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pilot_001"
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(20261030)

FS = 50
DT = 1 / FS
T = np.arange(0, 10, DT)

PHASES = {
    "reach": (0, 2),
    "align": (2, 4),
    "grip": (4, 6),
    "hold": (6, 8),
    "release": (8, 10),
}

CONDITIONS = [
    "baseline", "latency", "position",
    "smoothness", "grip", "continuity", "combined"
]

COMPONENTS = [
    "q_latency", "q_position",
    "q_smoothness", "q_grip"
]

# Provisional pilot weights, NOT final publication weights.
WEIGHTS = {k: 0.25 for k in COMPONENTS}


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3 - 2*x)


def desired_path():
    x = np.zeros(len(T))

    m = T < 2
    x[m] = 0.24 * smoothstep(T[m] / 2)

    m = (T >= 2) & (T < 4)
    x[m] = 0.24 + 0.02 * smoothstep((T[m]-2)/2)

    m = (T >= 4) & (T < 8)
    x[m] = 0.26

    m = T >= 8
    x[m] = 0.26 + 0.12 * smoothstep((T[m]-8)/2)

    y = 0.015 * np.sin(np.pi*T/10)
    z = np.full(len(T), 0.35)

    return np.column_stack((x, y, z))


def shifted_signal(signal, delay):
    return np.column_stack([
        np.interp(T + delay, T, signal[:, i])
        for i in range(signal.shape[1])
    ])


def estimate_delay(desired, actual):
    best_delay = 0
    best_error = float("inf")

    for delay in np.arange(0, 0.251, 0.01):
        shifted = shifted_signal(actual, delay)
        valid = T + delay <= T[-1]

        error = np.mean(
            np.sum((desired[valid] - shifted[valid])**2, axis=1)
        )

        if error < best_error:
            best_error = error
            best_delay = delay

    return float(best_delay)


def calculate_jerk(position):
    # Light smoothing prevents numerical differentiation from
    # being dominated by individual measurement-noise samples.
    filtered = np.column_stack([
        np.convolve(position[:, i], np.ones(5)/5, mode="same")
        for i in range(3)
    ])

    velocity = np.gradient(filtered, DT, axis=0)
    acceleration = np.gradient(velocity, DT, axis=0)
    jerk = np.gradient(acceleration, DT, axis=0)

    return float(np.sqrt(np.mean(np.sum(jerk**2, axis=1))))


def generate_hand_signals(condition, severity):
    """
    Synthetic Revo2-compatible telemetry.

    Two hands, six generic motor channels per hand.
    No finger-to-motor mapping is assumed.
    """

    closure = np.zeros(len(T))

    m = (T >= 4) & (T < 6)
    closure[m] = 0.05 + 0.75 * smoothstep((T[m]-4)/2)

    m = (T >= 6) & (T < 8)
    closure[m] = 0.80

    m = T >= 8
    closure[m] = 0.80 * (
        1 - smoothstep((T[m]-8)/2)
    )

    command = np.broadcast_to(
        closure[:, None, None], (len(T), 2, 6)
    ).copy()

    grip_severity = (
        severity if condition in ("grip", "combined") else 0
    )

    q = command + rng.normal(
        0, 0.008 + 0.045*grip_severity, command.shape
    )

    if grip_severity:
        q += (
            0.035 * grip_severity *
            np.sin(2*np.pi*7*T)[:, None, None]
        )

    q = np.clip(q, 0, 1)
    dq = np.gradient(q, DT, axis=0)

    load_mean = 0.68 - 0.42*grip_severity
    load = load_mean + rng.normal(
        0, 0.025 + 0.16*grip_severity, command.shape
    )

    active = ((T >= 4) & (T < 8))[:, None, None]
    load *= np.where(active, 1.0, 0.12)

    if grip_severity:
        load += (
            0.12 * grip_severity *
            np.sin(2*np.pi*5*T)[:, None, None]
        )

    load = np.clip(load, 0, 1)

    return command, q, dq, load


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)

    positive = scores[labels]
    negative = scores[~labels]

    if len(positive) == 0 or len(negative) == 0:
        return float("nan")

    comparisons = positive[:, None] - negative[None, :]

    return float(
        np.mean((comparisons > 0) + 0.5*(comparisons == 0))
    )


def run_trial(condition, severity, trial_id):
    desired = desired_path()

    latency = 0.035
    if condition in ("latency", "combined"):
        latency += 0.16 * severity

    actual = np.column_stack([
        np.interp(T-latency, T, desired[:, i])
        for i in range(3)
    ])

    position_factor = (
        severity if condition in ("position", "combined") else 0
    )

    actual += np.array([0.010, -0.006, 0.004]) * position_factor
    actual += rng.normal(
        0, 0.0008 + 0.018*position_factor, actual.shape
    )

    smooth_factor = (
        severity if condition in ("smoothness", "combined") else 0
    )

    actual += (
        0.0035 * smooth_factor *
        np.sin(2*np.pi*7*T)[:, None] *
        np.array([[1.0, 0.7, 0.4]])
    )

    # Simulate incomplete telemetry separately from task success.
    available = np.ones(len(T), dtype=bool)

    dropout = severity if condition == "continuity" else 0
    if condition == "combined":
        dropout = 0.35 * severity

    if dropout:
        count = int(len(T) * min(0.18, 0.02 + 0.14*dropout))
        indices = rng.choice(len(T), count, replace=False)
        available[indices] = False

        block = int((0.02 + 0.18*dropout) / DT)
        if block:
            start = rng.integers(0, len(T)-block)
            available[start:start+block] = False

    observed = actual.copy()
    observed[~available] = np.nan

    for i in range(3):
        valid = np.isfinite(observed[:, i])
        observed[:, i] = np.interp(
            T, T[valid], observed[valid, i]
        )

    estimated_delay = estimate_delay(desired, observed)
    compensated = shifted_signal(observed, estimated_delay)

    valid = T + estimated_delay <= T[-1]

    raw_error = np.linalg.norm(desired-observed, axis=1)
    compensated_error = np.linalg.norm(
        desired[valid]-compensated[valid], axis=1
    )

    raw_mm = float(np.mean(raw_error)*1000)
    compensated_mm = float(np.mean(compensated_error)*1000)

    jerk = calculate_jerk(observed)

    q_latency = np.exp(-latency/0.150)
    q_position = np.exp(-compensated_mm/20)
    q_smoothness = np.exp(-jerk/80)

    # Revo2 six-channel-per-hand telemetry.
    command, hand_q, hand_dq, hand_load = generate_hand_signals(
        condition, severity
    )

    hold = (T >= 6) & (T < 8)

    hold_load = hand_load[hold]
    load_cv = np.std(hold_load)/(np.mean(hold_load)+1e-6)
    finger_velocity = np.sqrt(np.mean(hand_dq[hold]**2))

    grip_instability = (
        0.65*np.clip(load_cv/0.50, 0, 1) +
        0.35*np.clip(finger_velocity/0.20, 0, 1)
    )

    q_grip = 1 - grip_instability

    components = {
        "q_latency": q_latency,
        "q_position": q_position,
        "q_smoothness": q_smoothness,
        "q_grip": q_grip
    }

    ctqi = 100 * sum(
        WEIGHTS[k]*components[k] for k in COMPONENTS
    )

    completeness = float(np.mean(available))
    longest_gap = 0
    current_gap = 0

    for present in available:
        current_gap = 0 if present else current_gap+1
        longest_gap = max(longest_gap, current_gap)

    max_gap_s = longest_gap*DT

    safety_valid = True
    protocol_valid = True
    continuity_valid = max_gap_s <= 0.10
    data_complete = completeness >= 0.98

    data_valid = (
        safety_valid and protocol_valid and
        continuity_valid and data_complete
    )

    # Independent outcome model: object and placement measurements.
    # Task success is never used to calculate CTQI.
    pf = severity if condition in ("position", "combined") else 0
    lf = severity if condition in ("latency", "combined") else 0
    sf = severity if condition in ("smoothness", "combined") else 0
    gf = severity if condition in ("grip", "combined") else 0

    placement_error = max(
        0, 8 + 58*pf + 38*lf + 24*sf + rng.normal(0, 5)
    )

    object_slip = max(
        0, 1.5 + 38*gf + 12*lf + rng.normal(0, 2)
    )

    object_retained = object_slip < 25

    task_success = (
        placement_error < 40 and object_retained and safety_valid
    )

    return {
        "trial_id": trial_id,
        "condition": condition,
        "severity": severity,
        "latency_ms": latency*1000,
        "estimated_delay_ms": estimated_delay*1000,
        "raw_position_error_mm": raw_mm,
        "compensated_position_error_mm": compensated_mm,
        "jerk_rms": jerk,
        "grip_instability": grip_instability,
        "load_cv": load_cv,
        "finger_velocity_rms": finger_velocity,
        "placement_error_mm": placement_error,
        "object_slip_mm": object_slip,
        "object_retained": object_retained,
        "task_success": bool(task_success),
        "data_completeness": completeness,
        "max_gap_ms": max_gap_s*1000,
        "data_valid": data_valid,
        "ctqi_eligible": data_valid,
        "ctqi": ctqi,
        **components
    }, desired, observed, compensated, command, hand_q, hand_dq, hand_load


def main():
    trials = []
    phases = []

    for condition in CONDITIONS:
        severities = (
            np.zeros(10) if condition == "baseline"
            else np.linspace(0.15, 0.95, 10)
        )

        for severity in severities:
            trial_id = len(trials)+1

            row, desired, actual, compensated, cmd, hq, hdq, load = (
                run_trial(condition, float(severity), trial_id)
            )

            trials.append(row)

            for phase, (start, end) in PHASES.items():
                mask = (T >= start) & (T < end)

                error = np.mean(np.linalg.norm(
                    desired[mask]-compensated[mask], axis=1
                ))

                qp = np.exp(-error/0.020)
                qs = np.exp(
                    -calculate_jerk(actual[mask])/80
                )
                ql = row["q_latency"]
                qg = np.nan

                if phase == "grip":
                    qg = np.exp(
                        -np.mean(np.abs(cmd[mask]-hq[mask]))/0.12
                    )
                elif phase == "hold":
                    hload = load[mask]
                    cv = np.std(hload)/(np.mean(hload)+1e-6)
                    vel = np.sqrt(np.mean(hdq[mask]**2))
                    qg = 1 - (
                        0.65*np.clip(cv/0.50, 0, 1) +
                        0.35*np.clip(vel/0.20, 0, 1)
                    )
                elif phase == "release":
                    residual = np.mean(load[mask][-10:])
                    qg = np.clip(1-residual/0.35, 0, 1)

                scores = [ql, qp, qs]
                if np.isfinite(qg):
                    scores.append(qg)

                phases.append({
                    "trial_id": trial_id,
                    "condition": condition,
                    "phase": phase,
                    "position_error_mm": error*1000,
                    "q_latency": ql,
                    "q_position": qp,
                    "q_smoothness": qs,
                    "q_grip": qg,
                    "phase_ctqi": 100*np.mean(scores)
                })

    df = pd.DataFrame(trials)
    phase_df = pd.DataFrame(phases)

    df.to_csv(OUT/"trial_summary.csv", index=False)
    phase_df.to_csv(OUT/"phase_metrics.csv", index=False)

    # Component ablation: descriptive pilot analysis.
    ablation = [{
        "model": "full_CTQI",
        "auc": auc(df.task_success, df.ctqi)
    }]

    for component in COMPONENTS:
        remaining = [c for c in COMPONENTS if c != component]
        score = 100*df[remaining].mean(axis=1)

        ablation.append({
            "model": "without_"+component,
            "auc": auc(df.task_success, score)
        })

    ablation_df = pd.DataFrame(ablation)
    ablation_df.to_csv(OUT/"ablation_auc.csv", index=False)

    corr = df[COMPONENTS].corr()
    corr.to_csv(OUT/"component_correlation.csv")

    # PCA diagnostic using SVD.
    x = df[COMPONENTS].to_numpy()
    x = (x-x.mean(axis=0))/(x.std(axis=0)+1e-9)
    singular = np.linalg.svd(x, compute_uv=False)
    variance = singular**2 / np.sum(singular**2)

    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(variance))],
        "explained_variance": variance,
        "cumulative_variance": np.cumsum(variance)
    }).to_csv(OUT/"pca_explained_variance.csv", index=False)

    # Figure 1: CTQI against independent task outcomes.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([
        df.loc[df.task_success, "ctqi"],
        df.loc[~df.task_success, "ctqi"]
    ], tick_labels=["Successful", "Unsuccessful"])
    ax.set_ylabel("CTQI (provisional)")
    ax.set_title("CTQI versus independent task outcome")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/"ctqi_task_outcome.png", dpi=180)
    plt.close(fig)

    # Figure 2: experimental conditions.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([
        df.loc[df.condition == c, "ctqi"] for c in CONDITIONS
    ], tick_labels=CONDITIONS)
    ax.set_ylabel("CTQI")
    ax.set_title("CTQI under controlled degradation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/"ctqi_conditions.png", dpi=180)
    plt.close(fig)

    # Figure 3: phase heatmap.
    pivot = phase_df.groupby(
        ["condition", "phase"]
    )["phase_ctqi"].mean().unstack()

    pivot = pivot.reindex(
        index=CONDITIONS, columns=list(PHASES)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(5), list(PHASES))
    ax.set_yticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_title("Phase-level CTQI")
    fig.colorbar(im, ax=ax, label="CTQI")
    fig.tight_layout()
    fig.savefig(OUT/"phase_ctqi_heatmap.png", dpi=180)
    plt.close(fig)

    # Figure 4: ablation.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ablation_df.model, ablation_df.auc)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exploratory ROC AUC")
    ax.set_title("CTQI component ablation")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/"ctqi_ablation.png", dpi=180)
    plt.close(fig)

    # Figure 5: independent placement error.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df.placement_error_mm, df.ctqi)
    ax.set_xlabel("Independent placement error (mm)")
    ax.set_ylabel("CTQI")
    ax.set_title("CTQI versus placement error")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT/"ctqi_placement_error.png", dpi=180)
    plt.close(fig)

    # Figure 6: component correlation.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(4), COMPONENTS, rotation=30, ha="right")
    ax.set_yticks(range(4), COMPONENTS)
    ax.set_title("CTQI component correlation")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUT/"component_correlation.png", dpi=180)
    plt.close(fig)

    summary = {
        "experiment": "CTQI Pilot 001",
        "synthetic_only": True,
        "trials": len(df),
        "conditions": CONDITIONS,
        "phases": list(PHASES),
        "revo2_schema": "2 hands x 6 generic channels",
        "hand_signals": [
            "position_norm", "velocity_norm", "synthetic_load_proxy"
        ],
        "provisional_weights": WEIGHTS,
        "task_success_auc": auc(df.task_success, df.ctqi),
        "task_success_rate": float(df.task_success.mean()),
        "data_valid_rate": float(df.data_valid.mean()),
        "methodology": (
            "Task success is an independent outcome, not a CTQI validity gate. "
            "CTQI is calculated for failed trials as well as successful trials. "
            "Data validity and learning eligibility are reported separately."
        ),
        "limitations": [
            "Synthetic data, not measured G1 or Revo2 performance.",
            "Generic channel indices; actual motor mapping not yet connected.",
            "Equal weights and normalisation scales are provisional.",
            "Exploratory AUC and ablation require held-out validation."
        ]
    }

    (OUT/"experiment_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n========== CTQI PILOT 001 ==========")
    print("Trials:", len(df))
    print("Task-success AUC:", round(summary["task_success_auc"], 3))
    print("Task success:", round(summary["task_success_rate"], 3))
    print("Data-valid rate:", round(summary["data_valid_rate"], 3))
    print("Output:", OUT)
    print("SYNTHETIC PILOT ONLY - NOT MEASURED ROBOT RESULTS")


if __name__ == "__main__":
    main()
