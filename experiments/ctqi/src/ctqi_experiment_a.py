"""CTQI Experiment A: physics-based construct-validity pilot.

Uses the stable Stage 0 G1/Revo2 model.
Lower body and Revo2 fingers remain constrained.
This experiment evaluates temporal, spatial and motion quality;
not manipulation success, grip stability or learning utility.
"""

import argparse
import heapq
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd

from ctqi_stage0 import (
    HAND_OPEN_POSE, LOCK_JOINTS, STAGE0, build,
    apply_open_hand_pose, controlled_joints
)

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = (
    "baseline", "latency", "jitter", "spatial",
    "oscillation", "dropout", "combined"
)

DURATION = 8.0
DT = 0.02

PHASES = (
    ("rest", 0, 1),
    ("reach", 1, 2.8),
    ("align", 2.8, 4),
    ("dwell", 4, 5.5),
    ("return", 5.5, 7.5),
    ("settle", 7.5, 8)
)


def phase(t):
    for name, start, end in PHASES:
        if start <= t < end:
            return name
    return "settle"


def smooth(x):
    x = np.clip(x, 0, 1)
    return x**3 * (10 - 15*x + 6*x*x)


def reference(t, nominal):
    if t < 1:
        r, a = 0., 0.
    elif t < 2.8:
        r, a = smooth((t-1)/1.8), 0.
    elif t < 4:
        r, a = 1., smooth((t-2.8)/1.2)
    elif t < 5.5:
        r, a = 1., 1.
    elif t < 7.5:
        r = a = 1. - smooth((t-5.5)/2)
    else:
        r, a = 0., 0.

    result = nominal.copy()

    for side, sign in (("left", 1), ("right", -1)):
        result[f"{side}_shoulder_pitch_joint"] += .13*r
        result[f"{side}_shoulder_roll_joint"] += sign*(.07*r+.025*a)
        result[f"{side}_elbow_joint"] += .15*r+.04*a
        result[f"{side}_wrist_pitch_joint"] += .04*a

    return result


def setup():
    build()
    model = mujoco.MjModel.from_xml_path(str(STAGE0))

    for joint in LOCK_JOINTS:
        name = "stage0_lock_" + joint
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        ) < 0:
            raise RuntimeError("Missing stance constraint: " + name)

    for joint in HAND_OPEN_POSE:
        name = "stage0_open_" + joint
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        ) < 0:
            raise RuntimeError("Missing Revo2 constraint: " + name)

    arms = [
        e for e in controlled_joints(model)
        if not e["hand"]
    ]

    if len(arms) != 14:
        raise RuntimeError("Expected 14 arm actuators")

    return model, arms


def create_references(model, arms, times):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    apply_open_hand_pose(model, data)

    nominal = {
        e["name"]: float(model.qpos0[e["q"]])
        for e in arms
    }

    wrist_ids = [
        model.body(side+"_wrist_yaw_link").id
        for side in ("left", "right")
    ]

    targets, wrists = [], []

    for t in times:
        desired = reference(t, nominal)
        command = np.array([
            desired[e["name"]] for e in arms
        ])

        for e, value in zip(arms, command):
            data.qpos[e["q"]] = value

        mujoco.mj_forward(model, data)

        targets.append(command)
        wrists.append(np.concatenate([
            data.xpos[i].copy() for i in wrist_ids
        ]))

    return np.array(targets), np.array(wrists), wrist_ids


def perturb(cmd, names, t, severity, condition):
    result = cmd.copy()

    if condition in ("spatial", "combined"):
        for i, name in enumerate(names):
            if "shoulder_pitch" in name:
                result[i] += .17*severity
            elif "elbow" in name:
                result[i] -= .10*severity

    if condition in ("oscillation", "combined"):
        signal = .11*severity*math.sin(2*math.pi*3.5*t)

        for i, name in enumerate(names):
            if "shoulder_pitch" in name:
                result[i] += signal
            elif "elbow" in name:
                result[i] -= .7*signal

    return result


def jerk(position):
    kernel = np.array([1, 4, 6, 4, 1])/16

    filtered = np.column_stack([
        np.convolve(
            np.pad(position[:, j], 2, mode="reflect"),
            kernel, "valid"
        )
        for j in range(position.shape[1])
    ])

    for _ in range(3):
        filtered = np.gradient(filtered, DT, axis=0)

    return filtered


def evaluate(times, wanted, observed, available, delays):
    completeness = float(available.mean())

    consecutive = worst = 0
    for ok in available:
        consecutive = 0 if ok else consecutive+1
        worst = max(worst, consecutive)

    valid_data = completeness >= .95 and worst*DT <= .10

    filled = np.column_stack([
        np.interp(
            times, times[available],
            observed[available, j]
        )
        for j in range(observed.shape[1])
    ])

    moving = (
        ((times >= 1.1) & (times <= 3.9)) |
        ((times >= 5.6) & (times <= 7.4))
    )

    best_lag, best_cost = 0., np.inf

    for lag in np.arange(0, .451, DT):
        selected = moving & (times+lag <= times[-1])

        delayed = np.column_stack([
            np.interp(
                times[selected]+lag,
                times, filled[:, j]
            )
            for j in range(filled.shape[1])
        ])

        cost = np.mean(np.sum(
            (wanted[selected]-delayed)**2, axis=1
        ))

        if cost < best_cost:
            best_lag, best_cost = float(lag), float(cost)

    aligned_times = np.minimum(times+best_lag, times[-1])

    aligned = np.column_stack([
        np.interp(aligned_times, times, filled[:, j])
        for j in range(filled.shape[1])
    ])

    aligned_available = (
        np.interp(
            aligned_times, times, available.astype(float)
        ) > .999
    )

    paired = (
        available & aligned_available &
        (times+best_lag <= times[-1])
    )

    spatial = float(np.mean(np.linalg.norm(
        wanted[paired]-aligned[paired], axis=1
    )))

    raw_error = float(np.mean(np.linalg.norm(
        wanted[available]-observed[available], axis=1
    )))

    motion_valid = moving & paired

    for i in np.flatnonzero(~paired):
        motion_valid[
            max(0, i-3):min(len(times), i+4)
        ] = False

    motion_valid[:5] = False
    motion_valid[-5:] = False

    excess = (
        float(np.sqrt(np.mean(np.sum(
            (jerk(aligned)[motion_valid] -
             jerk(wanted)[motion_valid])**2,
            axis=1
        ))))
        if motion_valid.sum() >= 15 else np.nan
    )

    delays = np.asarray(delays)
    mean_delay = float(delays.mean())
    jitter = float(delays.std())

    # Provisional scales; require scientific justification.
    q_temporal = float(np.exp(-(mean_delay+2*jitter)/.15))
    q_spatial = float(np.exp(-spatial/.025))
    q_motion = float(np.exp(-excess/5))

    diagnostic = float(
        100*np.mean([q_temporal, q_spatial, q_motion])
    )

    if not np.isfinite(diagnostic):
        valid_data = False

    return {
        "data_valid": bool(valid_data),
        "ctqi": diagnostic if valid_data else np.nan,
        "diagnostic_ctqi": diagnostic,
        "q_temporal": q_temporal,
        "q_spatial": q_spatial,
        "q_motion": q_motion,
        "completeness": completeness,
        "max_gap_ms": 1000*worst*DT,
        "transport_latency_ms": 1000*mean_delay,
        "transport_jitter_ms": 1000*jitter,
        "estimated_total_response_lag_ms": 1000*best_lag,
        "raw_wrist_error_mm": 1000*raw_error,
        "aligned_wrist_error_mm": 1000*spatial,
        "excess_wrist_jerk_m_s3": excess
    }


def run_trial(
    model, arms, targets, desired, wrist_ids, times,
    condition, severity, seed, telemetry_dir
):
    rng = np.random.default_rng(seed)

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0

    apply_open_hand_pose(model, data)
    mujoco.mj_forward(model, data)

    physics_dt = float(model.opt.timestep)
    stride = round(DT/physics_dt)

    if abs(stride*physics_dt-DT) > 1e-8:
        raise RuntimeError("Physics timestep incompatible with 50Hz")

    available = np.ones(len(times), dtype=bool)

    if condition in ("dropout", "combined"):
        available = rng.random(len(times)) > .28*severity

        if severity >= .5:
            start = int(rng.integers(120, 250))
            length = math.ceil(.16*severity/DT)
            available[start:start+length] = False

    queue = []
    sequence = 0
    last_sequence = -1
    current = targets[0].copy()
    last_timestamp = 0.

    names = [e["name"] for e in arms]

    base_delay = .03 + rng.uniform(-.004, .004)

    delay = (
        base_delay +
        (.17*severity if condition in ("latency", "combined") else 0)
    )

    jitter_sd = (
        .055*severity if condition in ("jitter", "combined") else 0
    )

    delays, rows = [], []

    for step in range(len(times)*stride):
        t = float(data.time)

        if step % stride == 0:
            idx = step//stride

            command = perturb(
                targets[idx], names, t, severity, condition
            )

            command += rng.normal(
                0, .0015, size=len(command)
            )

            arrival = t + max(
                0, delay+rng.normal(0, jitter_sd)
            )

            heapq.heappush(
                queue, (arrival, sequence, t, command)
            )
            sequence += 1

        while queue and queue[0][0] <= t+1e-9:
            _, number, timestamp, command = heapq.heappop(queue)

            if number <= last_sequence:
                continue

            current = command
            last_sequence = number
            last_timestamp = timestamp
            delays.append(t-timestamp)

        # Gravity-compensated arm control.
        # Finger joints remain constrained in Stage 0.
        for j, entry in enumerate(arms):
            error = current[j] - data.qpos[entry["q"]]

            torque = (
                data.qfrc_bias[entry["d"]]
                + entry["kp"]*error
                - entry["kd"]*data.qvel[entry["d"]]
            )

            data.ctrl[entry["aid"]] = np.clip(
                torque/entry["gear"],
                entry["lo"], entry["hi"]
            )

        mujoco.mj_step(model, data)

        if not (
            np.isfinite(data.qpos).all() and
            np.isfinite(data.qvel).all()
        ):
            raise RuntimeError(
                f"Unstable simulation: {condition}, seed {seed}"
            )

        if step % stride == 0:
            wrist = np.concatenate([
                data.xpos[i].copy() for i in wrist_ids
            ])

            row = {
                "time_s": float(data.time),
                "phase": phase(times[idx]),
                "available": int(available[idx]),
                "command_source_time_s": last_timestamp,
                "received_command_age_ms":
                    1000*(data.time-last_timestamp),
                "contacts": int(data.ncon),
            }

            for j, key in enumerate(
                ("lx", "ly", "lz", "rx", "ry", "rz")
            ):
                row[f"desired_wrist_{key}_m"] = desired[idx, j]
                row[f"actual_wrist_{key}_m"] = (
                    wrist[j] if available[idx] else np.nan
                )

            for j, entry in enumerate(arms):
                name = entry["name"]

                row[f"{name}_reference_rad"] = targets[idx, j]
                row[f"{name}_cmd_rad"] = current[j]
                row[f"{name}_actual_rad"] = (
                    float(data.qpos[entry["q"]])
                    if available[idx] else np.nan
                )

            for side in ("left", "right"):
                joint = model.joint(
                    f"{side}_index_proximal_joint"
                ).id

                row[f"{side}_revo2_index_rad"] = (
                    float(data.qpos[model.jnt_qposadr[joint]])
                    if available[idx] else np.nan
                )

            for obj in (
                "teleop_cube", "teleop_cylinder", "teleop_handle"
            ):
                xyz = data.xpos[model.body(obj).id]

                for j, axis in enumerate("xyz"):
                    row[f"{obj}_{axis}_m"] = (
                        float(xyz[j]) if available[idx] else np.nan
                    )

            rows.append(row)

    filename = f"{condition}_seed{seed:08d}.csv"
    pd.DataFrame(rows).to_csv(
        telemetry_dir/filename, index=False
    )

    measured = np.column_stack([
        np.array([
            r[f"actual_wrist_{axis}_m"] for r in rows
        ])
        for axis in ("lx", "ly", "lz", "rx", "ry", "rz")
    ])

    return {
        "condition": condition,
        "severity": severity,
        "seed": seed,
        "telemetry_file": filename,
        **evaluate(times, desired, measured, available, delays)
    }


def export_analysis(df, out):
    components = ["q_temporal", "q_spatial", "q_motion"]

    metrics = components + [
        "completeness", "diagnostic_ctqi",
        "raw_wrist_error_mm", "aligned_wrist_error_mm",
        "transport_latency_ms", "transport_jitter_ms",
        "excess_wrist_jerk_m_s3"
    ]

    df.groupby("condition")[metrics].agg(
        ["mean", "std"]
    ).to_csv(out/"condition_summary.csv")

    correlations = df[components].corr()
    correlations.to_csv(out/"component_correlations.csv")

    means = df.groupby("condition")[components].mean()
    (means-means.loc["baseline"]).to_csv(
        out/"construct_response.csv"
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 4), sharey=True
    )

    for ax, metric in zip(axes, components):
        ax.bar(CONDITIONS, [
            means.loc[c, metric] for c in CONDITIONS
        ])
        ax.set_title(metric)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=.2)

    fig.tight_layout()
    fig.savefig(
        out/"construct_components.png", dpi=170
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))

    for condition in CONDITIONS[1:]:
        sub = df[df.condition == condition]
        ax.plot(
            sub.severity, sub.diagnostic_ctqi,
            marker="o", label=condition
        )

    ax.set_xlabel("Controlled disturbance severity")
    ax.set_ylabel("Ungated diagnostic CTQI (0–100)")
    ax.set_title("Exploratory dose response")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/"dose_response.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(CONDITIONS, [
        df.loc[
            df.condition == c, "completeness"
        ].mean()
        for c in CONDITIONS
    ])
    ax.axhline(
        .95, linestyle="--",
        label="Provisional validity threshold"
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Observed data fraction")
    ax.set_title("Data integrity")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out/"data_integrity.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(
        correlations.to_numpy(),
        vmin=-1, vmax=1, cmap="coolwarm"
    )
    ax.set_xticks(
        range(3), components, rotation=30
    )
    ax.set_yticks(range(3), components)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(
        out/"component_correlation.png", dpi=170
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials-per-condition", type=int, default=3
    )
    parser.add_argument(
        "--seed", type=int, default=20261030
    )

    args = parser.parse_args()

    if args.trials_per_condition < 2:
        parser.error("Need at least two trials per condition")

    out = ROOT/"results/experiment_a"
    telemetry = out/"telemetry"
    telemetry.mkdir(parents=True, exist_ok=True)

    model, arms = setup()

    times = np.arange(
        round(DURATION/DT)
    )*DT

    targets, desired, wrist_ids = create_references(
        model, arms, times
    )

    results = []

    for ci, condition in enumerate(CONDITIONS):
        for i in range(args.trials_per_condition):

            severity = (
                0. if condition == "baseline"
                else float(np.linspace(
                    .2, .95, args.trials_per_condition
                )[i])
            )

            seed = args.seed + 1000*ci + i

            print(
                f"[{len(results)+1}/"
                f"{len(CONDITIONS)*args.trials_per_condition}] "
                f"{condition}, severity={severity:.2f}",
                flush=True
            )

            result = run_trial(
                model, arms, targets, desired, wrist_ids,
                times, condition, severity, seed, telemetry
            )

            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(out/"trial_metrics.csv", index=False)

    export_analysis(df, out)

    config = {
        "experiment": "CTQI Experiment A",
        "type": "Physics-driven construct-validity pilot",
        "trials": len(df),
        "seed": args.seed,
        "duration_s": DURATION,
        "fixed_stance": True,
        "revo2_fingers_locked": True,
        "provisional_normalisations": {
            "temporal_s": .15,
            "spatial_m": .025,
            "excess_jerk_m_s3": 5
        },
        "provisional_weights": {
            "q_temporal": 1/3,
            "q_spatial": 1/3,
            "q_motion": 1/3
        },
        "limitations": (
            "No grip evaluation, independent manipulation "
            "outcome validation, or downstream learning claim."
        )
    }

    (out/"experiment_config.json").write_text(
        json.dumps(config, indent=2)
    )

    print("\n========== EXPERIMENT A COMPLETE ==========")

    print(
        df.groupby("condition")[
            [
                "q_temporal", "q_spatial", "q_motion",
                "completeness", "diagnostic_ctqi"
            ]
        ].mean().round(3).to_string()
    )

    print(
        "CTQI-eligible trials:",
        int(df.data_valid.sum()),
        "/", len(df)
    )
    print("Results:", out)


if __name__ == "__main__":
    main()
