
"""Analyse RAVEN tripod-camera framing without loading Isaac Sim.

Reads the existing tripod_framing.csv. Never modifies that CSV or any controller.
All measurements come from the CSV, not the surrogate's privileged pose.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def pick(headers, override, aliases, fragments=()):
    if override:
        for col in headers:
            if key(col) == key(override):
                return col
        raise ValueError(f"Column '{override}' not found. Available: {headers}")
    indexed = {key(col): col for col in headers}
    for candidate in aliases:
        if key(candidate) in indexed:
            return indexed[key(candidate)]
    for col in headers:
        clean = key(col)
        if any(s in clean for s in ("mean", "mae", "p95", "rmse", "max", "absolute", "abs")):
            continue
        if all(any(piece in clean for piece in alternatives) for alternatives in fragments):
            return col
    return None


def number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def percentile(sorted_values, percent):
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * percent / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def statistics(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    if not values:
        return None
    absolute = sorted(abs(x) for x in values)
    return {
        "n": len(values),
        "signed_mean_deg": sum(values) / len(values),
        "mae_deg": sum(absolute) / len(absolute),
        "rmse_deg": math.sqrt(sum(x * x for x in values) / len(values)),
        "p95_abs_deg": percentile(absolute, 95),
        "max_abs_deg": absolute[-1],
    }


def fmt(stats):
    if stats is None:
        return "no valid samples"
    return (
        f"n={stats['n']:3d}  signed={stats['signed_mean_deg']:+.4f}°  "
        f"MAE={stats['mae_deg']:.4f}°  RMSE={stats['rmse_deg']:.4f}°  "
        f"P95={stats['p95_abs_deg']:.4f}°  max={stats['max_abs_deg']:.4f}°"
    )


def save_plot(data, out, stationary_end, pan_key, target_key):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[INFO] matplotlib is unavailable; JSON and binned CSV were still written.")
        return

    times = [r["t"] for r in data if r["h"] is not None]
    errors = [r["h"] for r in data if r["h"] is not None]
    plt.figure(figsize=(11, 4.5))
    plt.plot(times, errors, linewidth=1.4, label="Signed horizontal framing error")
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(stationary_end, color="black", linestyle="--", alpha=0.6,
                label="Initial stationary-window end")
    plt.xlabel("Logged time (s)")
    plt.ylabel("Horizontal framing error (degrees)")
    plt.title("RAVEN: Cinematography framing error")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out / "horizontal_error_vs_time.png", dpi=160)
    plt.close()

    if pan_key and target_key:
        valid = [r for r in data if r["pan"] is not None and r["target"] is not None]
        if valid:
            plt.figure(figsize=(11, 4.5))
            plt.plot([r["t"] for r in valid], [r["target"] for r in valid],
                     label=f"{target_key}")
            plt.plot([r["t"] for r in valid], [r["pan"] for r in valid],
                     label=f"{pan_key}")
            plt.xlabel("Logged time (s)")
            plt.ylabel("Angle (degrees)")
            plt.title("RAVEN: Requested versus physical tripod pan")
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
            plt.savefig(out / "pan_response_vs_time.png", dpi=160)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--time", help="Override logged time column name")
    parser.add_argument("--horizontal-error", help="Override SIGNED horizontal error column")
    parser.add_argument("--vertical-error", help="Override signed vertical error column")
    parser.add_argument("--pan", help="Override measured tripod-pan column")
    parser.add_argument("--target", help="Override commanded pan / target-bearing column")
    parser.add_argument("--stationary-start", type=float, default=0.0)
    parser.add_argument("--stationary-end", type=float, default=1.0)
    parser.add_argument("--bin-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.bin_seconds <= 0 or args.stationary_end <= args.stationary_start:
        parser.error("Provide a positive --bin-seconds and valid stationary window.")
    if not args.csv.is_file():
        parser.error(f"CSV not found: {args.csv}")

    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        raw_rows = list(reader)

    time_col = pick(headers, args.time,
                    ["time_s", "t_s", "elapsed_s", "elapsed_time_s", "sim_time_s",
                     "tracking_time_s", "time_sec", "sim_time", "time", "t"],
                    [("time", "elapsed", "timestamp")])
    h_col = pick(headers, args.horizontal_error,
                 ["horizontal_error_deg", "horizontal_framing_error_deg", "horizontal_signed_error_deg",
                  "signed_horizontal_error_deg", "framing_error_horizontal_deg", "h_error_deg",
                  "horizontal_angle_error_deg", "framing_error_deg", "yaw_error_deg"],
                 [("horizontal", "horiz"), ("error", "err")])
    v_col = pick(headers, args.vertical_error,
                 ["vertical_error_deg", "vertical_framing_error_deg", "vertical_signed_error_deg",
                  "signed_vertical_error_deg", "framing_error_vertical_deg", "v_error_deg",
                  "pitch_error_deg"],
                 [("vertical", "vert"), ("error", "err")])
    pan_col = pick(headers, args.pan,
                   ["tripod_pan_deg", "actual_pan_deg", "measured_pan_deg", "pan_deg",
                    "physical_pan_deg", "pan_angle_deg"],
                   [("tripod", "actual", "measured", "physical"), ("pan",)])
    target_col = pick(headers, args.target,
                      ["target_pan_deg", "pan_target_deg", "requested_pan_deg",
                       "desired_pan_deg", "commanded_pan_deg", "command_pan_deg",
                       "target_bearing_deg", "actor_bearing_deg", "target_angle_deg"],
                      [("target", "desired", "command", "requested"), ("pan", "bearing", "angle")])

    print("\n[RAVEN] Original CSV columns:")
    for name in headers:
        print(f"  {name}")
    print("\n[RAVEN] Detected measurement columns:")
    for label, col in (("time", time_col), ("signed horizontal error", h_col),
                       ("signed vertical error", v_col), ("measured tripod pan", pan_col),
                       ("target / command", target_col)):
        print(f"  {label:25} {col or 'NOT FOUND'}")
    if time_col is None or h_col is None:
        parser.error("Time and signed horizontal-error columns are needed. "
                     "Check the printed headers and rerun with --time and/or --horizontal-error.")
    if "abs" in key(h_col):
        parser.error("Horizontal error must be signed, not an absolute-error column.")

    data = []
    for raw in raw_rows:
        t = number(raw.get(time_col))
        if t is not None:
            data.append({"t": t, "h": number(raw.get(h_col)),
                         "v": number(raw.get(v_col)) if v_col else None,
                         "pan": number(raw.get(pan_col)) if pan_col else None,
                         "target": number(raw.get(target_col)) if target_col else None})
    data.sort(key=lambda r: r["t"])
    valid = [r for r in data if r["h"] is not None]
    if not valid:
        parser.error("No finite signed horizontal framing errors were found.")

    stationary = [r for r in valid if args.stationary_start <= r["t"] <= args.stationary_end]
    remainder = [r for r in valid if r["t"] > args.stationary_end]
    base = statistics([r["h"] for r in stationary])
    all_stats = statistics([r["h"] for r in valid])
    after_stats = statistics([r["h"] for r in remainder])
    vertical_stats = statistics([r["v"] for r in data])
    bias = base["signed_mean_deg"] if base else None
    corrected = statistics([r["h"] - bias for r in remainder]) if bias is not None else None

    print(f"\n[RAVEN] Read {len(raw_rows)} raw rows; {len(valid)} finite horizontal-error samples")
    print("  Full run:                  " + fmt(all_stats))
    print(f"  Stationary [{args.stationary_start:g}, {args.stationary_end:g}] s:   " + fmt(base))
    print("  Later samples (NOT assumed all moving): " + fmt(after_stats))
    print("  Initial-bias-subtracted later samples: " + fmt(corrected))
    if vertical_stats:
        print("  Vertical full run:         " + fmt(vertical_stats))
    if base is None:
        print("[WARNING] No samples in the specified stationary interval. Check logged time range.")

    bins = defaultdict(list)
    for r in valid:
        slot = math.floor(r["t"] / args.bin_seconds) * args.bin_seconds
        bins[slot].append(r["h"])
    print(f"\n[RAVEN] Horizontal error by {args.bin_seconds:g}-second intervals:")
    bin_rows = []
    for start in sorted(bins):
        row = {"start_s": start, "end_s": start + args.bin_seconds,
               **statistics(bins[start])}
        bin_rows.append(row)
        print(f"  {start:6.1f} to {start + args.bin_seconds:6.1f} s: " + fmt(row))

    output = args.output or (Path.cwd() / "raven_tracking_results" / "framing_diagnostics" /
                             datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "source_csv": str(args.csv),
        "columns": {"time": time_col, "horizontal_error": h_col,
                    "vertical_error": v_col, "measured_pan": pan_col, "target": target_col},
        "stationary_interval_s": [args.stationary_start, args.stationary_end],
        "full_horizontal": all_stats,
        "initial_stationary_horizontal": base,
        "later_horizontal": after_stats,
        "later_horizontal_after_initial_bias_subtraction": corrected,
        "full_vertical": vertical_stats,
        "cautions": [
            "Later samples are not automatically classified as moving actor samples.",
            "Removing the initial stationary signed mean does not calibrate position-dependent errors.",
            "Framing-only measurements cannot establish end-to-end perception-to-actuation latency.",
        ],
    }
    (output / "framing_diagnostics.summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    with (output / "framing_diagnostics.bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["start_s", "end_s", "n", "signed_mean_deg",
                                               "mae_deg", "rmse_deg", "p95_abs_deg", "max_abs_deg"])
        writer.writeheader()
        writer.writerows(bin_rows)
    save_plot(data, output, args.stationary_end, pan_col, target_col)
    print(f"\n[DONE] Diagnostics saved to: {output}")
    print("[NEXT] Compare sustained directional motion and reversals using the error plot.")
    print("[NEXT] Run separate stationary lateral-position trials to test parallax/geometric bias.")


if __name__ == "__main__":
    main()
