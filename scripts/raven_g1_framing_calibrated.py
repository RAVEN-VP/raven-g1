
"""RAVEN: independent physical optical-axis calibration experiment.

Depends on the two EXISTING working scripts in the same scripts/ directory:
  raven_g1_actor_tracking.py
  raven_g1_framing_verification.py

Does not edit either script. Runs their normal robot/grasp/pan controller and
tripod RGB framing recorder with a newly rotated tripod camera mount.

The values below come from the initial 1-second STATIONARY framing window in
an independent baseline measurement. They are installation-calibration
estimates, not a claim of improved performance until this run is recorded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

# Parse this experiment's options before the existing Isaac Lab
# launcher processes the remaining command-line arguments.
calibration_parser = argparse.ArgumentParser(add_help=False)
calibration_parser.add_argument(
    "--mount-yaw-deg",
    type=float,
    default=3.44,
)
calibration_parser.add_argument(
    "--mount-pitch-up-deg",
    type=float,
    default=2.46,
)
calibration_options, passthrough = (
    calibration_parser.parse_known_args()
)
sys.argv = [sys.argv[0], *passthrough]

# Import starts Isaac Sim through the proven baseline launcher.
# Do not create a second AppLauncher.
import raven_g1_framing_verification as framing

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

# These are the working tripod optical-origin settings from the
# previous successful camera test.
MOUNT_POSITION_M = (0.38, 0.0, 0.03)

WIDTH = 640
HEIGHT = 360


def mount_quaternion(yaw_deg: float, pitch_up_deg: float):
    """Return a WXYZ local mount rotation.

    Camera coordinate convention:
      +X = forward
      +Y = left
      +Z = up

    Positive yaw turns the camera left.
    Positive pitch_up raises the optical axis.
    """
    yaw = math.radians(yaw_deg)
    pitch_about_y = -math.radians(pitch_up_deg)

    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)
    cp = math.cos(pitch_about_y / 2)
    sp = math.sin(pitch_about_y / 2)

    # q = q_yaw_about_Z * q_pitch_about_Y
    return (
        cy * cp,
        -sy * sp,
        cy * sp,
        sy * cp,
    )


YAW_DEG = calibration_options.mount_yaw_deg
PITCH_UP_DEG = calibration_options.mount_pitch_up_deg

if not (
    math.isfinite(YAW_DEG)
    and math.isfinite(PITCH_UP_DEG)
):
    raise ValueError(
        "Mount yaw and pitch must be finite numbers"
    )

if abs(YAW_DEG) > 12.0 or abs(PITCH_UP_DEG) > 10.0:
    raise ValueError(
        "Calibration is limited to +/-12 deg yaw "
        "and +/-10 deg pitch"
    )

MOUNT_QUAT = mount_quaternion(
    YAW_DEG,
    PITCH_UP_DEG,
)


@configclass
class CalibratedFramingScene(framing.SceneWithFramingCamera):
    """Preserve the scene and replace only the tripod RGB sensor."""

    tripod_rgb = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Tripod/camera_link/"
            "raven_framing_rgb"
        ),
        update_period=0.1,
        width=WIDTH,
        height=HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 20.0),
            focus_distance=3.0,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=MOUNT_POSITION_M,
            rot=MOUNT_QUAT,
            convention="world",
        ),
    )


# Never overwrite the original baseline experiment.
baseline_output_dir = Path(
    framing.baseline.args.output_dir
).resolve()

run_tag = (
    f"camera_yaw_{YAW_DEG:+.2f}_pitch_{PITCH_UP_DEG:+.2f}"
    .replace("+", "p")
    .replace("-", "m")
    .replace(".", "p")
)

calibrated_output_dir = baseline_output_dir / run_tag

framing.baseline.args.output_dir = str(
    calibrated_output_dir
)

original_scene_class = framing.SceneWithFramingCamera
original_recorder_class = framing.FramingRecorder
original_baseline_scene_class = framing.baseline.SceneCfg


class CalibrationRecorder(original_recorder_class):
    """Extend the verified image recorder with calibration metadata."""

    def close(self):
        # Write the real tripod RGB measurements using the existing
        # recorder. No post-hoc bias correction is applied.
        super().close()

        if not (
            self.csv_path.exists()
            and self.summary_path.exists()
        ):
            return

        summary = json.loads(
            self.summary_path.read_text(
                encoding="utf-8"
            )
        )

        summary["mount_calibration"] = {
            "applied_yaw_left_deg": YAW_DEG,
            "applied_pitch_up_deg": PITCH_UP_DEG,
            "mount_position_local_m": MOUNT_POSITION_M,
            "mount_quaternion_wxyz": MOUNT_QUAT,
            "method": (
                "Physical rotation of tripod RGB sensor "
                "relative to camera_link"
            ),
            "actor_ground_truth_used_for_control": False,
        }

        with self.csv_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as stream:
            measurements = list(
                csv.DictReader(stream)
            )

        # Evaluate actual frames in the first stationary second.
        static = []

        for row in measurements:
            try:
                t = float(row["tracking_time_s"])
            except (TypeError, ValueError):
                continue

            if t > 1.0:
                continue

            if (
                row["tripod_detected"]
                .strip()
                .lower()
                not in ("true", "1")
            ):
                continue

            try:
                x = float(row["horizontal_error_deg"])
                y = float(row["vertical_error_deg"])
                pan = float(row["actual_pan_deg"])
            except (KeyError, TypeError, ValueError):
                continue

            if all(
                math.isfinite(value)
                for value in (x, y, pan)
            ):
                static.append((x, y, pan))

        if static:
            summary["stationary_window"] = {
                "window_s": [0.0, 1.0],
                "samples": len(static),
                "mean_signed_horizontal_error_deg": (
                    statistics.mean(
                        x for x, _, _ in static
                    )
                ),
                "mean_signed_vertical_error_deg": (
                    statistics.mean(
                        y for _, y, _ in static
                    )
                ),
                "pan_range_deg": (
                    max(p for _, _, p in static)
                    - min(p for _, _, p in static)
                ),
            }

        # Compare independent actual camera recordings.
        baseline_summary_path = (
            baseline_output_dir
            / "tripod_framing"
            / "tripod_framing.summary.json"
        )

        if baseline_summary_path.is_file():
            previous = json.loads(
                baseline_summary_path.read_text(
                    encoding="utf-8"
                )
            )

            summary["prior_run_comparison"] = {
                "original_summary": str(
                    baseline_summary_path
                ),
                "original_horizontal_mae_deg": (
                    previous.get(
                        "horizontal_mean_abs_error_deg"
                    )
                ),
                "new_horizontal_mae_deg": (
                    summary.get(
                        "horizontal_mean_abs_error_deg"
                    )
                ),
                "original_horizontal_p95_deg": (
                    previous.get(
                        "horizontal_p95_abs_error_deg"
                    )
                ),
                "new_horizontal_p95_deg": (
                    summary.get(
                        "horizontal_p95_abs_error_deg"
                    )
                ),
                "original_detection_rate": (
                    previous.get("detection_rate")
                ),
                "new_detection_rate": (
                    summary.get("detection_rate")
                ),
            }

        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\n" + "=" * 72)
        print(" RAVEN — OPTICAL CALIBRATION RESULTS")
        print("=" * 72)

        print(
            f"[CALIBRATION] Applied yaw LEFT: "
            f"{YAW_DEG:+.2f} deg"
        )
        print(
            f"[CALIBRATION] Applied pitch UP: "
            f"{PITCH_UP_DEG:+.2f} deg"
        )

        if static:
            stationary = summary["stationary_window"]

            print(
                "[CALIBRATION] Stationary horizontal error:",
                f"{stationary['mean_signed_horizontal_error_deg']:+.3f} deg",
            )
            print(
                "[CALIBRATION] Stationary vertical error:",
                f"{stationary['mean_signed_vertical_error_deg']:+.3f} deg",
            )

        print(
            "[CALIBRATION] Measured horizontal MAE:",
            summary.get("horizontal_mean_abs_error_deg"),
        )
        print(
            "[CALIBRATION] Measured horizontal p95:",
            summary.get("horizontal_p95_abs_error_deg"),
        )
        print(
            "[CALIBRATION] Detection rate:",
            summary.get("detection_rate"),
        )
        print(
            "[CALIBRATION] Summary:",
            self.summary_path,
        )


def main():
    print("\n" + "=" * 72)
    print(" RAVEN — INDEPENDENT TRIPOD OPTICAL CALIBRATION")
    print("=" * 72)

    print(
        f"Camera yaw LEFT: {YAW_DEG:+.2f} deg"
    )
    print(
        f"Camera pitch UP: {PITCH_UP_DEG:+.2f} deg"
    )
    print(
        "G1 grasp and physical pan controller: UNCHANGED"
    )
    print(
        "Original tripod framing experiment: UNCHANGED"
    )
    print(
        f"Output: {calibrated_output_dir}"
    )

    # Preserve the baseline robot controller and perception code.
    # Replace only the tripod-camera scene and measurement recorder.
    framing.SceneWithFramingCamera = CalibratedFramingScene
    framing.FramingRecorder = CalibrationRecorder

    try:
        framing.main()
    finally:
        framing.SceneWithFramingCamera = original_scene_class
        framing.FramingRecorder = original_recorder_class
        framing.baseline.SceneCfg = (
            original_baseline_scene_class
        )
        framing.baseline.args.output_dir = str(
            baseline_output_dir
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        framing.baseline.simulation_app.close()
