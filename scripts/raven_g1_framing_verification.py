
"""RAVEN: tripod-mounted RGB camera framing verification.

Runs the existing working raven_g1_actor_tracking.py unchanged.
Adds a second RGB camera attached to the tripod's camera_link.

Fixes the previous zero-offset optical mount and reports its physical
world-facing direction. Camera framing is measured using actual RGB
pixels, independently of the G1's actor tracking estimates.

The magenta target is a controlled surrogate, not general human
detection. The grasp remains physics-assisted.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

# This import launches Isaac Sim through the baseline AppLauncher.
# It does not invoke baseline.main(), which is guarded by __main__.
import raven_g1_actor_tracking as baseline

import torch
import isaaclab.sim as sim_utils

from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# --------------------------------------------------------------
# Tripod optical mounting
# --------------------------------------------------------------

# In the previous script the camera sensor was at (0, 0, 0),
# potentially inside the opaque rectangular camera housing.
#
# Move the virtual lens beyond the housing along local +X.
# The exact lens-to-link transform can be refined from the URDF.
#
# A diagnostic prints the camera's forward alignment with the
# stationary actor at the beginning of tracking.
#
# If alignment is NEGATIVE, switch CAMERA_REVERSED to True.
# That applies a fixed 180-degree mounting correction.
#
# This is sensor extrinsic calibration, not target control.

TRIPOD_CAMERA_REVERSED = False
TRIPOD_CAMERA_STANDOFF_M = 0.38
TRIPOD_CAMERA_HEIGHT_OFFSET_M = 0.03

if TRIPOD_CAMERA_REVERSED:
    TRIPOD_CAMERA_OFFSET_POS = (
        -TRIPOD_CAMERA_STANDOFF_M,
        0.0,
        TRIPOD_CAMERA_HEIGHT_OFFSET_M,
    )
    TRIPOD_CAMERA_OFFSET_QUAT = (
        0.0, 0.0, 0.0, 1.0
    )
else:
    TRIPOD_CAMERA_OFFSET_POS = (
        TRIPOD_CAMERA_STANDOFF_M,
        0.0,
        TRIPOD_CAMERA_HEIGHT_OFFSET_M,
    )
    TRIPOD_CAMERA_OFFSET_QUAT = (
        1.0, 0.0, 0.0, 0.0
    )

TRIPOD_CAMERA_WIDTH = 640
TRIPOD_CAMERA_HEIGHT = 360
MIN_TARGET_PIXELS = 60
SNAPSHOT_PERIOD_S = 4.0

CAMERA_SAMPLE_PERIOD_S = (
    baseline.TRACKING_STEPS / baseline.PHYSICS_HZ
)


# --------------------------------------------------------------
# Isaac Lab scene extension
# --------------------------------------------------------------

@configclass
class SceneWithFramingCamera(baseline.SceneCfg):
    tripod_rgb = CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Tripod/"
            "camera_link/raven_framing_rgb"
        ),
        update_period=0.1,
        width=TRIPOD_CAMERA_WIDTH,
        height=TRIPOD_CAMERA_HEIGHT,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.03, 20.0),
            focus_distance=3.0,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=TRIPOD_CAMERA_OFFSET_POS,
            rot=TRIPOD_CAMERA_OFFSET_QUAT,
            convention="world",
        ),
    )


# --------------------------------------------------------------
# RGB image processing
# --------------------------------------------------------------

def rgb_to_uint8(raw_rgb):
    """Convert RGB tensor to uint8 without propagating NaN."""
    if raw_rgb.dtype == torch.uint8:
        return raw_rgb[..., :3]

    clean = torch.nan_to_num(
        raw_rgb[..., :3],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    multiplier = (
        255.0 if float(clean.max().item()) <= 1.0
        else 1.0
    )

    return (clean * multiplier).clamp(
        0, 255
    ).to(torch.uint8)


def measure_framing(tripod_cam):
    """Measure target displacement from real tripod RGB pixels.

    Actor world position is not used in this function.
    Positive horizontal error means the target is to the
    right of the camera's optical centre.
    """
    data = tripod_cam.data
    rgb_buffer = data.output.get("rgb")

    if rgb_buffer is None:
        return None, None, "camera has no RGB output"

    if rgb_buffer.shape[0] == 0:
        return None, None, "camera buffer empty"

    rgb = rgb_to_uint8(rgb_buffer[0])

    r = rgb[..., 0].float()
    g = rgb[..., 1].float()
    b = rgb[..., 2].float()

    mask = (
        (r > 90)
        & (b > 65)
        & (r > g * 1.5)
        & (b > g * 1.3)
    )

    pixels = torch.nonzero(
        mask, as_tuple=False
    )

    if pixels.shape[0] < MIN_TARGET_PIXELS:
        brightness = rgb.amax(dim=-1).float()

        dark_fraction = float(
            (brightness < 18).float().mean().item()
        )

        if dark_fraction > 0.98:
            return (
                None,
                rgb,
                "raw tripod RGB is nearly uniform black "
                "(mount/view/render issue)",
            )

        return (
            None,
            rgb,
            "no magenta actor in tripod frame",
        )

    intrinsics = data.intrinsic_matrices[0]

    if not bool(
        torch.isfinite(intrinsics).all().item()
    ):
        return (
            None, rgb,
            "tripod camera intrinsics invalid"
        )

    fx = float(intrinsics[0, 0].item())
    fy = float(intrinsics[1, 1].item())
    cx = float(intrinsics[0, 2].item())
    cy = float(intrinsics[1, 2].item())

    if fx <= 1.0 or fy <= 1.0:
        return (
            None, rgb,
            "tripod camera focal length invalid"
        )

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    u = float(cols.float().mean().item())
    v = float(rows.float().mean().item())

    error_x_px = u - cx
    error_y_px = v - cy

    error_x_deg = math.degrees(
        math.atan2(error_x_px, fx)
    )
    error_y_deg = math.degrees(
        math.atan2(error_y_px, fy)
    )

    if not (
        math.isfinite(error_x_deg)
        and math.isfinite(error_y_deg)
    ):
        return (
            None, rgb,
            "invalid framing error"
        )

    return {
        "u_px": u,
        "v_px": v,
        "cx_px": cx,
        "cy_px": cy,
        "error_x_px": error_x_px,
        "error_y_px": error_y_px,
        "error_x_deg": error_x_deg,
        "error_y_deg": error_y_deg,
        "pixel_count": int(pixels.shape[0]),
        "bbox": (
            int(cols.min().item()),
            int(rows.min().item()),
            int(cols.max().item()),
            int(rows.max().item()),
        ),
    }, rgb, "ok"


def save_tripod_image(
    path: Path,
    rgb,
    measurement,
    label,
):
    if not HAS_PIL or rgb is None:
        return

    image = Image.fromarray(
        rgb.detach().cpu().numpy()
    )
    draw = ImageDraw.Draw(image)

    width, height = image.size
    centre = (width // 2, height // 2)

    draw.line(
        (
            centre[0] - 15, centre[1],
            centre[0] + 15, centre[1],
        ),
        fill=(255, 255, 255),
        width=2,
    )
    draw.line(
        (
            centre[0], centre[1] - 15,
            centre[0], centre[1] + 15,
        ),
        fill=(255, 255, 255),
        width=2,
    )

    if measurement is not None:
        draw.rectangle(
            measurement["bbox"],
            outline=(45, 255, 110),
            width=3,
        )

        u = round(measurement["u_px"])
        v = round(measurement["v_px"])

        draw.line(
            (centre[0], centre[1], u, v),
            fill=(255, 215, 0),
            width=2,
        )
        draw.ellipse(
            (u - 4, v - 4, u + 4, v + 4),
            fill=(45, 255, 110),
        )

    draw.text(
        (12, 12),
        label,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )

    image.save(path)


# --------------------------------------------------------------
# Framing telemetry
# --------------------------------------------------------------

class FramingRecorder:

    def __init__(self, scene):
        self.scene = scene
        self.tripod_camera = scene["tripod_rgb"]
        self.tripod = scene["tripod"]

        self.pan_joint_id = (
            self.tripod.data.joint_names.index(
                "pan_joint"
            )
        )

        self.output_dir = (
            Path(baseline.args.output_dir).resolve()
            / "tripod_framing"
        )
        self.output_dir.mkdir(
            parents=True, exist_ok=True
        )

        self.csv_path = (
            self.output_dir / "tripod_framing.csv"
        )
        self.summary_path = (
            self.output_dir
            / "tripod_framing.summary.json"
        )

        self.sample_index = 0
        self.detected_count = 0

        self.horizontal_abs_errors = []
        self.vertical_abs_errors = []

        self.zero_pan = None
        self.last_image_t = -1e9

        self._stream = None
        self._writer = None
        self._warning_at = -1
        self._mount_checked = False

        self.mount_alignment = None
        self.raw_rgb_mean = None

    def open(self):
        self._stream = self.csv_path.open(
            "w", newline="", encoding="utf-8"
        )

        self._writer = csv.DictWriter(
            self._stream,
            fieldnames=(
                "tracking_time_s",
                "tripod_detected",
                "tripod_status",
                "target_u_px",
                "target_v_px",
                "target_pixels",
                "frame_centre_u_px",
                "frame_centre_v_px",
                "horizontal_error_px",
                "vertical_error_px",
                "horizontal_error_deg",
                "vertical_error_deg",
                "actual_pan_deg",
                "g1_target_bearing_deg",
                "g1_target_detected",
            ),
        )
        self._writer.writeheader()

    def diagnose_camera_mount(self, rgb):
        """Diagnose the fixed camera extrinsics once.

        The actor's known scene position is used ONLY for
        installation diagnostics, never for target tracking
        or framing measurement.
        """
        if self._mount_checked:
            return

        self._mount_checked = True

        camera_link_id = list(
            self.tripod.data.body_names
        ).index("camera_link")

        link = self.tripod.data.body_pose_w[
            0, camera_link_id
        ]

        q = link[3:7].unsqueeze(0)
        device = link.device
        dtype = link.dtype

        offset = torch.tensor(
            [TRIPOD_CAMERA_OFFSET_POS],
            dtype=dtype,
            device=device,
        )

        local_forward_sign = (
            -1.0 if TRIPOD_CAMERA_REVERSED
            else 1.0
        )

        forward_local = torch.tensor(
            [[local_forward_sign, 0.0, 0.0]],
            dtype=dtype,
            device=device,
        )

        sensor_position = (
            link[:3]
            + quat_apply(q, offset)[0]
        )

        sensor_forward = quat_apply(
            q, forward_local
        )[0]

        actor_centre = torch.tensor(
            [baseline.ACTOR_X, 0.0, 1.1],
            device=device,
            dtype=dtype,
        )

        towards_actor = (
            actor_centre - sensor_position
        )
        towards_actor = (
            towards_actor
            / torch.linalg.vector_norm(
                towards_actor
            ).clamp_min(1e-6)
        )

        alignment = float(
            torch.dot(
                sensor_forward,
                towards_actor,
            ).item()
        )
        self.mount_alignment = alignment

        raw_mean = rgb.float().mean(
            dim=(0, 1)
        )
        self.raw_rgb_mean = [
            round(float(c.item()), 2)
            for c in raw_mean
        ]

        print(
            "[TRIPOD CAMERA] Optical origin XYZ:",
            [
                round(float(x), 3)
                for x in sensor_position.tolist()
            ],
        )
        print(
            "[TRIPOD CAMERA] Optical forward XYZ:",
            [
                round(float(x), 3)
                for x in sensor_forward.tolist()
            ],
        )
        print(
            "[TRIPOD CAMERA] Actor alignment:",
            f"{alignment:+.3f}",
        )
        print(
            "[TRIPOD CAMERA] Raw RGB mean:",
            self.raw_rgb_mean,
        )

        if alignment < 0:
            print(
                "[TRIPOD CAMERA] Camera faces away from actor. "
                "Set TRIPOD_CAMERA_REVERSED=True."
            )
        elif (
            self.raw_rgb_mean is not None
            and max(self.raw_rgb_mean) < 20
        ):
            print(
                "[TRIPOD CAMERA] Camera direction is plausible, "
                "but RGB is dark. Inspect housing occlusion "
                "and RTX rendering."
            )
        else:
            print(
                "[TRIPOD CAMERA] Optical mount appears "
                "plausible."
            )

    def observe(self, g1_result):
        if self._writer is None:
            raise RuntimeError(
                "FramingRecorder.open() not called"
            )

        t = (
            self.sample_index
            * CAMERA_SAMPLE_PERIOD_S
        )
        self.sample_index += 1

        if self.zero_pan is None:
            self.zero_pan = float(
                self.tripod.data.joint_pos[
                    0, self.pan_joint_id
                ].item()
            )

        actual_pan = math.degrees(
            float(
                self.tripod.data.joint_pos[
                    0, self.pan_joint_id
                ].item()
            ) - self.zero_pan
        )

        measurement, rgb, reason = measure_framing(
            self.tripod_camera
        )

        if rgb is not None and not self._mount_checked:
            self.diagnose_camera_mount(rgb)

        row = {
            "tracking_time_s": round(t, 3),
            "tripod_detected": measurement is not None,
            "tripod_status": reason,
            "target_u_px": "",
            "target_v_px": "",
            "target_pixels": "",
            "frame_centre_u_px": "",
            "frame_centre_v_px": "",
            "horizontal_error_px": "",
            "vertical_error_px": "",
            "horizontal_error_deg": "",
            "vertical_error_deg": "",
            "actual_pan_deg": round(actual_pan, 4),
            "g1_target_bearing_deg": (
                round(g1_result["bearing_deg"], 4)
                if g1_result else ""
            ),
            "g1_target_detected": (
                g1_result is not None
            ),
        }

        if measurement is not None:
            self.detected_count += 1

            self.horizontal_abs_errors.append(
                abs(measurement["error_x_deg"])
            )
            self.vertical_abs_errors.append(
                abs(measurement["error_y_deg"])
            )

            row.update({
                "target_u_px": round(
                    measurement["u_px"], 2
                ),
                "target_v_px": round(
                    measurement["v_px"], 2
                ),
                "target_pixels": (
                    measurement["pixel_count"]
                ),
                "frame_centre_u_px": round(
                    measurement["cx_px"], 2
                ),
                "frame_centre_v_px": round(
                    measurement["cy_px"], 2
                ),
                "horizontal_error_px": round(
                    measurement["error_x_px"], 2
                ),
                "vertical_error_px": round(
                    measurement["error_y_px"], 2
                ),
                "horizontal_error_deg": round(
                    measurement["error_x_deg"], 4
                ),
                "vertical_error_deg": round(
                    measurement["error_y_deg"], 4
                ),
            })

        self._writer.writerow(row)

        if (
            t - self.last_image_t
            >= SNAPSHOT_PERIOD_S - 1e-6
        ):
            if measurement is not None:
                label = (
                    f"TRIPOD RGB t={t:.1f}s "
                    f"| error="
                    f"{measurement['error_x_deg']:+.2f}deg"
                )
            else:
                label = (
                    f"TRIPOD RGB t={t:.1f}s "
                    f"| target NOT DETECTED"
                )

            save_tripod_image(
                self.output_dir
                / f"tripod_frame_{int(round(t)):03d}.png",
                rgb,
                measurement,
                label,
            )
            self.last_image_t = t

        second = int(t)

        if second != self._warning_at:
            self._warning_at = second

            if measurement is None:
                print(
                    f"[FRAMING] t={t:5.1f}s "
                    f"| {reason}"
                )
            else:
                print(
                    f"[FRAMING] t={t:5.1f}s "
                    f"| x="
                    f"{measurement['error_x_px']:+6.1f}px "
                    f"| horizontal="
                    f"{measurement['error_x_deg']:+6.2f}deg "
                    f"| pan={actual_pan:+6.2f}deg"
                )

        if self.sample_index % 50 == 0:
            self._stream.flush()

    def close(self):
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None

        errors = sorted(
            self.horizontal_abs_errors
        )

        p95 = (
            errors[
                math.ceil(0.95 * len(errors)) - 1
            ]
            if errors else None
        )

        summary = {
            "milestone": (
                "tripod RGB cinematic framing"
            ),
            "status": (
                "MEASURED"
                if errors
                else "NO_TRIPOD_TARGET_DETECTION"
            ),
            "camera_parent": "Tripod/camera_link",
            "controller": (
                "unchanged raven_g1_actor_tracking.py"
            ),
            "framing_source": (
                "tripod camera RGB actor segmentation"
            ),
            "assisted_grasp": True,
            "sample_count": self.sample_index,
            "detected_count": self.detected_count,
            "detection_rate": (
                self.detected_count / self.sample_index
                if self.sample_index else 0.0
            ),
            "horizontal_mean_abs_error_deg": (
                statistics.mean(errors)
                if errors else None
            ),
            "horizontal_p95_abs_error_deg": p95,
            "horizontal_max_abs_error_deg": (
                max(errors) if errors else None
            ),
            "vertical_mean_abs_error_deg": (
                statistics.mean(
                    self.vertical_abs_errors
                )
                if self.vertical_abs_errors
                else None
            ),
            "csv": str(self.csv_path),
            "images_dir": str(self.output_dir),
            "camera_mount_offset": (
                TRIPOD_CAMERA_OFFSET_POS
            ),
            "camera_reversed": (
                TRIPOD_CAMERA_REVERSED
            ),
            "initial_actor_forward_alignment": (
                self.mount_alignment
            ),
            "initial_raw_rgb_mean": (
                self.raw_rgb_mean
            ),
        }

        self.summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(
            "\n[FRAMING] Tripod image detections:",
            f"{self.detected_count}/{self.sample_index}",
        )
        print(
            "[FRAMING] Summary:",
            self.summary_path,
        )
        print(
            "[FRAMING] CSV:",
            self.csv_path,
        )


# --------------------------------------------------------------
# Application
# --------------------------------------------------------------

def main():
    # Adds only the additional camera.
    # The baseline robot controller is untouched.
    baseline.SceneCfg = SceneWithFramingCamera

    original_run = baseline.run
    original_detect = baseline.detect_actor

    def run_with_framing(sim, scene, actor_root):
        recorder = FramingRecorder(scene)
        recorder.open()

        print(
            "[FRAMING] Camera attached under camera_link."
        )
        print(
            "[FRAMING] Lens offset:",
            TRIPOD_CAMERA_OFFSET_POS,
        )
        print(
            "[FRAMING] Camera reversed:",
            TRIPOD_CAMERA_REVERSED,
        )

        def recorded_detect(
            g1_camera,
            tripod,
            robot,
            torso_body,
        ):
            result, rgb, reason = original_detect(
                g1_camera,
                tripod,
                robot,
                torso_body,
            )

            recorder.observe(result)

            return result, rgb, reason

        baseline.detect_actor = recorded_detect

        try:
            return original_run(
                sim,
                scene,
                actor_root,
            )
        finally:
            baseline.detect_actor = original_detect
            recorder.close()

    baseline.run = run_with_framing

    try:
        baseline.main()
    finally:
        baseline.run = original_run
        baseline.SceneCfg = SceneWithFramingCamera


if __name__ == "__main__":
    try:
        main()
    finally:
        baseline.simulation_app.close()
