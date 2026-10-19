
"""RAVEN: G1-mounted RGB-D perception of a moving humanoid target.

Milestone 1 of autonomous cinematography: acquire G1 camera RGB/depth, detect
an actor from actual camera pixels, recover its position using measured depth,
then compute the pan angle requested of the existing tripod.

THIS SCRIPT DOES NOT DRIVE THE TRIPOD. The verified 20-degree physical pan
controller remains untouched; perception must produce valid image-derived
measurements before we connect the two systems.

Scene actor: an intentionally magenta, human-shaped surrogate. The detector is
RGB colour segmentation, NOT a trained general-purpose human detector.
Ground-truth actor coordinates are used ONLY for evaluation in the CSV; never
for detection, localisation or the proposed pan command.

Target environment: Isaac Sim 5.1 / Isaac Lab 2.3.2, Windows, G1 BrainCo asset.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RAVEN G1 RGB-D perception milestone")
parser.add_argument("--duration-s", type=float, default=30.0)
parser.add_argument("--actor-period-s", type=float, default=10.0)
parser.add_argument("--actor-amplitude-m", type=float, default=0.40)
parser.add_argument("--output-dir", default="raven_perception_results")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.duration_s <= 2 or args.actor_period_s <= 2:
    parser.error("duration and actor period must exceed 2 seconds")
if not 0.05 <= args.actor_amplitude_m <= 0.50:
    parser.error("actor amplitude must be between 0.05 and 0.50 m")
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import torch
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from raven_g1.assets import G1_BRAINCO_MODE15_CFG, TRIPOD_CAMERA_CFG

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

DT = 1.0 / 120.0
PERCEPTION_EVERY_STEPS = 12
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 360
TRIPOD_X = 0.61
ACTOR_X = 3.20
PAN_LIMIT_DEG = 10.0
MIN_COLOURED_PIXELS = 80
ACTOR_ROOT = "/World/RAVEN_Actor"

robot_cfg = G1_BRAINCO_MODE15_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
tripod_cfg = TRIPOD_CAMERA_CFG.copy()
tripod_cfg.prim_path = "{ENV_REGEX_NS}/Tripod"
original_pos = tripod_cfg.init_state.pos
tripod_cfg.init_state.pos = (TRIPOD_X, original_pos[1], original_pos[2])


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2800.0,
            color=(0.85, 0.85, 0.85),
        ),
    )
    robot = robot_cfg
    tripod = tripod_cfg

    # The Unitree G1 URDF places d435_link at this fixed translation
    # relative to torso_link. Mount under torso_link because some
    # importers merge fixed child links.
    #
    # The optical axis is aligned with the G1's forward +X direction.
    g1_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/raven_d435_rgbd",
        update_period=0.1,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
            focus_distance=3.0,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0576235, 0.01753, 0.42987),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


def material(stage, name: str, rgb: tuple[float, float, float]):
    path = f"/World/Looks/{name}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*rgb))
    shader.CreateInput(
        "emissiveColor", Sdf.ValueTypeNames.Color3f
    ).Set(
        Gf.Vec3f(rgb[0] * 0.3, rgb[1] * 0.3, rgb[2] * 0.3)
    )
    shader.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    ).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    return mat


def make_actor(stage):
    """Create a non-colliding human-shaped coloured proxy."""
    root = UsdGeom.Xform.Define(stage, ACTOR_ROOT)
    pink = material(
        stage, "RAVEN_Actor_Magenta", (0.98, 0.03, 0.75)
    )
    grey = material(
        stage, "RAVEN_Actor_Grey", (0.16, 0.19, 0.25)
    )

    def capsule(name, centre, radius, height, mat):
        prim = UsdGeom.Capsule.Define(
            stage, ACTOR_ROOT + "/" + name
        )
        prim.CreateRadiusAttr(radius)
        prim.CreateHeightAttr(height)
        prim.CreateAxisAttr("Z")
        UsdGeom.XformCommonAPI(prim).SetTranslate(
            Gf.Vec3d(*centre)
        )
        prim.CreateDisplayColorAttr([
            Gf.Vec3f(0.98, 0.03, 0.75)
        ])
        UsdShade.MaterialBindingAPI.Apply(
            prim.GetPrim()
        ).Bind(mat)

    capsule(
        "Torso", (0.0, 0.0, 1.10), 0.24, 0.70, pink
    )
    capsule(
        "LeftLeg", (0.0, 0.12, 0.43), 0.075, 0.70, grey
    )
    capsule(
        "RightLeg", (0.0, -0.12, 0.43), 0.075, 0.70, grey
    )
    capsule(
        "LeftArm", (0.0, 0.31, 1.09), 0.065, 0.65, grey
    )
    capsule(
        "RightArm", (0.0, -0.31, 1.09), 0.065, 0.65, grey
    )

    head = UsdGeom.Sphere.Define(
        stage, ACTOR_ROOT + "/Head"
    )
    head.CreateRadiusAttr(0.14)
    UsdGeom.XformCommonAPI(head).SetTranslate(
        Gf.Vec3d(0.0, 0.0, 1.74)
    )
    UsdShade.MaterialBindingAPI.Apply(
        head.GetPrim()
    ).Bind(grey)
    return root


def move_actor(actor_root, actor_y):
    UsdGeom.XformCommonAPI(actor_root).SetTranslate(
        Gf.Vec3d(ACTOR_X, actor_y, 0.0)
    )


def detect_actor(g1_camera, tripod):
    """Detect coloured actor from RGB and reconstruct its position using depth."""
    data = g1_camera.data
    rgb = data.output["rgb"][0, ..., :3]

    if rgb.dtype != torch.uint8:
        if rgb.max() <= 1:
            rgb8 = (rgb * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            rgb8 = rgb.clamp(0, 255).to(torch.uint8)
    else:
        rgb8 = rgb

    r = rgb8[..., 0].to(torch.float32)
    g = rgb8[..., 1].to(torch.float32)
    b = rgb8[..., 2].to(torch.float32)

    mask = (
        (r > 90)
        & (b > 65)
        & (r > g * 1.5)
        & (b > g * 1.3)
    )

    coords = torch.nonzero(mask, as_tuple=False)
    if coords.shape[0] < MIN_COLOURED_PIXELS:
        return None, rgb8

    depth = data.output[
        "distance_to_image_plane"
    ][0]
    if depth.ndim == 3:
        depth = depth[..., 0]

    good = (
        mask
        & torch.isfinite(depth)
        & (depth > 0.2)
        & (depth < 15.0)
    )
    valid = torch.nonzero(good, as_tuple=False)
    if valid.shape[0] < MIN_COLOURED_PIXELS:
        return None, rgb8

    rows = valid[:, 0]
    cols = valid[:, 1]

    u = cols.float().mean()
    v = rows.float().mean()
    z = depth[rows, cols].median()

    k = data.intrinsic_matrices[0]
    fx, fy = k[0, 0], k[1, 1]
    cx, cy = k[0, 2], k[1, 2]

    point_optical = torch.stack((
        (u - cx) * z / fx,
        (v - cy) * z / fy,
        z,
    ))

    world_q = data.quat_w_ros[0]
    world_p = data.pos_w[0] + quat_apply(
        world_q.unsqueeze(0),
        point_optical.unsqueeze(0),
    )[0]

    tripod_root = tripod.data.root_pose_w[0]
    tripod_q_inverse = tripod_root[3:7].clone()
    tripod_q_inverse[1:] = -tripod_q_inverse[1:]

    relative = quat_apply(
        tripod_q_inverse.unsqueeze(0),
        (world_p - tripod_root[:3]).unsqueeze(0),
    )[0]

    bearing = math.degrees(math.atan2(
        float(relative[1]),
        float(relative[0]),
    ))
    commanded_bearing = max(
        -PAN_LIMIT_DEG,
        min(PAN_LIMIT_DEG, bearing),
    )

    return {
        "u": float(u.item()),
        "v": float(v.item()),
        "depth_m": float(z.item()),
        "position_w": [
            float(x) for x in world_p.tolist()
        ],
        "bearing_deg": bearing,
        "desired_pan_deg": commanded_bearing,
        "pixels": int(valid.shape[0]),
        "bbox": (
            int(cols.min().item()),
            int(rows.min().item()),
            int(cols.max().item()),
            int(rows.max().item()),
        ),
    }, rgb8


def save_frame(path: Path, rgb8, detection, label: str):
    if not HAS_PIL:
        return

    image = Image.fromarray(
        rgb8.detach().cpu().numpy(),
        mode="RGB",
    )

    if detection is not None:
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            detection["bbox"],
            outline=(50, 255, 100),
            width=3,
        )
        u = round(detection["u"])
        v = round(detection["v"])
        draw.line(
            [(u - 10, v), (u + 10, v)],
            fill=(50, 255, 100),
            width=2,
        )
        draw.line(
            [(u, v - 10), (u, v + 10)],
            fill=(50, 255, 100),
            width=2,
        )

    ImageDraw.Draw(image).text(
        (12, 10),
        label,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    image.save(path)


def main():
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device="cpu",
            use_fabric=False,
        )
    )
    sim.set_camera_view(
        eye=(2.1, 2.5, 1.9),
        target=(1.55, 0.0, 1.0),
    )

    scene = InteractiveScene(
        SceneCfg(num_envs=1, env_spacing=2.0)
    )

    actor = make_actor(
        omni.usd.get_context().get_stage()
    )
    move_actor(actor, 0.0)
    sim.reset()

    camera = scene["g1_rgbd"]
    tripod = scene["tripod"]

    csv_path = out_dir / "raven_g1_rgbd_actor.csv"
    names = (
        "time_s",
        "detected",
        "pixels",
        "u_px",
        "v_px",
        "depth_m",
        "estimated_x_m",
        "estimated_y_m",
        "estimated_z_m",
        "desired_pan_deg",
        "true_actor_pan_deg",
        "bearing_error_deg",
    )

    readings = 0
    total = 0
    errors = []
    last_report_sec = -1
    last_image_time = -100.0
    max_steps = int(args.duration_s / DT)

    print("\n" + "=" * 72)
    print(" RAVEN — G1 d435 RGB-D ACTOR PERCEPTION")
    print("=" * 72)
    print("Mounted at the G1 d435 location (torso_link fixed offset).")
    print("Target detection: ACTUAL RGB pixels.")
    print("Target position: ACTUAL depth image.")
    print("Privileged actor position is used FOR EVALUATION ONLY.")
    print("This milestone computes a pan request; tripod stays locked.")
    print(f"Results directory: {out_dir}\n")

    try:
        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=names
            )
            writer.writeheader()

            for step in range(max_steps):
                if not simulation_app.is_running():
                    break

                t = step * DT
                lateral = (
                    args.actor_amplitude_m
                    * math.sin(
                        2 * math.pi * t
                        / args.actor_period_s
                    )
                )
                move_actor(actor, lateral)

                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(DT)

                if (
                    step < 36
                    or step % PERCEPTION_EVERY_STEPS != 0
                ):
                    continue

                total += 1
                result, rgb = detect_actor(
                    camera, tripod
                )

                tripod_xy = (
                    tripod.data.root_pose_w[0, :2]
                )

                # Ground truth is evaluation-only.
                gt_bearing = math.degrees(
                    math.atan2(
                        lateral - float(tripod_xy[1]),
                        ACTOR_X - float(tripod_xy[0]),
                    )
                )

                row = {
                    name: "" for name in names
                }
                row["time_s"] = round(t, 3)
                row["true_actor_pan_deg"] = round(
                    gt_bearing, 3
                )
                row["detected"] = (
                    result is not None
                )

                if result is not None:
                    readings += 1
                    err = (
                        result["bearing_deg"]
                        - gt_bearing
                    )
                    errors.append(abs(err))

                    row.update({
                        "pixels": result["pixels"],
                        "u_px": round(result["u"], 2),
                        "v_px": round(result["v"], 2),
                        "depth_m": round(
                            result["depth_m"], 3
                        ),
                        "estimated_x_m": round(
                            result["position_w"][0], 3
                        ),
                        "estimated_y_m": round(
                            result["position_w"][1], 3
                        ),
                        "estimated_z_m": round(
                            result["position_w"][2], 3
                        ),
                        "desired_pan_deg": round(
                            result["desired_pan_deg"], 3
                        ),
                        "bearing_error_deg": round(
                            err, 3
                        ),
                    })

                writer.writerow(row)

                if (
                    t - last_image_time >= 3.0
                ):
                    if result:
                        label = (
                            f"G1 RGB-D t={t:.1f}s | "
                            f"target "
                            f"{result['desired_pan_deg']:+.1f} deg"
                        )
                    else:
                        label = (
                            f"G1 RGB-D t={t:.1f}s | "
                            "TARGET NOT DETECTED"
                        )

                    save_frame(
                        out_dir
                        / f"g1_frame_{round(t):03d}.png",
                        rgb,
                        result,
                        label,
                    )
                    last_image_time = t

                if int(t) != last_report_sec:
                    if result is None:
                        print(
                            f"[G1 RGB-D] t={t:5.1f}s"
                            " | target NOT DETECTED"
                        )
                    else:
                        print(
                            f"[G1 RGB-D] t={t:5.1f}s"
                            f" | pixel=("
                            f"{result['u']:5.1f},"
                            f"{result['v']:5.1f})"
                            f" | depth="
                            f"{result['depth_m']:.2f}m"
                            f" | requestedPan="
                            f"{result['desired_pan_deg']:+6.2f}deg"
                            f" | trueBearing="
                            f"{gt_bearing:+6.2f}deg"
                            f" | error={err:+5.2f}deg"
                        )

                    last_report_sec = int(t)

                if total % 50 == 0:
                    stream.flush()
    finally:
        summary = {
            "milestone": (
                "G1 d435 RGB-D vision: "
                "coloured humanoid surrogate"
            ),
            "scope": (
                "image-derived target localisation "
                "and pan-angle request; NO PHYSICAL PAN"
            ),
            "frames_evaluated": total,
            "frames_detected": readings,
            "detection_rate": (
                readings / total if total else 0.0
            ),
            "mean_abs_bearing_error_deg": (
                sum(errors) / len(errors)
                if errors else None
            ),
            "max_abs_bearing_error_deg": (
                max(errors) if errors else None
            ),
            "csv": str(csv_path),
            "images": str(out_dir),
        }

        summary_path = (
            out_dir / "raven_g1_rgbd_summary.json"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print(
            "\n[RAVEN] Detection rate: "
            f"{summary['detection_rate'] * 100:.1f}%"
        )
        print("[RAVEN] RGB-D CSV:", csv_path)
        print("[RAVEN] RGB frames:", out_dir)
        print("[RAVEN] Summary:", summary_path)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
