
"""RAVEN: GEOMETRY-FIRST +/-30-degree G1/tripod workspace planner.

Runs a KINEMATIC PREFLIGHT on the actual imported robot articulation.
It does not actuate the tripod or claim a completed physical pan.

Why this exists:
  * The original waist-yaw + 7 left-arm 6D IK was stable at +/-10 deg.
  * The experimental 10-DoF controller saturated waist roll/pitch before panning.
  * We therefore return to the verified 8-DoF chain and search tripod geometry.

For every candidate tripod XY offset and orientation, this planner:
  1) Takes the known calibrated grasp and the real USD handle/pivot transforms.
  2) Computes desired hand poses along 0 -> -30 and 0 -> +30 degrees.
  3) Uses the simulated G1 Jacobian and FK to solve contiguous IK paths.
  4) Checks operational/hard joint limits and an elbow-torso clearance proxy.
  5) Writes diagnostics and a candidate joint-space trajectory to JSON/CSV.

IMPORTANT: This is NOT exact mesh-based self-collision planning, does NOT
check dynamic loads or contacts, and does NOT prove unaided BrainCo gripping.
A candidate passing here still needs a collision-checked physical test.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RAVEN +/-30 deg geometry-first workspace planner")
parser.add_argument("--sweep-deg", type=float, default=30.0)
parser.add_argument("--yaw-workspace-deg", type=float, default=40.0,
                    help="Operational waist-yaw range each way; NOT the G1's mechanical limit")
parser.add_argument("--search", choices=("small", "wide"), default="small",
                    help="Small=12 tripod layouts; wide=45 tripod layouts")
parser.add_argument("--step-deg", type=float, default=5.0,
                    help="Angular step used for waypoint evaluation")
parser.add_argument("--max-iterations", type=int, default=55)
parser.add_argument("--csv", default="raven_workspace_candidates.csv")
parser.add_argument("--output", default="raven_workspace_result.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not (10 <= args.sweep_deg <= 30):
    parser.error("--sweep-deg must be between 10 and 30")
if not (24 <= args.yaw_workspace_deg <= 60):
    parser.error("--yaw-workspace-deg must be between 24 and 60")
if not (2 <= args.step_deg <= 10):
    parser.error("--step-deg must be between 2 and 10")
if not (10 <= args.max_iterations <= 150):
    parser.error("--max-iterations must be between 10 and 150")
simulation_app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms, quat_apply, quat_mul, subtract_frame_transforms,
)
from isaaclab_assets import G1_INSPIRE_FTP_CFG
from raven_g1.assets import G1_BRAINCO_MODE15_CFG, TRIPOD_CAMERA_CFG

TRIPOD_X = 0.61
DT = 1.0 / 120.0
ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
)
IK_NAMES = ("waist_yaw_joint",) + ARM_NAMES
CALIBRATED_ARM = (-1.10463631, 0.45274511, 0.89249939, 2.09439969,
                  0.67981434, -0.22342572, -0.18944184)

# Conservative IK operating envelope. These are design choices, NOT G1 hardware limits.
ELBOW_MIN_DEG = 95.0
ELBOW_MAX_DEG = 118.0
ELBOW_PREF_DEG = 108.0
MIN_ELBOW_OUT_M = 0.14    # geometric proxy, not a verified collision distance
POS_TOL_M = 0.007
ROT_TOL_DEG = 4.0
MAX_IK_STEP_RAD = 0.085
NULLSPACE_STEP_RAD = 0.007

robot_cfg = G1_BRAINCO_MODE15_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
robot_cfg.actuators = copy.deepcopy(G1_INSPIRE_FTP_CFG.actuators)
robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_pinky_.*"],
    effort_limit_sim=45.0, velocity_limit_sim=10.0,
    stiffness=30.0, damping=2.0, armature=0.001,
)
tripod_cfg = TRIPOD_CAMERA_CFG.copy()
tripod_cfg.prim_path = "{ENV_REGEX_NS}/Tripod"
p = tripod_cfg.init_state.pos
tripod_cfg.init_state.pos = (TRIPOD_X, p[1], p[2])
tripod_cfg.actuators = {
    "pan_lock": ImplicitActuatorCfg(
        joint_names_expr=["pan_joint"], effort_limit_sim=100.0,
        velocity_limit_sim=5.0, stiffness=500.0, damping=30.0,
    ),
    "tilt_lock": ImplicitActuatorCfg(
        joint_names_expr=["tilt_joint"], effort_limit_sim=100.0,
        velocity_limit_sim=5.0, stiffness=500.0, damping=30.0,
    ),
}

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
    robot = robot_cfg
    tripod = tripod_cfg


def yaw_quaternion(deg: float, device: str, dtype: torch.dtype) -> torch.Tensor:
    radians = math.radians(deg)
    return torch.tensor([[math.cos(radians / 2), 0, 0, math.sin(radians / 2)]],
                        dtype=dtype, device=device)


def position_orientation_error(target_p, target_q, measured_p, measured_q):
    pe = float(torch.linalg.vector_norm(target_p - measured_p, dim=-1)[0].item())
    dot = float(torch.abs(torch.sum(target_q * measured_q, dim=-1))[0].clamp(max=1.0).item())
    ae = math.degrees(2 * math.acos(max(0.0, min(1.0, dot))))
    return pe, ae


def candidate_grid():
    # A 5-cm or 10-cm repositioning is a physically meaningful adjustment.
    if args.search == "small":
        for dx in (0.0, 0.05, 0.10):
            for dy in (0.0, 0.07):
                for yaw in (0.0, -12.0):
                    yield (dx, dy, yaw)
    else:
        for dx in (-0.05, 0.0, 0.05, 0.10, 0.15):
            for dy in (-0.07, 0.0, 0.07):
                for yaw in (-15.0, 0.0, 15.0):
                    yield (dx, dy, yaw)


def sample_angles():
    # Evaluate both sides from one common neutral pose, retaining continuity
    # on each branch. The reverse paths are the exact reverse of these samples.
    count = math.ceil(args.sweep_deg / args.step_deg)
    levels = [args.sweep_deg * i / count for i in range(1, count + 1)]
    return [0.0] + [-v for v in levels] + [v for v in levels]


def solve_workspace(sim, scene):
    robot, tripod = scene["robot"], scene["tripod"]
    rn = {name: i for i, name in enumerate(robot.data.joint_names)}
    rb = {name: i for i, name in enumerate(robot.data.body_names)}
    tb = {name: i for i, name in enumerate(tripod.data.body_names)}
    ik_ids = [rn[name] for name in IK_NAMES]
    arm_ids = [rn[name] for name in ARM_NAMES]
    hand_body = rb["left_base_link"]
    elbow_body = rb["left_elbow_link"]
    torso_body = rb["torso_link"]
    jacobian_body = hand_body - 1 if robot.is_fixed_base else hand_body
    eidx = IK_NAMES.index("left_elbow_joint")
    init = robot.data.default_joint_pos.clone()
    calibrated = init.clone()
    calibrated[:, arm_ids] = torch.tensor(CALIBRATED_ARM, device=sim.device).view(1, -1)
    zeros = torch.zeros_like(calibrated)
    physical = robot.data.joint_pos_limits[:, ik_ids, :]
    low = physical[..., 0].clone()
    high = physical[..., 1].clone()

    # Upper-arm and wrist joints retain their URDF limits; operational margins
    # are explicitly imposed on waist yaw and elbow only.
    home_waist = float(init[0, rn["waist_yaw_joint"]].item())
    low[:, 0] = torch.maximum(low[:, 0], torch.tensor(home_waist - math.radians(args.yaw_workspace_deg), device=sim.device))
    high[:, 0] = torch.minimum(high[:, 0], torch.tensor(home_waist + math.radians(args.yaw_workspace_deg), device=sim.device))
    low[:, eidx] = torch.maximum(low[:, eidx], torch.tensor(math.radians(ELBOW_MIN_DEG), device=sim.device))
    high[:, eidx] = torch.minimum(high[:, eidx], torch.tensor(math.radians(ELBOW_MAX_DEG), device=sim.device))
    if bool(torch.any(low >= high).item()):
        raise RuntimeError("Configured waist/elbow workspace does not overlap imported URDF limits")

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method="dls", ik_params={"lambda_val": 0.05}),
        num_envs=1, device=sim.device,
    )

    def put_q(q):
        """Kinematic FK probe; NOT a commanded physical camera movement."""
        full = calibrated.clone()
        full[:, ik_ids] = q
        robot.write_joint_state_to_sim(full, zeros)
        robot.set_joint_position_target(full)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)
        return robot.data.joint_pos[:, ik_ids].clone()

    # Read actual USD geometry with G1 in its existing calibrated grasp pose.
    put_q(calibrated[:, ik_ids])

    # Set the tripod pan to its default, locked neutral state.
    tripod_pan_id = list(tripod.data.joint_names).index("pan_joint")
    tripod.set_joint_position_target(torch.zeros((1, 1), device=sim.device),
                                     joint_ids=[tripod_pan_id])
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(DT)
    nominal_hand = robot.data.body_pose_w[:, hand_body].clone()
    nominal_handle = tripod.data.body_pose_w[:, tb["control_handle_link"]].clone()
    nominal_pivot = tripod.data.body_pose_w[:, tb["pan_link"], :3].clone()
    original_tripod_root = tripod.data.root_pose_w[:, :3].clone()
    grip_p, grip_q = subtract_frame_transforms(
        nominal_handle[:, :3], nominal_handle[:, 3:7],
        nominal_hand[:, :3], nominal_hand[:, 3:7],
    )
    angles = sample_angles()
    starting_q = calibrated[:, ik_ids].clone()
    pref = starting_q.clone()
    pref[:, 0] = home_waist
    pref[:, eidx] = math.radians(ELBOW_PREF_DEG)
    pref = torch.maximum(torch.minimum(pref, high), low)
    neutral_seed = torch.maximum(torch.minimum(starting_q, high), low)

    def solve(target_p, target_q, seed):
        q = seed.clone()
        ik.reset()

        # A little regularisation keeps the 8-DoF solution on a continuous
        # branch. No additional roll/pitch waist joints are controlled.
        hand_pos_error, rot_error = float("inf"), float("inf")
        for it in range(args.max_iterations):
            measured = put_q(q)
            hand = robot.data.body_pose_w[:, hand_body]
            hand_pos_error, rot_error = position_orientation_error(
                target_p, target_q, hand[:, :3], hand[:, 3:7]
            )
            if hand_pos_error <= POS_TOL_M and rot_error <= ROT_TOL_DEG:
                break

            base = robot.data.root_pose_w
            target_bp, target_bq = subtract_frame_transforms(
                base[:, :3], base[:, 3:7], target_p, target_q
            )
            hand_bp, hand_bq = subtract_frame_transforms(
                base[:, :3], base[:, 3:7], hand[:, :3], hand[:, 3:7]
            )
            command = torch.cat((target_bp, target_bq), dim=-1)
            ik.set_command(command)
            jac = robot.root_physx_view.get_jacobians()[:, jacobian_body, :, ik_ids]
            computed = ik.compute(hand_bp, hand_bq, jac, measured)

            # Proper 6D null-space projection: elbow preferred angle and
            # waist near home, without allowing the secondary objective to
            # consume the hand-position/orientation task.
            jpinv = torch.linalg.pinv(jac, rtol=1e-4)
            nmat = (torch.eye(len(ik_ids), device=sim.device, dtype=jac.dtype).unsqueeze(0)
                    - torch.bmm(jpinv, jac))
            weights = torch.full_like(measured, 0.15)
            weights[:, 0] = 0.75
            weights[:, eidx] = 0.90
            correction = torch.bmm(nmat, (weights * (pref - measured)).unsqueeze(-1)).squeeze(-1)
            correction = torch.clamp(0.07 * correction, -NULLSPACE_STEP_RAD, NULLSPACE_STEP_RAD)
            dq = torch.clamp(computed + correction - measured, -MAX_IK_STEP_RAD, MAX_IK_STEP_RAD)
            q = torch.maximum(torch.minimum(measured + dq, high), low)
        else:
            put_q(q)

        hand = robot.data.body_pose_w[:, hand_body]
        hand_pos_error, rot_error = position_orientation_error(
            target_p, target_q, hand[:, :3], hand[:, 3:7]
        )
        elbow_angle = math.degrees(float(robot.data.joint_pos[0, rn["left_elbow_joint"]].item()))
        waist_angle = math.degrees(float(robot.data.joint_pos[0, rn["waist_yaw_joint"]].item()) - home_waist)
        elbow = robot.data.body_pose_w[:, elbow_body]
        torso = robot.data.body_pose_w[:, torso_body]
        axis = torch.tensor([[0.0, 1.0, 0.0]], device=sim.device, dtype=elbow.dtype)
        outward = quat_apply(torso[:, 3:7], axis)
        elbow_out_m = float(torch.sum((elbow[:, :3] - torso[:, :3]) * outward, dim=-1)[0].item())
        actual_q = robot.data.joint_pos[:, ik_ids].clone()

        # The actual measured state must match the joint states requested for FK.
        tracking_gap = math.degrees(float(torch.max(torch.abs(actual_q - q)).item()))
        ok = (hand_pos_error <= POS_TOL_M and rot_error <= ROT_TOL_DEG
              and ELBOW_MIN_DEG <= elbow_angle <= ELBOW_MAX_DEG + 0.2
              and abs(waist_angle) <= args.yaw_workspace_deg + 0.2
              and elbow_out_m >= MIN_ELBOW_OUT_M
              and tracking_gap < 2.5)
        return {
            "ok": bool(ok), "q": actual_q[0].tolist(), "position_error_m": hand_pos_error,
            "orientation_error_deg": rot_error, "elbow_deg": elbow_angle,
            "waist_yaw_deg": waist_angle, "elbow_out_m": elbow_out_m,
            "fk_tracking_gap_deg": tracking_gap,
        }

    def target_at(angle_deg, offset_x, offset_y, tripod_yaw_deg):
        shift = torch.tensor([[offset_x, offset_y, 0.0]], dtype=nominal_pivot.dtype, device=sim.device)
        rot = yaw_quaternion(tripod_yaw_deg, sim.device, nominal_handle.dtype)
        new_root = original_tripod_root + shift
        pivot = new_root + quat_apply(rot, nominal_pivot - original_tripod_root)
        handle_zero_p = new_root + quat_apply(rot, nominal_handle[:, :3] - original_tripod_root)
        handle_zero_q = quat_mul(rot, nominal_handle[:, 3:7])
        pan_rot = yaw_quaternion(angle_deg, sim.device, nominal_handle.dtype)
        handle_p = pivot + quat_apply(pan_rot, handle_zero_p - pivot)
        handle_q = quat_mul(pan_rot, handle_zero_q)
        return combine_frame_transforms(handle_p, handle_q, grip_p, grip_q)

    result_rows = []
    winner = None
    winner_score = None
    best = None
    candidates = list(candidate_grid())

    print("\n" + "=" * 72)
    print(" RAVEN — +/-30 DEGREE GEOMETRY-FIRST PREFLIGHT")
    print("=" * 72)
    print("Current tested baseline: +/-10 deg. Planning does NOT turn the camera.")
    print(f"Search: {len(candidates)} tripod geometries x {len(angles)} sample poses")
    print(f"Controlled IK joints: {', '.join(IK_NAMES)}")
    print(f"Waist yaw +/-{args.yaw_workspace_deg:g} deg; elbow {ELBOW_MIN_DEG:g}..{ELBOW_MAX_DEG:g} deg")
    print("Collision screening: elbow-torso clearance PROXY ONLY, not mesh collision")
    print("Negative angles and positive angles are each solved continuously from centre.\n")

    for ci, (dx, dy, yaw) in enumerate(candidates, 1):
        achieved_neg = 0.0
        achieved_pos = 0.0
        failures = []
        solved = {}

        # First determine a single centre grasp, then probe both directions
        # independently. A failure on the negative side must not conceal a
        # good positive workspace (or vice versa).
        neutral_p, neutral_q = target_at(0.0, dx, dy, yaw)
        neutral = solve(neutral_p, neutral_q, neutral_seed)
        result_rows.append({
            "candidate": ci, "dx_m": dx, "dy_m": dy, "tripod_yaw_deg": yaw,
            "pan_deg": 0.0, **{k: v for k, v in neutral.items() if k != "q"},
        })

        if neutral["ok"]:
            solved["+0.000"] = neutral["q"]
            seed_neutral = torch.tensor([neutral["q"]], dtype=init.dtype, device=sim.device)
            for sign in (-1, +1):
                previous = seed_neutral.clone()
                count = math.ceil(args.sweep_deg / args.step_deg)
                for i in range(1, count + 1):
                    angle = sign * args.sweep_deg * i / count
                    world_p, world_q = target_at(angle, dx, dy, yaw)
                    data = solve(world_p, world_q, previous)
                    result_rows.append({
                        "candidate": ci, "dx_m": dx, "dy_m": dy,
                        "tripod_yaw_deg": yaw, "pan_deg": angle,
                        **{k: v for k, v in data.items() if k != "q"},
                    })
                    if not data["ok"]:
                        failures.append((angle, data))
                        break
                    solved[f"{angle:+.3f}"] = data["q"]
                    previous = torch.tensor([data["q"]], dtype=init.dtype, device=sim.device)
                    if sign < 0:
                        achieved_neg = abs(angle)
                    else:
                        achieved_pos = angle
        else:
            failures.append((0.0, neutral))

        reached = min(achieved_neg, achieved_pos)
        min_margin = min(
            min(data["elbow_deg"] - ELBOW_MIN_DEG, ELBOW_MAX_DEG - data["elbow_deg"],
                args.yaw_workspace_deg - abs(data["waist_yaw_deg"]))
            for data in (r for r in result_rows if r["candidate"] == ci and r["ok"])
        ) if solved else -999

        candidate = {
            "candidate": ci, "tripod_offset_x_m": dx, "tripod_offset_y_m": dy,
            "tripod_base_yaw_deg": yaw, "negative_reached_deg": achieved_neg,
            "positive_reached_deg": achieved_pos, "balanced_reached_deg": reached,
            "min_joint_margin_deg": min_margin, "waypoints": solved,
            "failures": [{"pan_deg": a, **{k: v for k, v in d.items() if k != "q"}}
                         for a, d in failures],
        }

        # Select widest coverage, then prefer joint clearance and smaller reposition.
        score = (reached, min_margin, -(abs(dx) + abs(dy)), -abs(yaw))
        if best is None or score > best[0]:
            best = (score, candidate)
        if not failures and (winner_score is None or score > winner_score):
            winner = candidate
            winner_score = score

        status = "FULL PREFLIGHT" if not failures else "INCOMPLETE"
        print(f"[{ci:02d}/{len(candidates)}] x={dx:+.2f}m y={dy:+.2f}m yaw={yaw:+4.0f}deg "
              f"=> neg={achieved_neg:4.1f}deg pos={achieved_pos:4.1f}deg "
              f"({status}); firstFailure={failures[0][0]:+.1f}deg" if failures else
              f"[{ci:02d}/{len(candidates)}] x={dx:+.2f}m y={dy:+.2f}m yaw={yaw:+4.0f}deg "
              f"=> neg={achieved_neg:4.1f}deg pos={achieved_pos:4.1f}deg ({status})")

    csv_path = Path(args.csv).resolve()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["candidate", "dx_m", "dy_m", "tripod_yaw_deg", "pan_deg", "ok",
                  "position_error_m", "orientation_error_deg", "elbow_deg",
                  "waist_yaw_deg", "elbow_out_m", "fk_tracking_gap_deg"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    chosen = winner if winner is not None else best[1]
    output = {
        "kind": "RAVEN_G1_TRIPOD_WORKSPACE_PREFLIGHT",
        "result": "KINEMATIC_CANDIDATE_NOT_PHYSICALLY_VALIDATED" if winner else "NO_FULL_RANGE_CANDIDATE",
        "target_sweep_deg": args.sweep_deg, "joint_order": list(IK_NAMES),
        "nominal_tripod_x_m": TRIPOD_X,
        "robot_tripod_geometry": "Candidate XY shifts and base yaw relative to calibrated tripod",
        "torso_check": "lateral elbow clearance only, NOT a collision-mesh check",
        "grasp_assumption": "original calibrated rigid hand-to-handle pose preserved",
        "selected_candidate": chosen,
    }
    json_path = Path(args.output).resolve()
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("[RAVEN] " + output["result"])
    print(f"[RAVEN] Best candidate: x-offset={chosen['tripod_offset_x_m']:+.2f} m, "
          f"y-offset={chosen['tripod_offset_y_m']:+.2f} m, "
          f"tripod yaw={chosen['tripod_base_yaw_deg']:+.1f} deg")
    print(f"[RAVEN] Confirmed sampled coverage: -{chosen['negative_reached_deg']:.1f} / "
          f"+{chosen['positive_reached_deg']:.1f} deg")
    print(f"[RAVEN] CSV: {csv_path}")
    print(f"[RAVEN] JSON: {json_path}")
    print("[RAVEN] No camera was panned: this was a kinematic geometry study.")
    print("[RAVEN] Physical validation and collision-mesh checking are still required.")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cpu", use_fabric=False))
    sim.set_camera_view(eye=(2.1, 1.8, 1.55), target=(0.35, 0.10, 0.85))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    solve_workspace(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
