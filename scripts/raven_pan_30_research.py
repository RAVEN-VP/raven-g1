"""RAVEN: posture-stable 30-degree staged physical pan research test.

Corrects the observed v1 failure at +10 degrees (elbow switched configuration,
waist roll and pitch hit their workspace bounds, pan fell back from +7 to +2).
The temporary USD spherical joint is unchanged so this test isolates the IK
change: the 10-DOF solver now tracks hand POSITION + ORIENTATION, with a modest
orientation compliance factor, rather than position alone. Wrist is still
free to rotate in the physics attachment if external loads require it.

Adds actual-joint saturation watchdogs, records orientation tracking error,
and refuses to pass unsafe stages. The pan joint is never actively driven
once released; the hand/waist actuators move the handle.

Experimental and NOT yet validated in Isaac Sim on the user's scene.
Not a demonstration of unaided BrainCo contact grip or full collision safety.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RAVEN 30-degree workspace research test")
parser.add_argument("--max-sweep-deg", type=float, default=30.0)
parser.add_argument("--waist-yaw-deg", type=float, default=60.0)
parser.add_argument("--waist-roll-deg", type=float, default=10.0)
parser.add_argument("--waist-pitch-deg", type=float, default=10.0)
parser.add_argument("--log-csv", default="raven_pan_30_telemetry_v2.csv")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 10 <= args.max_sweep_deg <= 30:
    parser.error("--max-sweep-deg must be between 10 and 30")
if not 24 <= args.waist_yaw_deg <= 90:
    parser.error("--waist-yaw-deg must be between 24 and 90")
if not 0 <= args.waist_roll_deg <= 15 or not 0 <= args.waist_pitch_deg <= 15:
    parser.error("waist roll/pitch limits must be in [0, 15] degrees")
simulation_app = AppLauncher(args).app

import torch
import omni.usd
from pxr import Gf, Sdf, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul, subtract_frame_transforms
from isaaclab_assets import G1_INSPIRE_FTP_CFG
from raven_g1.assets import G1_BRAINCO_MODE15_CFG, TRIPOD_CAMERA_CFG

TRIPOD_X = 0.61  # Keep existing working calibration unchanged in this experiment.
PHYSICS_HZ = 120.0
PAN_LOCK_STIFFNESS = 500.0
PAN_LOCK_DAMPING = 30.0
PAN_FREE_DAMPING = 0.45
PAN_WORLD_SIGN = +1.0  # Empirically converged in the last working sweep.
PAN_KP = 1.5
PAN_KD = 0.12
MAX_LEAD_DEG = 6.0
RAMP_S = 2.8
HOLD_S = 0.30
TIMEOUT_S = 7.0
TARGET_ERROR_TOL_DEG = 0.85
SETTLE_SPEED_TOL_DEG_S = 3.5
WAIST_EMERGENCY_MARGIN_DEG = 3.0
ELBOW_OPERATIONAL_MIN_DEG = 94.0
ELBOW_OPERATIONAL_MAX_DEG = 118.0
ELBOW_EMERGENCY_MIN_DEG = 89.0
ELBOW_SATURATION_LOW_DEG = 95.0
ELBOW_SATURATION_HIGH_DEG = 119.0
WAIST_SATURATION_MARGIN_DEG = 0.6
JOINT_SATURATION_TIME_S = 0.30
ORIENTATION_TRACKING_BLEND = 0.60
MAX_ORIENTATION_ERROR_DEG = 30.0
MAX_ASSIST_POS_ERROR_M = 0.008
MIN_ELBOW_LATERAL_M = 0.10  # Only a proxy, NOT collision-mesh clearance.
JOINT_STEP_RAD = 0.012
WAIST_STEP_RAD = 0.008
WAIST_ROLL_PITCH_STEP_RAD = 0.0035
POSTURE_GAIN = 0.07
POSTURE_MAX_STEP = 0.00065
ELBOW_GOAL_DEG = 108.0

MOVE_S, PRE_CLOSE_S, CLOSE_S, GRASP_SETTLE_S = 8.0, 2.0, 6.0, 2.0
RELEASE_S, FREE_SETTLE_S = 2.5, 1.5
GRASP_READY_S = MOVE_S + PRE_CLOSE_S + CLOSE_S + GRASP_SETTLE_S
PAN_RELEASE_S = GRASP_READY_S + RELEASE_S
PAN_READY_S = PAN_RELEASE_S + FREE_SETTLE_S

ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
)
WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
IK_NAMES = WAIST_NAMES + ARM_NAMES
ARM_GRASP = (-1.10463631, 0.45274511, 0.89249939, 2.09439969, 0.67981434, -0.22342572, -0.18944184)
HAND_CLOSED = {
    "left_thumb_metacarpal_joint": 0.40,
    "left_thumb_proximal_joint": 0.48,
    "left_thumb_distal_joint": 0.48,
    "left_index_proximal_joint": 0.60,
    "left_index_distal_joint": 0.60 * 1.155,
    "left_middle_proximal_joint": 0.60,
    "left_middle_distal_joint": 0.60 * 1.155,
    "left_ring_proximal_joint": 0.60,
    "left_ring_distal_joint": 0.60 * 1.155,
    "left_pinky_proximal_joint": 0.60,
    "left_pinky_distal_joint": 0.60 * 1.155,
}

robot_cfg = G1_BRAINCO_MODE15_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
robot_cfg.actuators = copy.deepcopy(G1_INSPIRE_FTP_CFG.actuators)
robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_pinky_.*"],
    effort_limit_sim=45.0, velocity_limit_sim=10.0, stiffness=30.0, damping=2.0, armature=0.001,
)
tripod_cfg = TRIPOD_CAMERA_CFG.copy()
tripod_cfg.prim_path = "{ENV_REGEX_NS}/Tripod"
p = tripod_cfg.init_state.pos
tripod_cfg.init_state.pos = (TRIPOD_X, p[1], p[2])
tripod_cfg.actuators = {
    "pan_lock_release": ImplicitActuatorCfg(
        joint_names_expr=["pan_joint"], effort_limit_sim=100, velocity_limit_sim=5,
        stiffness=PAN_LOCK_STIFFNESS, damping=PAN_LOCK_DAMPING, friction=0.02,
    ),
    "tilt_lock": ImplicitActuatorCfg(
        joint_names_expr=["tilt_joint"], effort_limit_sim=100, velocity_limit_sim=5,
        stiffness=500, damping=30, friction=0.10,
    ),
}

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2500, color=(0.8, 0.8, 0.8)))
    robot = robot_cfg
    tripod = tripod_cfg


def smoothstep(value: float) -> float:
    a = min(1.0, max(0.0, value))
    return a * a * (3 - 2 * a)


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    q = torch.zeros((angle.shape[0], 4), dtype=angle.dtype, device=angle.device)
    q[:, 0] = torch.cos(0.5 * angle)
    q[:, 3] = torch.sin(0.5 * angle)
    return q


def col(value: float, device: str) -> torch.Tensor:
    return torch.tensor([[value]], device=device, dtype=torch.float32)


def attach_position_only(hand_path: str, handle_path: str, handle_to_hand_pos: torch.Tensor) -> None:
    """Temporary spherical joint: same grasp POINT, unconstrained relative rotation."""
    stage = omni.usd.get_context().get_stage()
    for path in (hand_path, handle_path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Expected a rigid-body prim for temporary assisted attachment: {path}")
    path = "/World/RAVEN_AssistedSphericalGrasp"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    joint = UsdPhysics.SphericalJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(handle_path)])
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    pos = handle_to_hand_pos[0].detach().cpu().tolist()
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, pos)))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateJointEnabledAttr(True)
    print("[RAVEN] Attached POSITION-ONLY spherical grasp (rotations intentionally free)")
    print("[RAVEN] This remains a virtual assisted grasp; finger-contact validity is NOT demonstrated")


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot, tripod = scene["robot"], scene["tripod"]
    dt = sim.get_physics_dt()
    rn = {n: i for i, n in enumerate(robot.data.joint_names)}
    rb = {n: i for i, n in enumerate(robot.data.body_names)}
    tn = {n: i for i, n in enumerate(tripod.data.joint_names)}
    tb = {n: i for i, n in enumerate(tripod.data.body_names)}
    ik_ids = [rn[n] for n in IK_NAMES]
    arm_ids = [rn[n] for n in ARM_NAMES]
    fingers_ids = [rn[n] for n in HAND_CLOSED]
    hand_body, elbow_body, torso_body = (rb[n] for n in ("left_base_link", "left_elbow_link", "torso_link"))
    hand_jac_idx = hand_body - 1 if robot.is_fixed_base else hand_body
    pan_id, tilt_id = tn["pan_joint"], tn["tilt_joint"]
    pan_body, handle_body = tb["pan_link"], tb["control_handle_link"]
    init = robot.data.default_joint_pos.clone()
    grasp_arm = torch.tensor(ARM_GRASP, device=sim.device, dtype=init.dtype).unsqueeze(0)
    closed = torch.tensor(list(HAND_CLOSED.values()), device=sim.device, dtype=init.dtype).unsqueeze(0)
    limits = robot.data.joint_pos_limits[:, ik_ids, :].clone()
    waist_home = init[0, [rn[n] for n in WAIST_NAMES]].clone()
    waist_limits_deg = (args.waist_yaw_deg, args.waist_roll_deg, args.waist_pitch_deg)
    for i, deg in enumerate(waist_limits_deg):
        v = math.radians(deg)
        limits[:, i, 0] = torch.maximum(limits[:, i, 0], waist_home[i] - v)
        limits[:, i, 1] = torch.minimum(limits[:, i, 1], waist_home[i] + v)
    eidx = IK_NAMES.index("left_elbow_joint")
    limits[:, eidx, 0] = max(math.radians(ELBOW_OPERATIONAL_MIN_DEG), limits[0, eidx, 0].item())
    limits[:, eidx, 1] = min(math.radians(ELBOW_OPERATIONAL_MAX_DEG), limits[0, eidx, 1].item())
    if bool((limits[..., 0] >= limits[..., 1]).any()):
        raise RuntimeError("No valid joint limits after requested workspace configuration")

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                    ik_method="dls", ik_params={"lambda_val": 0.045}),
        num_envs=1, device=sim.device,
    )
    grip_pos = None
    grip_quat = None
    grasp_link_attached = False
    posture_ref = None
    saturation_seconds = 0.0
    pan_zero = None
    last_phase = None
    step = 0
    stage_index = 0
    stage_start = None
    stage_initial_deg = 0.0
    hold_time = 0.0
    prev_angle = None
    velocity_lpf = 0.0
    stage_result = "INCOMPLETE"
    stage_levels = [x for x in (10, 15, 20, 25, 30) if x <= args.max_sweep_deg + 0.01]
    targets: list[float] = []
    for level in stage_levels:
        targets.extend((-float(level), 0.0, float(level), 0.0))
    if not targets:
        raise RuntimeError("No test targets available")

    def desired_hand_pose(lead_joint_rad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Target the calibrated hand pose carried around the passive pan axis."""
        h = tripod.data.body_pose_w[:, handle_body]
        pivot = tripod.data.body_pose_w[:, pan_body, :3]
        rotation = yaw_quat(PAN_WORLD_SIGN * lead_joint_rad)
        lead_pos = pivot + quat_apply(rotation, h[:, :3] - pivot)
        lead_quat = quat_mul(rotation, h[:, 3:7])
        return combine_frame_transforms(lead_pos, lead_quat, grip_pos, grip_quat)

    def control_hand(target_world_pos: torch.Tensor, target_world_quat: torch.Tensor) -> torch.Tensor:
        """6D primary IK + projected posture holding; no unbounded null drift."""
        hand = robot.data.body_pose_w[:, hand_body]
        base = robot.data.root_pose_w
        target_pos_base, target_quat_base = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], target_world_pos, target_world_quat
        )
        current_pos_base, current_quat_base = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], hand[:, :3], hand[:, 3:7]
        )
        # A spherical attachment permits rotation; ASK the IK to keep the
        # original human-like wrist orientation, rather than leaving it free.
        # Partial quaternion interpolation avoids over-stiff rotational motion.
        aligned_target_q = torch.where(
            torch.sum(current_quat_base * target_quat_base, dim=-1, keepdim=True) < 0.0,
            -target_quat_base, target_quat_base,
        )
        request_quat = torch.nn.functional.normalize(
            (1.0 - ORIENTATION_TRACKING_BLEND) * current_quat_base
            + ORIENTATION_TRACKING_BLEND * aligned_target_q,
            dim=-1,
        )
        ik.set_command(torch.cat((target_pos_base, request_quat), dim=-1))
        j = robot.root_physx_view.get_jacobians()[:, hand_jac_idx, :, ik_ids]
        q = robot.data.joint_pos[:, ik_ids]
        proposed = ik.compute(current_pos_base, current_quat_base, j, q)

        # Project posture preference into the SIX-dimensional hand-pose null
        # space. The original v1 projected only position (3D), which let
        # waist roll/pitch and the wrist twist into a different IK solution.
        pinv = torch.linalg.pinv(j, rtol=1.0e-4)
        null = (torch.eye(len(IK_NAMES), dtype=q.dtype, device=sim.device)
                .unsqueeze(0) - torch.bmm(pinv, j))
        bias = posture_ref - q
        weights = torch.full_like(bias, 0.45)
        weights[:, 0] = 0.25  # waist yaw may move for a wide shot
        weights[:, 1:3] = 2.5  # discourage torso roll and pitch
        weights[:, eidx] = 2.0  # keep elbow near calibrated 108 degrees
        secondary = torch.bmm(null, (weights * bias).unsqueeze(-1)).squeeze(-1)
        secondary = torch.clamp(POSTURE_GAIN * secondary,
                                -POSTURE_MAX_STEP, POSTURE_MAX_STEP)
        proposed = proposed + secondary

        max_step = torch.full_like(q, JOINT_STEP_RAD)
        max_step[:, 0] = WAIST_STEP_RAD
        max_step[:, 1:3] = WAIST_ROLL_PITCH_STEP_RAD
        next_q = q + torch.clamp(proposed - q, -max_step, max_step)
        return torch.maximum(torch.minimum(next_q, limits[..., 1]), limits[..., 0])

    log_path = Path(args.log_csv).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("time_s", "target_deg", "requested_deg", "actual_pan_deg", "velocity_deg_s",
              "waist_yaw_deg", "waist_roll_deg", "waist_pitch_deg", "elbow_deg",
              "elbow_lateral_m", "assist_error_m", "hand_orientation_error_deg",
              "joint_saturation_time_s", "stage_index")
    print("\n" + "=" * 74)
    print(" RAVEN - STAGED +/-30 DEGREE PAN / POSTURE-STABLE 6D HAND IK")
    print("=" * 74)
    print(f"Waist yaw +/-{args.waist_yaw_deg:g}, roll +/-{args.waist_roll_deg:g}, pitch +/-{args.waist_pitch_deg:g} deg")
    print("Stages:", " -> ".join(f"+/-{n}" for n in stage_levels))
    print("Camera pan stays passive; tilt locked; no finger-only grip claim")
    print("Correction: 6D hand-pose IK, 6D null-space posture, joint-saturation watchdog")
    print("If a stage cannot reach safely, the experiment STOPS instead of faking a pass")
    print(f"CSV telemetry: {log_path}\n")

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        while simulation_app.is_running():
            t = step * dt
            desired_joint_targets = init.clone()
            phase = "APPROACH"
            if t < MOVE_S:
                a = smoothstep(t / MOVE_S)
                desired_joint_targets[:, arm_ids] = init[:, arm_ids] + a * (grasp_arm - init[:, arm_ids])
                desired_joint_targets[:, fingers_ids] = init[:, fingers_ids]
            elif t < MOVE_S + PRE_CLOSE_S:
                phase = "PRE_CLOSE"
                desired_joint_targets[:, arm_ids] = grasp_arm
                desired_joint_targets[:, fingers_ids] = init[:, fingers_ids]
            elif t < MOVE_S + PRE_CLOSE_S + CLOSE_S:
                phase = "CLOSING"
                a = smoothstep((t - MOVE_S - PRE_CLOSE_S) / CLOSE_S)
                desired_joint_targets[:, arm_ids] = grasp_arm
                desired_joint_targets[:, fingers_ids] = init[:, fingers_ids] + a * (closed - init[:, fingers_ids])
            elif t < GRASP_READY_S:
                phase = "LOCKED_SETTLE"
                desired_joint_targets[:, arm_ids] = grasp_arm
                desired_joint_targets[:, fingers_ids] = closed
            else:
                desired_joint_targets[:, fingers_ids] = closed
                if not grasp_link_attached:
                    h = tripod.data.body_pose_w[:, handle_body].clone()
                    hand = robot.data.body_pose_w[:, hand_body].clone()
                    grip_pos, grip_quat = subtract_frame_transforms(
                        h[:, :3], h[:, 3:7], hand[:, :3], hand[:, 3:7])
                    attach_position_only(
                        "/World/envs/env_0/Robot/left_base_link",
                        "/World/envs/env_0/Tripod/control_handle_link",
                        grip_pos,
                    )
                    posture_ref = robot.data.joint_pos[:, ik_ids].clone()
                    posture_ref[:, eidx] = math.radians(ELBOW_GOAL_DEG)
                    for i in (1, 2):
                        posture_ref[:, i] = waist_home[i]
                    grasp_link_attached = True
                    print("[RAVEN] GRASP READY. Starting controlled pan release")
                zero = torch.zeros((1,), dtype=torch.float32, device=sim.device)
                if t < PAN_RELEASE_S:
                    phase = "SOFT_RELEASE"
                    a = smoothstep((t - GRASP_READY_S) / RELEASE_S)
                    stiffness = PAN_LOCK_STIFFNESS * (1.0 - a)
                    damping = PAN_LOCK_DAMPING + (PAN_FREE_DAMPING - PAN_LOCK_DAMPING) * a
                    tripod.write_joint_stiffness_to_sim(col(stiffness, sim.device), joint_ids=[pan_id])
                    tripod.write_joint_damping_to_sim(col(damping, sim.device), joint_ids=[pan_id])
                    desired_joint_targets[:, ik_ids] = control_hand(*desired_hand_pose(zero))
                elif t < PAN_READY_S:
                    phase = "FREE_SETTLE"
                    tripod.write_joint_stiffness_to_sim(col(0.0, sim.device), joint_ids=[pan_id])
                    tripod.write_joint_damping_to_sim(col(PAN_FREE_DAMPING, sim.device), joint_ids=[pan_id])
                    desired_joint_targets[:, ik_ids] = control_hand(*desired_hand_pose(zero))
                else:
                    phase = "ACTIVE_PAN"
                    tripod.write_joint_stiffness_to_sim(col(0.0, sim.device), joint_ids=[pan_id])
                    tripod.write_joint_damping_to_sim(col(PAN_FREE_DAMPING, sim.device), joint_ids=[pan_id])
                    if pan_zero is None:
                        pan_zero = tripod.data.joint_pos[:, pan_id].clone()
                        physical_margin = math.radians(args.max_sweep_deg + 4.0)
                        bounds = torch.stack((pan_zero - physical_margin, pan_zero + physical_margin), dim=-1).unsqueeze(1)
                        tripod.write_joint_position_limit_to_sim(bounds, joint_ids=[pan_id])
                        stage_start = t
                        stage_initial_deg = 0.0
                        print(f"[RAVEN] PAN FREE. zero={math.degrees(pan_zero[0].item()):+.2f} deg; "+
                              f"PHYSICAL LIMIT +/-{args.max_sweep_deg + 4.0:.1f} deg")
                        print(f"[RAVEN] FIRST TARGET {targets[0]:+.1f} deg")
                    actual = math.degrees((tripod.data.joint_pos[0, pan_id] - pan_zero[0]).item())
                    target = targets[stage_index]
                    elapsed = t - stage_start
                    request = stage_initial_deg + (target - stage_initial_deg) * smoothstep(elapsed / RAMP_S)
                    if prev_angle is None:
                        prev_angle = actual
                    raw_velocity = (actual - prev_angle) / dt
                    prev_angle = actual
                    velocity_lpf = 0.93 * velocity_lpf + 0.07 * raw_velocity
                    actual_waist = robot.data.joint_pos[0, [rn[n] for n in WAIST_NAMES]]
                    waist_deg = [math.degrees((actual_waist[i] - waist_home[i]).item()) for i in range(3)]
                    elbow_deg = math.degrees(robot.data.joint_pos[0, rn["left_elbow_joint"]].item())
                    torso = robot.data.body_pose_w[:, torso_body]
                    elbow = robot.data.body_pose_w[:, elbow_body]
                    axis_y = torch.tensor([[0.0, 1.0, 0.0]], device=sim.device)
                    outward_y = quat_apply(torso[:, 3:7], axis_y)
                    elbow_clearance = float(torch.sum((elbow[:, :3] - torso[:, :3]) * outward_y, dim=-1)[0].item())
                    h = tripod.data.body_pose_w[:, handle_body]
                    hand = robot.data.body_pose_w[:, hand_body]
                    expected_pos, _ = combine_frame_transforms(h[:, :3], h[:, 3:7], grip_pos, None)
                    assist_error = float(torch.linalg.vector_norm(hand[:, :3] - expected_pos, dim=-1)[0].item())
                    actual_hand_quat = hand[:, 3:7]
                    expected_hand_quat = quat_mul(h[:, 3:7], grip_quat)
                    quat_dot = torch.sum(actual_hand_quat * expected_hand_quat, dim=-1).abs().clamp(0, 1)
                    orient_error_deg = math.degrees(2.0 * torch.acos(quat_dot)[0].item())
                    near_joint_limit = (
                        abs(waist_deg[1]) >= args.waist_roll_deg - WAIST_SATURATION_MARGIN_DEG
                        or abs(waist_deg[2]) >= args.waist_pitch_deg - WAIST_SATURATION_MARGIN_DEG
                        or elbow_deg < ELBOW_SATURATION_LOW_DEG
                        or elbow_deg > ELBOW_SATURATION_HIGH_DEG
                    )
                    saturation_seconds = (saturation_seconds + dt) if near_joint_limit else 0.0
                    error = request - actual
                    lead_rad = math.radians(PAN_KP * error - PAN_KD * velocity_lpf)
                    lead = torch.tensor([max(-math.radians(MAX_LEAD_DEG), min(math.radians(MAX_LEAD_DEG), lead_rad))], device=sim.device)
                    safety = None
                    if abs(waist_deg[0]) >= args.waist_yaw_deg + WAIST_EMERGENCY_MARGIN_DEG:
                        safety = "waist yaw workspace exceeded"
                    if abs(waist_deg[1]) > args.waist_roll_deg + 3.0 or abs(waist_deg[2]) > args.waist_pitch_deg + 3.0:
                        safety = "waist roll/pitch workspace exceeded"
                    if elbow_deg < ELBOW_EMERGENCY_MIN_DEG:
                        safety = "elbow below emergency flexion floor"
                    if saturation_seconds >= JOINT_SATURATION_TIME_S:
                        safety = ("sustained joint saturation: "
                                  f"waist=({waist_deg[0]:+.1f},{waist_deg[1]:+.1f},{waist_deg[2]:+.1f}), "
                                  f"elbow={elbow_deg:.1f}")
                    if orient_error_deg > MAX_ORIENTATION_ERROR_DEG:
                        safety = f"uncontrolled hand rotation ({orient_error_deg:.1f} deg)"
                    if assist_error > MAX_ASSIST_POS_ERROR_M:
                        safety = "assisted-grasp positional separation"
                    if elbow_clearance < MIN_ELBOW_LATERAL_M:
                        safety = "elbow near torso proxy threshold"
                    if abs(actual) > args.max_sweep_deg + 4.5:
                        safety = "tripod pan exceeded physical envelope"
                    if safety is not None:
                        stage_result = "FAIL"
                        print(f"[RAVEN] SAFETY ABORT at target={target:+.1f}, actual={actual:+.2f}: {safety}")
                        break
                    desired_joint_targets[:, ik_ids] = control_hand(*desired_hand_pose(lead))
                    if step % 12 == 0:
                        writer.writerow({
                            "time_s": f"{t:.4f}", "target_deg": f"{target:.2f}", "requested_deg": f"{request:.3f}",
                            "actual_pan_deg": f"{actual:.3f}", "velocity_deg_s": f"{velocity_lpf:.3f}",
                            "waist_yaw_deg": f"{waist_deg[0]:.3f}", "waist_roll_deg": f"{waist_deg[1]:.3f}",
                            "waist_pitch_deg": f"{waist_deg[2]:.3f}", "elbow_deg": f"{elbow_deg:.3f}",
                            "elbow_lateral_m": f"{elbow_clearance:.4f}", "assist_error_m": f"{assist_error:.5f}",
                            "hand_orientation_error_deg": f"{orient_error_deg:.2f}",
                            "joint_saturation_time_s": f"{saturation_seconds:.3f}",
                            "stage_index": stage_index,
                        })
                    if step % 120 == 0:
                        print(f"[PAN] target={target:+5.1f} | requested={request:+6.1f} | actual={actual:+6.2f} | "+
                              f"velocity={velocity_lpf:+6.2f} | waistYaw={waist_deg[0]:+5.1f} | "+
                              f"waistRoll={waist_deg[1]:+4.1f} | waistPitch={waist_deg[2]:+4.1f} | "+
                              f"elbow={elbow_deg:5.1f} | clearance={elbow_clearance:.3f}m | " +
                              f"grasp={assist_error:.4f}m | handRotErr={orient_error_deg:.1f}deg | " +
                              f"satFor={saturation_seconds:.2f}s")
                    arrived = abs(actual - target) < TARGET_ERROR_TOL_DEG and abs(velocity_lpf) < SETTLE_SPEED_TOL_DEG_S
                    hold_time = hold_time + dt if arrived and elapsed >= RAMP_S else 0.0
                    if hold_time >= HOLD_S:
                        print(f"[RAVEN] PASS target={target:+.1f} actual={actual:+.2f}")
                        stage_index += 1
                        if stage_index == len(targets):
                            stage_result = "PASS"
                            print("[RAVEN] ALL STAGES PASSED: +/-30 degree physical pan candidate")
                            break
                        stage_start = t
                        stage_initial_deg = actual
                        hold_time = 0.0
                        saturation_seconds = 0.0
                        print(f"[RAVEN] NEXT TARGET {targets[stage_index]:+.1f} deg")
                    elif elapsed > TIMEOUT_S:
                        stage_result = "FAIL"
                        print(f"[RAVEN] STAGE FAILED: target={target:+.1f}, actual={actual:+.2f}, "+
                              f"err={target-actual:+.2f}; halting without claiming +/-30")
                        break
            if phase != last_phase:
                print(f"[RAVEN] Phase -> {phase}")
                last_phase = phase
            robot.set_joint_position_target(desired_joint_targets)
            tripod.set_joint_position_target(col(0.0, sim.device), joint_ids=[tilt_id])
            if t < PAN_RELEASE_S:
                tripod.set_joint_position_target(col(0.0, sim.device), joint_ids=[pan_id])
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            step += 1
    print(f"[RAVEN] Result: {stage_result}")
    print(f"[RAVEN] Telemetry written: {log_path}")


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, device="cpu", use_fabric=False))
    sim.set_camera_view(eye=(2.1, 1.8, 1.55), target=(0.35, 0.10, 0.85))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    try:
        run(sim, scene)
    finally:
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath("/World/RAVEN_AssistedSphericalGrasp").IsValid():
            stage.RemovePrim("/World/RAVEN_AssistedSphericalGrasp")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
