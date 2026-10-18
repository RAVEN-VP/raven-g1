"""RAVEN: physics-assisted tripod-pan proof of concept (Isaac Lab 2.3.2 / Sim 5.1).

A temporary USD FixedJoint connects the G1 left hand base to the tripod handle
*after* the calibrated grasp. That assists gripping; it is NOT evidence that
Revo2 finger contact alone can hold the handle.

The only source of tripod pan motion is G1 actuator action transmitted through
this joint. The tripod pan actuator's stiffness is zero during the test.
Tilt stays locked. CPU PhysX is intentional: runtime cross-articulation joints
are unreliable on some GPU-PhysX builds. Renderer may still use the NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import copy
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RAVEN assisted-grasp physical pan test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import torch
import omni.usd
from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_assets import G1_INSPIRE_FTP_CFG
from raven_g1.assets import G1_BRAINCO_MODE15_CFG, TRIPOD_CAMERA_CFG

# --- Physical scene and test design -------------------------------------------
TRIPOD_X = 0.61
PHYSICS_HZ = 120.0

# Effective POC range, NOT the final cinematography FOV.
TARGETS_DEG = (8.0, 0.0, -8.0, 0.0)
TARGET_TIMEOUT_S = 3.0
AT_TARGET_TOL_DEG = 0.7
AT_TARGET_HOLD_S = 0.20

# The *physical* pan joint limits below will be set relative to its angle
# after grasp/release, not relative to the URDF's absolute zero.
PAN_PHYSICAL_LIMIT_DEG = 11.0

# Only the hand's lead (not the pan joint) is requested by the controller.
PAN_KP = 1.5
PAN_KD = 0.10
MAX_LEAD_DEG = 6.0
PAN_WORLD_SIGN = +1.0  # last converging runs used this convention
PAN_FREE_DAMPING = 0.45
PAN_LOCK_STIFFNESS = 500.0
PAN_LOCK_DAMPING = 30.0

# Keep G1 waist yaw modest. The calibrated elbow is already near its upper limit.
WAIST_YAW_LIMIT_DEG = 12.0
MAX_WAIST_STEP = 0.007
MAX_ARM_STEP = 0.012
MAX_HAND_POSITION_ERROR_M = 0.065

# Simulated times (seconds).
MOVE_S = 8.0
PRE_CLOSE_S = 2.0
CLOSE_S = 6.0
GRASP_SETTLE_S = 2.0
RELEASE_S = 2.5
FREE_SETTLE_S = 1.0
GRASP_READY_S = MOVE_S + PRE_CLOSE_S + CLOSE_S + GRASP_SETTLE_S
PAN_RELEASE_S = GRASP_READY_S + RELEASE_S
PAN_START_S = PAN_RELEASE_S + FREE_SETTLE_S

ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
IK_NAMES = ("waist_yaw_joint",) + ARM_NAMES
ARM_CALIBRATED = (
    -1.10463631, +0.45274511, +0.89249939, +2.09439969,
    +0.67981434, -0.22342572, -0.18944184,
)
HAND_CLOSE = {
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
# Stock Inspire configuration has compliant fingers intended for a different
# hand. The temporary weld carries the grasp load; a modest hand drive helps
# fingers remain visually wrapped around the shaft.
robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_index_.*", ".*_middle_.*", ".*_thumb_.*", ".*_ring_.*", ".*_pinky_.*"],
    effort_limit_sim=45.0,
    velocity_limit_sim=10.0,
    stiffness=30.0,
    damping=2.0,
    armature=0.001,
)

tripod_cfg = TRIPOD_CAMERA_CFG.copy()
tripod_cfg.prim_path = "{ENV_REGEX_NS}/Tripod"
old_pos = tripod_cfg.init_state.pos
tripod_cfg.init_state.pos = (TRIPOD_X, old_pos[1], old_pos[2])
tripod_cfg.actuators = {
    "pan_lock_release": ImplicitActuatorCfg(
        joint_names_expr=["pan_joint"],
        effort_limit_sim=100.0,
        velocity_limit_sim=5.0,
        stiffness=PAN_LOCK_STIFFNESS,
        damping=PAN_LOCK_DAMPING,
        friction=0.02,
    ),
    "tilt_lock": ImplicitActuatorCfg(
        joint_names_expr=["tilt_joint"],
        effort_limit_sim=100.0,
        velocity_limit_sim=5.0,
        stiffness=500.0,
        damping=30.0,
        friction=0.10,
    ),
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )
    robot = robot_cfg
    tripod = tripod_cfg


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def yaw_quat(angle: torch.Tensor) -> torch.Tensor:
    q = torch.zeros((angle.shape[0], 4), dtype=angle.dtype, device=angle.device)
    q[:, 0] = torch.cos(0.5 * angle)
    q[:, 3] = torch.sin(0.5 * angle)
    return q


def torch_column(n: int, val: float, device: str) -> torch.Tensor:
    return torch.full((n, 1), val, device=device, dtype=torch.float32)


def create_assisted_grasp_joint(
    hand_path: str,
    handle_path: str,
    handle_to_hand_pos: torch.Tensor,
    handle_to_hand_quat: torch.Tensor,
) -> str:
    """Attach two existing articulation links at their *current* relative pose.

    Joint coordinate frame on body0 = left hand body's origin.
    Joint coordinate frame on body1 = where that hand origin currently lies,
    expressed in the handle frame. Frames coincide in world space, so PhysX
    should not snap either body to a mismatched authored USD transform.
    """
    stage = omni.usd.get_context().get_stage()
    for path in (hand_path, handle_path):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Cannot attach: rigid-body prim not found: {path}")
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Cannot attach: prim is not a rigid body: {path}")

    joint_path = "/World/RAVEN_AssistedGraspJoint"
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(hand_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(handle_path)])
    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)

    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    p = handle_to_hand_pos[0].detach().cpu().tolist()
    q = handle_to_hand_quat[0].detach().cpu().tolist()  # w,x,y,z
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, p)))
    joint.CreateLocalRot1Attr(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    joint.CreateJointEnabledAttr(True)
    print(f"[RAVEN] Physics-assisted attachment: {joint_path}")
    print("[RAVEN] Temporary FIXED JOINT; finger grip alone is NOT being validated.")
    return joint_path


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    robot, tripod = scene["robot"], scene["tripod"]
    dt = sim.get_physics_dt()
    rn = {name: i for i, name in enumerate(robot.data.joint_names)}
    rb = {name: i for i, name in enumerate(robot.data.body_names)}
    tn = {name: i for i, name in enumerate(tripod.data.joint_names)}
    tb = {name: i for i, name in enumerate(tripod.data.body_names)}

    arm_ids = [rn[x] for x in ARM_NAMES]
    ik_ids = [rn[x] for x in IK_NAMES]
    finger_ids = [rn[x] for x in HAND_CLOSE]
    waist_id = rn["waist_yaw_joint"]
    hand_body = rb["left_base_link"]
    handle_body = tb["control_handle_link"]
    pan_link_body = tb["pan_link"]
    pan_id, tilt_id = tn["pan_joint"], tn["tilt_joint"]
    hand_path = "/World/envs/env_0/Robot/left_base_link"
    handle_path = "/World/envs/env_0/Tripod/control_handle_link"

    init = robot.data.default_joint_pos.clone()
    arm_init = init[:, arm_ids].clone()
    hand_init = init[:, finger_ids].clone()
    arm_grasp = torch.tensor(ARM_CALIBRATED, device=sim.device).view(1, -1)
    finger_closed = torch.tensor(list(HAND_CLOSE.values()), device=sim.device).view(1, -1)

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
            ik_params={"lambda_val": 0.04},
        ),
        num_envs=1,
        device=sim.device,
    )
    command = torch.zeros((1, ik.action_dim), device=sim.device)
    jac_id = hand_body - 1 if robot.is_fixed_base else hand_body

    grip_joint = None
    grip_pos, grip_quat = None, None
    pan_zero = None
    target_index = 0
    segment_start = PAN_START_S
    in_tolerance_for = 0.0
    last_log = -1
    phase_old = None
    stopped = False
    last_arm_target = None

    # Use physical joint limits: the calibrated elbow is at its upper limit.
    # Soft limits (usually 90% of physical range) would abruptly yank that
    # elbow away from the calibrated hand pose at the moment of attachment.
    joint_limits = robot.data.joint_pos_limits[:, ik_ids, :]
    # Extra-specific workspace bound for waist yaw, independent of URDF range.
    waist_home = float(init[0, waist_id].item())
    waist_limit_rad = math.radians(WAIST_YAW_LIMIT_DEG)

    def ik_target(world_pos: torch.Tensor, world_quat: torch.Tensor) -> torch.Tensor:
        base = robot.data.root_pose_w
        current = robot.data.body_pose_w[:, hand_body]
        target_p, target_q = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], world_pos, world_quat
        )
        current_p, current_q = subtract_frame_transforms(
            base[:, :3], base[:, 3:7], current[:, :3], current[:, 3:7]
        )
        command[:, :3] = target_p
        command[:, 3:7] = target_q
        ik.set_command(command)
        jac = robot.root_physx_view.get_jacobians()[:, jac_id, :, ik_ids]
        current_joint = robot.data.joint_pos[:, ik_ids]
        desired_joint = ik.compute(current_p, current_q, jac, current_joint)
        max_steps = torch.full_like(current_joint, MAX_ARM_STEP)
        max_steps[:, 0] = MAX_WAIST_STEP
        next_joint = current_joint + torch.clamp(desired_joint - current_joint, -max_steps, max_steps)
        next_joint = torch.maximum(torch.minimum(next_joint, joint_limits[..., 1]), joint_limits[..., 0])
        next_joint[:, 0] = next_joint[:, 0].clamp(
            waist_home - waist_limit_rad, waist_home + waist_limit_rad
        )
        return next_joint

    def hand_pose_for_handle_lead(lead_joint: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        handle_pose = tripod.data.body_pose_w[:, handle_body]
        pivot = tripod.data.body_pose_w[:, pan_link_body, :3]
        qlead = yaw_quat(PAN_WORLD_SIGN * lead_joint)
        handle_pos = pivot + quat_apply(qlead, handle_pose[:, :3] - pivot)
        handle_quat = quat_mul(qlead, handle_pose[:, 3:7])
        return combine_frame_transforms(handle_pos, handle_quat, grip_pos, grip_quat)

    print("\n" + "=" * 74)
    print(" RAVEN — PHYSICS-ASSISTED G1 CAMERA PAN + WAIST YAW")
    print("=" * 74)
    print(f"CPU PhysX | {PHYSICS_HZ:.0f} Hz | {TARGETS_DEG} deg | physical bounds +/- {PAN_PHYSICAL_LIMIT_DEG} deg")
    print("The pan motor will NOT be driven; the temporary hand-handle joint will.")
    print("WARNING: this is NOT a test of unaided BrainCo contact grip.\n")

    step = 0
    while simulation_app.is_running():
        t = step * dt
        target = init.clone()
        phase = "APPROACH"

        if t < MOVE_S:
            a = smoothstep(t / MOVE_S)
            target[:, arm_ids] = arm_init + a * (arm_grasp - arm_init)
            target[:, finger_ids] = hand_init
        elif t < MOVE_S + PRE_CLOSE_S:
            phase = "PRE_CLOSE"
            target[:, arm_ids] = arm_grasp
            target[:, finger_ids] = hand_init
        elif t < MOVE_S + PRE_CLOSE_S + CLOSE_S:
            phase = "CLOSING"
            a = smoothstep((t - MOVE_S - PRE_CLOSE_S) / CLOSE_S)
            target[:, arm_ids] = arm_grasp
            target[:, finger_ids] = hand_init + a * (finger_closed - hand_init)
        elif t < GRASP_READY_S:
            phase = "LOCKED_SETTLE"
            target[:, arm_ids] = arm_grasp
            target[:, finger_ids] = finger_closed
        else:
            target[:, finger_ids] = finger_closed
            if grip_joint is None:
                # Capture current PhysX frames, not original/imported USD poses.
                h = tripod.data.body_pose_w[:, handle_body].clone()
                g = robot.data.body_pose_w[:, hand_body].clone()
                grip_pos, grip_quat = subtract_frame_transforms(
                    h[:, :3], h[:, 3:7], g[:, :3], g[:, 3:7]
                )
                grip_joint = create_assisted_grasp_joint(
                    hand_path, handle_path, grip_pos, grip_quat
                )
                ik.reset()
                print("[RAVEN] Grasp attached. Gradually releasing tripod pan drive.")

            if t < PAN_RELEASE_S:
                phase = "SOFT_RELEASE"
                a = smoothstep((t - GRASP_READY_S) / RELEASE_S)
                stiffness = PAN_LOCK_STIFFNESS * (1.0 - a)
                damping = PAN_LOCK_DAMPING * (1.0 - a) + PAN_FREE_DAMPING * a
                tripod.write_joint_stiffness_to_sim(torch_column(1, stiffness, sim.device), joint_ids=[pan_id])
                tripod.write_joint_damping_to_sim(torch_column(1, damping, sim.device), joint_ids=[pan_id])
            else:
                phase = "FREE_SETTLE" if t < PAN_START_S else "PAN_CONTROL"
                tripod.write_joint_stiffness_to_sim(torch_column(1, 0.0, sim.device), joint_ids=[pan_id])
                tripod.write_joint_damping_to_sim(torch_column(1, PAN_FREE_DAMPING, sim.device), joint_ids=[pan_id])

            lead = torch.zeros((1,), device=sim.device)
            if t >= PAN_START_S:
                if pan_zero is None:
                    pan_zero = tripod.data.joint_pos[:, pan_id].clone()
                    # This is an actual PhysX limit, not merely a controller threshold.
                    limit = math.radians(PAN_PHYSICAL_LIMIT_DEG)
                    bounds = torch.stack((pan_zero - limit, pan_zero + limit), dim=-1).unsqueeze(1)
                    tripod.write_joint_position_limit_to_sim(bounds, joint_ids=[pan_id])
                    segment_start = t
                    print(f"[RAVEN] PAN ACTIVE | zero={math.degrees(pan_zero[0].item()):+.2f} deg")
                    print(f"[RAVEN] Physics limits set to +/-{PAN_PHYSICAL_LIMIT_DEG:.1f} deg from that zero")

                desired = float(TARGETS_DEG[target_index])
                actual_rad = tripod.data.joint_pos[:, pan_id] - pan_zero
                actual_deg = math.degrees(actual_rad[0].item())
                velocity = tripod.data.joint_vel[:, pan_id]
                error_deg = desired - actual_deg

                # Hard fail-safe on pan runaway, joint constraint overload or
                # incorrect frames. Freeze commands, do not chase indefinitely.
                if abs(actual_deg) >= PAN_PHYSICAL_LIMIT_DEG + 1.0:
                    stopped = True
                    print(f"[RAVEN] SAFETY STOP: pan={actual_deg:+.2f} outside envelope")

                # Monitor *physical* hand-to-handle relation. Once an assisted
                # joint is present, sustained deviation suggests solver error.
                handle_now = tripod.data.body_pose_w[:, handle_body]
                hand_now = robot.data.body_pose_w[:, hand_body]
                expected_p, _ = combine_frame_transforms(
                    handle_now[:, :3], handle_now[:, 3:7], grip_pos, grip_quat
                )
                grasp_error = torch.linalg.vector_norm(hand_now[:, :3] - expected_p, dim=-1)[0].item()
                if grasp_error > MAX_HAND_POSITION_ERROR_M:
                    stopped = True
                    print(f"[RAVEN] SAFETY STOP: joint separation {grasp_error:.3f} m")

                if not stopped:
                    lead_rad = PAN_KP * (math.radians(desired) - actual_rad) - PAN_KD * velocity
                    lead = lead_rad.clamp(-math.radians(MAX_LEAD_DEG), +math.radians(MAX_LEAD_DEG))

                    if abs(error_deg) < AT_TARGET_TOL_DEG and abs(math.degrees(velocity[0].item())) < 3.0:
                        in_tolerance_for += dt
                    else:
                        in_tolerance_for = 0.0

                    if in_tolerance_for >= AT_TARGET_HOLD_S or t - segment_start >= TARGET_TIMEOUT_S:
                        why = "reached" if in_tolerance_for >= AT_TARGET_HOLD_S else "timeout"
                        print(f"[RAVEN] {why.upper():7s}: target={desired:+.1f}, actual={actual_deg:+.2f}")
                        target_index = (target_index + 1) % len(TARGETS_DEG)
                        segment_start = t
                        in_tolerance_for = 0.0
                        print(f"[RAVEN] NEW TARGET {TARGETS_DEG[target_index]:+.1f} deg")

                second = int(t - PAN_START_S)
                if second != last_log:
                    print(
                        f"[PAN] target={desired:+5.1f} | actual={actual_deg:+6.2f} | "
                        f"vel={math.degrees(velocity[0].item()):+6.2f} deg/s | "
                        f"waist={math.degrees(robot.data.joint_pos[0, waist_id].item()):+5.1f} | "
                        f"assistError={grasp_error:.4f} m | {('STOP' if stopped else 'RUN')}"
                    )
                    last_log = second

            if stopped:
                if last_arm_target is None:
                    last_arm_target = robot.data.joint_pos[:, ik_ids].clone()
                target[:, ik_ids] = last_arm_target
            else:
                desired_p, desired_q = hand_pose_for_handle_lead(lead)
                new_joint_target = ik_target(desired_p, desired_q)
                target[:, ik_ids] = new_joint_target
                last_arm_target = new_joint_target.clone()

        if phase != phase_old:
            print(f"[RAVEN] Phase -> {phase}")
            phase_old = phase

        robot.set_joint_position_target(target)
        tripod.set_joint_position_target(torch_column(1, 0.0, sim.device), joint_ids=[tilt_id])
        if t < PAN_RELEASE_S:
            tripod.set_joint_position_target(torch_column(1, 0.0, sim.device), joint_ids=[pan_id])
        # NO PAN POSITION COMMAND AFTER THE RELEASE.
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step += 1


def main() -> None:
    # CPU PhysX makes runtime USD joints considerably more dependable.
    # RTX rendering continues on the NVIDIA GPU through Isaac Sim's GUI.
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, device="cpu", use_fabric=False)
    )
    sim.set_camera_view(eye=(2.1, 1.8, 1.55), target=(0.35, 0.10, 0.85))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    try:
        run(sim, scene)
    finally:
        # Clean up joint without changing the user's source assets.
        stage = omni.usd.get_context().get_stage()
        if stage and stage.GetPrimAtPath("/World/RAVEN_AssistedGraspJoint").IsValid():
            stage.RemovePrim("/World/RAVEN_AssistedGraspJoint")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
