"""RAVEN: physics-assisted tripod-pan proof of concept (Isaac Lab 2.3.2 / Sim 5.1).

A temporary USD FixedJoint connects the G1 left hand base to the tripod handle
*after* the calibrated grasp. That assists gripping; it is NOT evidence that
Revo2 finger contact alone can hold the handle.

A stationary elbow-out preparation is followed by a bounded POSTURE HOLD
within the primary hand-IK null space. This prevents the arm drifting into a
folded or hyperflexed configuration during repeated tripod sweeps.

Human-like camera movement uses eased angular setpoints rather than
instantaneous target jumps. The tripod pan joint remains passive. An actual elbow angle below 89 degrees aborts the run.

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
parser.add_argument(
    "--sweep-deg", type=float, default=8.0,
    help="Pan setpoint amplitude (6 to 12 degrees, default 8)",
)
parser.add_argument(
    "--waist-limit-deg", type=float, default=24.0,
    help="Waist yaw workspace around default pose (12 to 26 degrees, default 24)",
)
parser.add_argument(
    "--run-sweep", action="store_true",
    help="After elbow-out preparation, release pan and run the assisted +/- sweep",
)
parser.add_argument(
    "--elbow-out-cm", type=float, default=2.0,
    help="Desired outward elbow displacement during preparation ONLY (1-6 cm)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not 6.0 <= args_cli.sweep_deg <= 12.0:
    parser.error("--sweep-deg must be between 6 and 12")
if not 12.0 <= args_cli.waist_limit_deg <= 26.0:
    parser.error("--waist-limit-deg must be between 12 and 26")
if not 1.0 <= args_cli.elbow_out_cm <= 6.0:
    parser.error("--elbow-out-cm must be between 1 and 6")
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
# Test the previously weak, negative direction FIRST. A setpoint change is
# immediate; timeout controls only how long we wait before the next target.
TARGETS_DEG = (-args_cli.sweep_deg, 0.0, args_cli.sweep_deg, 0.0)
# A real operator eases into and out of the pan instead of snapping to angles.
PAN_RAMP_SEC = 1.60
TARGET_HOLD_MIN_S = 0.55
TARGET_TIMEOUT_S = 3.50
AT_TARGET_TOL_DEG = 0.85
AT_TARGET_HOLD_S = 0.25

# The *physical* pan joint limits below will be set relative to its angle
# after grasp/release, not relative to the URDF's absolute zero.
PAN_PHYSICAL_LIMIT_DEG = args_cli.sweep_deg + 5.0

# Only the hand's lead (not the pan joint) is requested by the controller.
PAN_KP = 1.5
PAN_KD = 0.10
MAX_LEAD_DEG = 6.0
PAN_WORLD_SIGN = +1.0  # last converging runs used this convention
PAN_FREE_DAMPING = 0.45
PAN_LOCK_STIFFNESS = 500.0
PAN_LOCK_DAMPING = 30.0

# Keep G1 waist yaw modest. The calibrated elbow is already near its upper limit.
# The previous 12-degree clamp was reached on negative pan. Give the waist
# more room to rotate the torso rather than folding the elbow into the chest.
# Still narrower than full URDF joint travel; the joint is also clamped to
# its imported physical limits by ik_target().
WAIST_YAW_LIMIT_DEG = args_cli.waist_limit_deg
MAX_WAIST_STEP = 0.007
MAX_ARM_STEP = 0.012
MAX_HAND_POSITION_ERROR_M = 0.045

# Keep the welded hand in place, and use the remaining IK redundancy to move
# the elbow away from the torso. No command is ever sent directly to the pan
# joint during the sweep. This elbow change is tested with pan LOCKED first.
ELBOW_OUT_GOAL_M = args_cli.elbow_out_cm / 100.0
ELBOW_PREP_S = 6.0
ELBOW_NULLSPACE_MAX_STEP_RAD = 0.0035  # per 120-Hz physics step
ELBOW_NULLSPACE_GAIN = 0.25
ELBOW_DLS_REGULARIZER = 0.001
ELBOW_HAND_ERROR_CUTOFF_M = 0.022
PAN_LOCK_DRIFT_LIMIT_DEG = 0.75

# This is an OPERATIONAL posture constraint, not the URDF joint limit.
# Validating the previous assisted-grasp run showed elbow ~=98-120 degrees
# throughout a functioning +/-10-degree sweep. The modified elbow-out
# controller allowed it to unwind to negative angles, which is unacceptable.
ELBOW_WORKSPACE_MIN_DEG = 95.0
ELBOW_EMERGENCY_MIN_DEG = 89.0
ELBOW_NEAR_LIMIT_MARGIN_DEG = 3.0

# Keep a consistent arm posture while moving the camera. This is a bounded
# reference-configuration objective, NOT an endlessly increasing elbow-out
# objective. Project it into the primary hand-pose Jacobian null space.
POSTURE_HOLD_GAIN = 0.075
POSTURE_MAX_STEP_RAD = 0.00075
ELBOW_PREFERRED_DEG = 110.0
POSTURE_WEIGHT_WAIST = 1.8
POSTURE_WEIGHT_ELBOW = 1.1

# Simulated times (seconds).
MOVE_S = 8.0
PRE_CLOSE_S = 2.0
CLOSE_S = 6.0
GRASP_SETTLE_S = 2.0
RELEASE_S = 2.5
FREE_SETTLE_S = 1.0
GRASP_READY_S = MOVE_S + PRE_CLOSE_S + CLOSE_S + GRASP_SETTLE_S
ELBOW_PREP_END_S = GRASP_READY_S + ELBOW_PREP_S
PAN_RELEASE_S = ELBOW_PREP_END_S + RELEASE_S
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
    elbow_body = rb["left_elbow_link"]
    torso_body = rb["torso_link"]
    elbow_jac_id = elbow_body - 1 if robot.is_fixed_base else elbow_body
    torso_jac_id = torso_body - 1 if robot.is_fixed_base else torso_body
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
    saturation_announced_for_target = False

    # Use physical joint limits: the calibrated elbow is at its upper limit.
    # Soft limits (usually 90% of physical range) would abruptly yank that
    # elbow away from the calibrated hand pose at the moment of attachment.
    joint_limits = robot.data.joint_pos_limits[:, ik_ids, :]
    # Extra-specific workspace bound for waist yaw, independent of URDF range.
    waist_home = float(init[0, waist_id].item())
    waist_limit_rad = math.radians(WAIST_YAW_LIMIT_DEG)
    waist_physical_lower = float(robot.data.joint_pos_limits[0, waist_id, 0].item())
    waist_physical_upper = float(robot.data.joint_pos_limits[0, waist_id, 1].item())
    elbow_id = rn["left_elbow_joint"]
    elbow_ik_index = IK_NAMES.index("left_elbow_joint")
    elbow_baseline = None
    elbow_goal = None
    elbow_prep_pan_ref = None
    posture_test_last_log = -1
    posture_halt = False
    posture_reported = False
    sweep_disallowed = False
    elbow_upper = float(robot.data.joint_pos_limits[0, elbow_id, 1].item())
    elbow_lower = float(robot.data.joint_pos_limits[0, elbow_id, 0].item())
    if math.radians(ELBOW_WORKSPACE_MIN_DEG) >= elbow_upper:
        raise RuntimeError(
            "Elbow safe operating floor is above the URDF physical upper limit. "
            "Check the joint limit convention before running."
        )
    if math.radians(ELBOW_WORKSPACE_MIN_DEG) <= elbow_lower:
        raise RuntimeError(
            "Elbow safe operating floor must lie inside the physical joint range."
        )
    elbow_operating_lower = math.radians(ELBOW_WORKSPACE_MIN_DEG)
    posture_reference = None
    posture_hold_initialized = False
    segment_start_angle_deg = None
    filtered_pan_vel = torch.zeros((1,), device=sim.device)
    previous_pan_angle = None

    def elbow_out_distance_and_gradient() -> tuple[torch.Tensor, torch.Tensor]:
        """Return left-elbow lateral clearance from torso and its joint gradient.

        +Y of the G1 torso is outward for the left arm. The gradient includes
        the torso's translational/rotational Jacobians, so waist yaw is not
        mistaken for true elbow clearance.
        """
        torso_pose = robot.data.body_pose_w[:, torso_body]
        elbow_pose = robot.data.body_pose_w[:, elbow_body]
        offset = elbow_pose[:, :3] - torso_pose[:, :3]
        y_axis = torch.zeros((1, 3), device=sim.device, dtype=offset.dtype)
        y_axis[:, 1] = 1.0
        outward = quat_apply(torso_pose[:, 3:7], y_axis)
        clearance = torch.sum(offset * outward, dim=-1)

        all_j = robot.root_physx_view.get_jacobians()
        j_elbow = all_j[:, elbow_jac_id, :, ik_ids]
        j_torso = all_j[:, torso_jac_id, :, ik_ids]
        # d[(elbow - torso) dot outward] / dq, in world coordinates.
        linear_part = torch.einsum(
            "bi,bij->bj", outward, j_elbow[:, :3, :] - j_torso[:, :3, :]
        )
        angular_part = torch.einsum(
            "bi,bij->bj", torch.cross(outward, offset, dim=-1), j_torso[:, 3:, :]
        )
        return clearance, linear_part + angular_part

    def ik_target(
        world_pos: torch.Tensor,
        world_quat: torch.Tensor,
        elbow_out_active: bool = False,
        posture_hold_active: bool = False,
    ) -> torch.Tensor:
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
        hand_j = robot.root_physx_view.get_jacobians()[:, jac_id, :, ik_ids]
        current_joint = robot.data.joint_pos[:, ik_ids]
        desired_joint = ik.compute(current_p, current_q, hand_j, current_joint)

        if elbow_out_active and elbow_goal is not None:
            hand_separation = torch.linalg.vector_norm(current[:, :3] - world_pos, dim=-1)
            if float(hand_separation[0].item()) < ELBOW_HAND_ERROR_CUTOFF_M:
                clearance, clearance_j = elbow_out_distance_and_gradient()
                outward_error = torch.clamp(elbow_goal - clearance, min=0.0, max=0.06)
                if outward_error[0] > 0.001:
                    # Null-space projector: protect the six-dimensional hand
                    # pose while biasing the redundant waist+arm degrees of
                    # freedom toward greater elbow/torso clearance.
                    j_pinv = torch.linalg.pinv(hand_j, rtol=1e-4)
                    identity = torch.eye(len(ik_ids), device=sim.device).unsqueeze(0)
                    nullspace = identity - torch.bmm(j_pinv, hand_j)
                    projected_gradient = torch.bmm(
                        nullspace, clearance_j.unsqueeze(-1)
                    ).squeeze(-1)
                    norm_sq = torch.sum(projected_gradient.square(), dim=-1, keepdim=True)
                    secondary = (
                        ELBOW_NULLSPACE_GAIN
                        * outward_error.unsqueeze(-1)
                        * projected_gradient
                        / (norm_sq + ELBOW_DLS_REGULARIZER)
                    )
                    secondary = torch.clamp(
                        secondary,
                        -ELBOW_NULLSPACE_MAX_STEP_RAD,
                        ELBOW_NULLSPACE_MAX_STEP_RAD,
                    )
                    desired_joint = desired_joint + secondary

        # Ergonomic null-space posture hold for the moving-camera phase.
        # This preserves the already established handle grasp as the primary
        # 6-DoF task and avoids the IK wandering to a different elbow branch.
        if posture_hold_active and posture_reference is not None:
            jac_pinv = torch.linalg.pinv(hand_j, rtol=1e-4)
            identity = torch.eye(len(ik_ids), device=sim.device, dtype=hand_j.dtype).unsqueeze(0)
            nullspace = identity - torch.bmm(jac_pinv, hand_j)
            posture_error = posture_reference - current_joint
            weights = torch.ones_like(posture_error)
            weights[:, 0] = POSTURE_WEIGHT_WAIST
            weights[:, elbow_ik_index] = POSTURE_WEIGHT_ELBOW
            posture_correction = torch.bmm(
                nullspace, (weights * posture_error).unsqueeze(-1)
            ).squeeze(-1)
            posture_correction = torch.clamp(
                POSTURE_HOLD_GAIN * posture_correction,
                -POSTURE_MAX_STEP_RAD,
                POSTURE_MAX_STEP_RAD,
            )
            desired_joint = desired_joint + posture_correction

        max_steps = torch.full_like(current_joint, MAX_ARM_STEP)
        max_steps[:, 0] = MAX_WAIST_STEP
        next_joint = current_joint + torch.clamp(
            desired_joint - current_joint, -max_steps, max_steps
        )
        next_joint = torch.maximum(
            torch.minimum(next_joint, joint_limits[..., 1]), joint_limits[..., 0]
        )
        next_joint[:, 0] = next_joint[:, 0].clamp(
            waist_home - waist_limit_rad, waist_home + waist_limit_rad
        )
        # Prevent the IK from changing to the backwards/folded elbow branch.
        # The 95-degree floor is an additional manipulation workspace limit;
        # the underlying URDF elbow physical limits are still enforced above.
        next_joint[:, elbow_ik_index] = torch.clamp(
            next_joint[:, elbow_ik_index], min=elbow_operating_lower,
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
    print(" RAVEN — HUMAN-LIKE TRIPOD OPERATION / POSTURE-HOLD PAN")
    print("=" * 74)
    print(f"CPU PhysX | {PHYSICS_HZ:.0f} Hz | {TARGETS_DEG} deg | physical bounds +/- {PAN_PHYSICAL_LIMIT_DEG:.1f} deg")
    print(f"Waist workspace +/-{WAIST_YAW_LIMIT_DEG:.1f} deg from default; elbow physical limit remains enforced")
    print(f"Elbow-out preparation request: +{ELBOW_OUT_GOAL_M * 100:.1f} cm")
    print(f"Elbow operational floor: {ELBOW_WORKSPACE_MIN_DEG:.1f} deg (ABORT below {ELBOW_EMERGENCY_MIN_DEG:.1f})")
    print("Elbow-out preparation is limited to the locked phase.")
    print(f"During pan: ergonomic null-space posture hold (preferred elbow {ELBOW_PREFERRED_DEG:.0f} deg).")
    print(f"Camera pan: eased {PAN_RAMP_SEC:.1f}s movement + {TARGET_HOLD_MIN_S:.1f}s hold per setpoint.")
    print("First: elbow-out with tripod pan LOCKED.")
    print("Mode:", "elbow-out followed by sweep" if args_cli.run_sweep else "POSTURE ONLY (no pan sweep)")
    print("The pan motor will NOT be driven during an enabled sweep; the assisted joint will.")
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

            if t < ELBOW_PREP_END_S or not args_cli.run_sweep or sweep_disallowed:
                phase = "ELBOW_OUT" if t < ELBOW_PREP_END_S else "POSTURE_HOLD"
                if elbow_baseline is None:
                    clearance, _ = elbow_out_distance_and_gradient()
                    elbow_baseline = clearance.clone()
                    elbow_goal = clearance + ELBOW_OUT_GOAL_M
                    elbow_prep_pan_ref = tripod.data.joint_pos[:, pan_id].clone()
                    print(
                        f"[RAVEN] ELBOW-OUT START: lateral={clearance[0].item():.3f} m, "
                        f"goal={elbow_goal[0].item():.3f} m "
                        f"(+{ELBOW_OUT_GOAL_M * 100:.1f} cm)"
                    )
                actual_pan_drift = math.degrees(
                    (tripod.data.joint_pos[0, pan_id] - elbow_prep_pan_ref[0]).item()
                )
                if abs(actual_pan_drift) > PAN_LOCK_DRIFT_LIMIT_DEG and not posture_halt:
                    posture_halt = True
                    print(
                        f"[RAVEN] POSTURE SAFETY HOLD: pan drift {actual_pan_drift:+.2f} deg"
                    )
                # The camera stays locked while the arm changes posture.
                zero_lead = torch.zeros((1,), device=sim.device)
                desired_p, desired_q = hand_pose_for_handle_lead(zero_lead)
                target[:, ik_ids] = ik_target(
                    desired_p,
                    desired_q,
                    elbow_out_active=(not posture_halt and t < ELBOW_PREP_END_S),
                )
                if posture_halt:
                    target[:, ik_ids] = robot.data.joint_pos[:, ik_ids].clone()
                current_outward, _ = elbow_out_distance_and_gradient()
                achieved_cm = (current_outward[0] - elbow_baseline[0]).item() * 100.0
                elapsed = t - GRASP_READY_S
                this_sec = int(elapsed)
                if this_sec != posture_test_last_log:
                    hand_error = torch.linalg.vector_norm(
                        robot.data.body_pose_w[:, hand_body, :3] - desired_p, dim=-1
                    )[0].item()
                    print(
                        f"[ELBOW] outwardChange={achieved_cm:+.2f} cm | "
                        f"goal={ELBOW_OUT_GOAL_M * 100:.1f} cm | "
                        f"panDrift={actual_pan_drift:+.3f} deg | "
                        f"handError={hand_error:.4f} m | "
                        f"waist={math.degrees(robot.data.joint_pos[0, waist_id].item() - waist_home):+.1f} deg"
                    )
                    posture_test_last_log = this_sec
                if t >= ELBOW_PREP_END_S and not posture_reported:
                    if args_cli.run_sweep and (achieved_cm < 0.5 or posture_halt):
                        sweep_disallowed = True
                        print("[RAVEN] SWEEP BLOCKED: elbow preparation did not meet safety/minimum movement checks; pan remains locked.")
                    print(
                        f"[RAVEN] ELBOW POSTURE RESULT: +{achieved_cm:.2f} cm lateral change "
                        f"(requested +{ELBOW_OUT_GOAL_M * 100:.1f} cm)."
                    )
                    if not args_cli.run_sweep:
                        print("[RAVEN] POSTURE-ONLY HOLD: tripod PAN remains locked. Close GUI when finished.")
                    else:
                        print("[RAVEN] Elbow-out preparation complete; ergonomic posture-hold will activate on sweep.")
                        print(f"[RAVEN] Elbow angle will be constrained to >= {ELBOW_WORKSPACE_MIN_DEG:.1f} deg.")
                    posture_reported = True

            elif t < PAN_RELEASE_S:
                # The preparation branch uses t < ELBOW_PREP_END_S, so its
                # end-of-prep status must be assessed HERE before we release
                # the pan joint. Otherwise a failed posture adjustment would
                # silently proceed into the full camera sweep.
                if not posture_reported:
                    clearance, _ = elbow_out_distance_and_gradient()
                    achieved_cm = (clearance[0] - elbow_baseline[0]).item() * 100.0
                    drift_deg = math.degrees(
                        (tripod.data.joint_pos[0, pan_id] - elbow_prep_pan_ref[0]).item()
                    )
                    print(
                        f"[RAVEN] ELBOW PREPARATION COMPLETE: "
                        f"outwardChange={achieved_cm:+.2f} cm, "
                        f"panDrift={drift_deg:+.2f} deg."
                    )
                    if achieved_cm < 0.5 or posture_halt or abs(drift_deg) > PAN_LOCK_DRIFT_LIMIT_DEG:
                        sweep_disallowed = True
                        print(
                            "[RAVEN] SWEEP BLOCKED: elbow preparation was "
                            "insufficient or tripod drift exceeded the safety limit."
                        )
                    else:
                        print(
                            "[RAVEN] Active elbow-out gradient DISABLED for the sweep. "
                            "Bounded ergonomic posture-hold enabled."
                        )
                    posture_reported = True
                if not sweep_disallowed and not posture_hold_initialized:
                    # Stable reference is the tested elbow-out grasp posture,
                    # rather than the factory standing pose or a free-running
                    # elbow-out direction. Prefer a slightly relaxed elbow and
                    # modest torso yaw when redundancy permits it.
                    posture_reference = robot.data.joint_pos[:, ik_ids].clone()
                    posture_reference[:, 0] = waist_home
                    posture_reference[:, elbow_ik_index] = math.radians(ELBOW_PREFERRED_DEG)
                    posture_hold_initialized = True
                    print("[RAVEN] Ergonomic posture reference captured: "
                          f"elbow={ELBOW_PREFERRED_DEG:.1f} deg, waist=home")
                if sweep_disallowed:
                    # Keep the pan locked and hold current IK posture. Do NOT
                    # enter release phase after a failed posture preparation.
                    phase = "POSTURE_HOLD"
                    target[:, ik_ids] = robot.data.joint_pos[:, ik_ids].clone()
                else:
                    phase = "SOFT_RELEASE"
                if not sweep_disallowed:
                    a = smoothstep((t - ELBOW_PREP_END_S) / RELEASE_S)
                    stiffness = PAN_LOCK_STIFFNESS * (1.0 - a)
                    damping = PAN_LOCK_DAMPING * (1.0 - a) + PAN_FREE_DAMPING * a
                    tripod.write_joint_stiffness_to_sim(torch_column(1, stiffness, sim.device), joint_ids=[pan_id])
                    tripod.write_joint_damping_to_sim(torch_column(1, damping, sim.device), joint_ids=[pan_id])
            else:
                if sweep_disallowed:
                    # A failed elbow preparation must never release the tripod.
                    phase = "POSTURE_HOLD"
                    target[:, ik_ids] = robot.data.joint_pos[:, ik_ids].clone()
                else:
                    phase = "FREE_SETTLE" if t < PAN_START_S else "PAN_CONTROL"
                    tripod.write_joint_stiffness_to_sim(
                        torch_column(1, 0.0, sim.device), joint_ids=[pan_id]
                    )
                    tripod.write_joint_damping_to_sim(
                        torch_column(1, PAN_FREE_DAMPING, sim.device), joint_ids=[pan_id]
                    )

            lead = torch.zeros((1,), device=sim.device)
            if t >= PAN_START_S and args_cli.run_sweep and not sweep_disallowed:
                if pan_zero is None:
                    pan_zero = tripod.data.joint_pos[:, pan_id].clone()
                    # This is an actual PhysX limit, not merely a controller threshold.
                    limit = math.radians(PAN_PHYSICAL_LIMIT_DEG)
                    bounds = torch.stack((pan_zero - limit, pan_zero + limit), dim=-1).unsqueeze(1)
                    tripod.write_joint_position_limit_to_sim(bounds, joint_ids=[pan_id])
                    segment_start = t
                    segment_start_angle_deg = 0.0
                    previous_pan_angle = tripod.data.joint_pos[:, pan_id].clone()
                    print(f"[RAVEN] PAN ACTIVE | zero={math.degrees(pan_zero[0].item()):+.2f} deg")
                    print(f"[RAVEN] Physics limits set to +/-{PAN_PHYSICAL_LIMIT_DEG:.1f} deg from that zero")

                desired = float(TARGETS_DEG[target_index])
                actual_rad = tripod.data.joint_pos[:, pan_id] - pan_zero
                actual_deg = math.degrees(actual_rad[0].item())
                # Smooth human-style pan trajectory, not a discontinuous jump.
                elapsed = t - segment_start
                alpha = smoothstep(elapsed / PAN_RAMP_SEC)
                if segment_start_angle_deg is None:
                    segment_start_angle_deg = actual_deg
                requested_deg = segment_start_angle_deg + alpha * (desired - segment_start_angle_deg)

                # Filter measured angular rate: instantaneous PhysX joint_vel
                # can oscillate strongly during the fixed-joint test.
                if previous_pan_angle is None:
                    previous_pan_angle = tripod.data.joint_pos[:, pan_id].clone()
                new_angle = tripod.data.joint_pos[:, pan_id]
                raw_vel = (new_angle - previous_pan_angle) / dt
                previous_pan_angle = new_angle.clone()
                velocity = 0.95 * filtered_pan_vel + 0.05 * raw_vel
                filtered_pan_vel = velocity.clone()
                error_deg = requested_deg - actual_deg
                waist_relative_deg = math.degrees(
                    robot.data.joint_pos[0, waist_id].item() - waist_home
                )
                waist_margin_custom_deg = WAIST_YAW_LIMIT_DEG - abs(waist_relative_deg)
                waist_now = float(robot.data.joint_pos[0, waist_id].item())
                waist_margin_physical_deg = math.degrees(
                    min(waist_now - waist_physical_lower, waist_physical_upper - waist_now)
                )
                waist_margin_deg = min(waist_margin_custom_deg, waist_margin_physical_deg)
                elbow_deg = math.degrees(robot.data.joint_pos[0, elbow_id].item())
                elbow_margin_deg = math.degrees(elbow_upper) - elbow_deg
                elbow_lower_room = elbow_deg - ELBOW_WORKSPACE_MIN_DEG

                # A physically welded grasp can generate contact forces even
                # when the joint target is safe. Abort if the actual elbow
                # escapes significantly below our operational workspace.
                if elbow_deg < ELBOW_EMERGENCY_MIN_DEG:
                    print(
                        f"[RAVEN] EMERGENCY ABORT: left elbow {elbow_deg:.1f} deg "
                        f"< {ELBOW_EMERGENCY_MIN_DEG:.1f} deg. "
                        "Closing simulation to avoid folded-arm collision."
                    )
                    break

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
                    # If a target cannot be reached because the waist is at
                    # our workspace boundary, do not continue applying an IK
                    # command there for four seconds. Record it and return
                    # toward centre so the arm can unwind.
                    saturated_waist = (
                        waist_margin_deg < 0.6
                        and abs(error_deg) > 1.5
                        and (t - segment_start) > PAN_RAMP_SEC
                    )
                    saturated_elbow = (
                        elbow_lower_room < ELBOW_NEAR_LIMIT_MARGIN_DEG
                        and abs(desired) > 1.0
                        and abs(error_deg) > 1.5
                        and (t - segment_start) > PAN_RAMP_SEC
                    )
                    if saturated_elbow and not saturation_announced_for_target:
                        print(
                            f"[RAVEN] ELBOW WORKSPACE LIMIT: "
                            f"elbow={elbow_deg:.1f} deg, "
                            f"target={desired:+.1f}, actual={actual_deg:+.2f}. "
                            "Unwinding rather than forcing the IK branch."
                        )
                        saturation_announced_for_target = True
                    if saturated_waist and not saturation_announced_for_target:
                        print(
                            f"[RAVEN] WAIST WORKSPACE LIMIT: "
                            f"waistOffset={waist_relative_deg:+.1f}, "
                            f"target={desired:+.1f}, actual={actual_deg:+.2f}"
                        )
                        saturation_announced_for_target = True

                    lead_rad = PAN_KP * (math.radians(requested_deg) - actual_rad) - PAN_KD * velocity
                    lead = lead_rad.clamp(-math.radians(MAX_LEAD_DEG), +math.radians(MAX_LEAD_DEG))

                    if (elapsed >= PAN_RAMP_SEC + TARGET_HOLD_MIN_S
                            and abs(desired - actual_deg) < AT_TARGET_TOL_DEG
                            and abs(math.degrees(velocity[0].item())) < 3.0):
                        in_tolerance_for += dt
                    else:
                        in_tolerance_for = 0.0

                    if (
                        in_tolerance_for >= AT_TARGET_HOLD_S
                        or t - segment_start >= TARGET_TIMEOUT_S
                        or saturated_waist
                        or saturated_elbow
                    ):
                        why = (
                            "reached" if in_tolerance_for >= AT_TARGET_HOLD_S
                            else "elbow workspace limit" if saturated_elbow
                            else "waist limit" if saturated_waist
                            else "timeout"
                        )
                        if saturated_elbow:
                            # Don't finish the current frame by pushing deeper
                            # into an elbow configuration we have just ruled
                            # out. The next simulation step will run the next
                            # setpoint (normally centre) instead.
                            lead = torch.zeros_like(lead)
                        print(f"[RAVEN] {why.upper():7s}: target={desired:+.1f}, actual={actual_deg:+.2f}")
                        target_index = (target_index + 1) % len(TARGETS_DEG)
                        segment_start_angle_deg = actual_deg
                        segment_start = t
                        in_tolerance_for = 0.0
                        saturation_announced_for_target = False
                        print(f"[RAVEN] NEW TARGET {TARGETS_DEG[target_index]:+.1f} deg")

                second = int(t - PAN_START_S)
                if second != last_log:
                    print(
                        f"[PAN] target={desired:+5.1f} | actual={actual_deg:+6.2f} | "
                        f"request={requested_deg:+5.1f} | "
                        f"velLPF={math.degrees(velocity[0].item()):+6.2f} deg/s | "
                        f"waistOffset={waist_relative_deg:+5.1f} | "
                        f"waistRoom={waist_margin_deg:4.1f} | "
                        f"elbow={elbow_deg:5.1f} | elbowMinRoom={elbow_lower_room:4.1f} | "
                        f"elbowUpperRoom={elbow_margin_deg:4.1f} | "
                        f"assistError={grasp_error:.4f} m | {('STOP' if stopped else 'RUN')}"
                    )
                    last_log = second

            if t >= ELBOW_PREP_END_S and args_cli.run_sweep and not sweep_disallowed:
                if stopped:
                    if last_arm_target is None:
                        last_arm_target = robot.data.joint_pos[:, ik_ids].clone()
                    target[:, ik_ids] = last_arm_target
                else:
                    desired_p, desired_q = hand_pose_for_handle_lead(lead)
                    new_joint_target = ik_target(
                        desired_p, desired_q,
                        elbow_out_active=False,
                        posture_hold_active=posture_hold_initialized,
                    )
                    target[:, ik_ids] = new_joint_target
                    last_arm_target = new_joint_target.clone()

        if phase != phase_old:
            print(f"[RAVEN] Phase -> {phase}")
            phase_old = phase

        robot.set_joint_position_target(target)
        tripod.set_joint_position_target(torch_column(1, 0.0, sim.device), joint_ids=[tilt_id])
        if t < PAN_RELEASE_S or not args_cli.run_sweep or sweep_disallowed:
            tripod.set_joint_position_target(
                torch_column(1, 0.0, sim.device), joint_ids=[pan_id]
            )
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
