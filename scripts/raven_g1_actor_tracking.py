
"""RAVEN - G1 RGB-D Actor Tracking with Physical Tripod Pan.

Isaac Sim 5.1 / Isaac Lab 2.3.2

Sequence:
    1. Initialise G1 in a neutral standing pose.
    2. Approach the existing tripod.
    3. Close the BrainCo Revo2 hand.
    4. Attach temporary physics-assisted grasp.
    5. Establish the stable elbow operating posture.
    6. Release the passive tripod pan joint.
    7. Detect a moving actor using G1 RGB-D images.
    8. Physically pan the tripod through G1 arm and waist movement.

Camera pan range: +/-10 degrees (20 degrees total).

The assisted grasp is a temporary FixedJoint, not proof of
unaided BrainCo finger-contact manipulation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math

from pathlib import Path
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="RAVEN G1 RGB-D physical actor tracking"
)

parser.add_argument("--sweep-deg", type=float, default=10.0)
parser.add_argument("--waist-limit-deg", type=float, default=24.0)
parser.add_argument("--elbow-out-cm", type=float, default=2.0)
parser.add_argument("--tracking-duration-s", type=float, default=35.0)
parser.add_argument("--actor-period-s", type=float, default=11.0)
parser.add_argument("--actor-amplitude-m", type=float, default=0.4)
parser.add_argument(
    "--output-dir",
    default="raven_tracking_results",
)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not 6.0 <= args.sweep_deg <= 10.0:
    parser.error("Sweep amplitude must be between 6 and 10 degrees")

if not 12.0 <= args.waist_limit_deg <= 26.0:
    parser.error("Waist workspace must be between 12 and 26 degrees")

if not 1.0 <= args.elbow_out_cm <= 6.0:
    parser.error("Elbow-out preparation must be between 1 and 6 cm")

if args.tracking_duration_s < 5.0:
    parser.error("Tracking duration must be at least 5 seconds")

if args.actor_period_s < 4.0:
    parser.error("Actor period must be at least 4 seconds")

if not 0.05 <= args.actor_amplitude_m <= 0.5:
    parser.error("Actor amplitude must be between 0.05 and 0.5 metres")

args.enable_cameras = True
simulation_app = AppLauncher(args).app


import torch
import omni.usd

from pxr import Gf, Sdf, UsdPhysics, UsdGeom, UsdShade

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_assets import G1_INSPIRE_FTP_CFG
from raven_g1.assets import G1_BRAINCO_MODE15_CFG, TRIPOD_CAMERA_CFG


try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------

PHYSICS_HZ = 120.0
DT = 1.0 / PHYSICS_HZ

TRACKING_HZ = 10.0
TRACKING_STEPS = round(PHYSICS_HZ / TRACKING_HZ)

TRIPOD_X = 0.61

D435_MOUNT_POS = (
    0.0576235,
    0.01753,
    0.42987,
)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 360
MIN_ACTOR_PIXELS = 80

ACTOR_X = 3.20
ACTOR_PATH = "/World/RAVEN_TrackingActor"

PAN_KP = 1.5
PAN_KD = 0.10
PAN_WORLD_SIGN = 1.0

MAX_LEAD_DEG = 6.0
PAN_SLEW_DEG_S = 8.0

PAN_LOCK_STIFFNESS = 500.0
PAN_LOCK_DAMPING = 30.0
PAN_FREE_DAMPING = 0.45

PAN_PHYSICAL_LIMIT_DEG = args.sweep_deg + 5.0

WAIST_YAW_LIMIT_DEG = args.waist_limit_deg

MAX_WAIST_STEP = 0.007
MAX_ARM_STEP = 0.012

MAX_HAND_POSITION_ERROR_M = 0.045

ELBOW_OUT_GOAL_M = args.elbow_out_cm / 100.0
ELBOW_PREP_S = 6.0

ELBOW_NULLSPACE_GAIN = 0.25
ELBOW_NULLSPACE_MAX_STEP_RAD = 0.0035
ELBOW_DLS_REGULARIZER = 0.001

ELBOW_HAND_ERROR_CUTOFF_M = 0.022
PAN_LOCK_DRIFT_LIMIT_DEG = 0.75

ELBOW_WORKSPACE_MIN_DEG = 95.0
ELBOW_EMERGENCY_MIN_DEG = 89.0

ELBOW_PREFERRED_DEG = 110.0

POSTURE_HOLD_GAIN = 0.075
POSTURE_MAX_STEP_RAD = 0.00075

POSTURE_WEIGHT_WAIST = 1.8
POSTURE_WEIGHT_ELBOW = 1.1

NO_DETECTION_RETURN_S = 1.2


# ---------------------------------------------------------------------
# Phase durations
# ---------------------------------------------------------------------

MOVE_S = 8.0
PRE_CLOSE_S = 2.0
CLOSE_S = 6.0
GRASP_SETTLE_S = 2.0

RELEASE_S = 2.5
FREE_SETTLE_S = 1.0

GRASP_READY_S = (
    MOVE_S
    + PRE_CLOSE_S
    + CLOSE_S
    + GRASP_SETTLE_S
)

ELBOW_PREP_END_S = GRASP_READY_S + ELBOW_PREP_S
PAN_RELEASE_S = ELBOW_PREP_END_S + RELEASE_S
PAN_START_S = PAN_RELEASE_S + FREE_SETTLE_S


# ---------------------------------------------------------------------
# G1 joints and poses
# ---------------------------------------------------------------------

ARM_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

RIGHT_ARM_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

IK_NAMES = ("waist_yaw_joint",) + ARM_NAMES


# Explicit neutral starting configuration.
# Both arms start near the sides rather than reaching forward.

LEFT_ARM_STAND = (
    0.10,
    0.18,
    0.00,
    0.35,
    0.00,
    0.00,
    0.00,
)

RIGHT_ARM_STAND = (
    0.10,
    -0.18,
    0.00,
    0.35,
    0.00,
    0.00,
    0.00,
)


# Original calibrated grasp pose. Do not change these values.

ARM_CALIBRATED = (
    -1.10463631,
    +0.45274511,
    +0.89249939,
    +2.09439969,
    +0.67981434,
    -0.22342572,
    -0.18944184,
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


# ---------------------------------------------------------------------
# Robot configuration
# ---------------------------------------------------------------------

robot_cfg = G1_BRAINCO_MODE15_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"

robot_cfg.actuators = copy.deepcopy(
    G1_INSPIRE_FTP_CFG.actuators
)

robot_cfg.actuators["hands"] = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*_index_.*",
        ".*_middle_.*",
        ".*_thumb_.*",
        ".*_ring_.*",
        ".*_pinky_.*",
    ],
    effort_limit_sim=45.0,
    velocity_limit_sim=10.0,
    stiffness=30.0,
    damping=2.0,
    armature=0.001,
)


tripod_cfg = TRIPOD_CAMERA_CFG.copy()
tripod_cfg.prim_path = "{ENV_REGEX_NS}/Tripod"

original_pos = tripod_cfg.init_state.pos

tripod_cfg.init_state.pos = (
    TRIPOD_X,
    original_pos[1],
    original_pos[2],
)

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

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2500.0,
            color=(0.8, 0.8, 0.8),
        ),
    )

    robot = robot_cfg
    tripod = tripod_cfg

    g1_rgbd = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/raven_d435_rgbd",
        update_period=0.1,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
            focus_distance=3.0,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=D435_MOUNT_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def smoothstep(value):
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def yaw_quat(angle):
    q = torch.zeros(
        (angle.shape[0], 4),
        dtype=angle.dtype,
        device=angle.device,
    )
    q[:, 0] = torch.cos(0.5 * angle)
    q[:, 3] = torch.sin(0.5 * angle)
    return q


def torch_column(value, device):
    return torch.full(
        (1, 1),
        value,
        device=device,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------
# Physics-assisted grasp
# ---------------------------------------------------------------------

def create_assisted_grasp_joint(
    hand_path,
    handle_path,
    relative_p,
    relative_q,
):
    stage = omni.usd.get_context().get_stage()

    for path in (hand_path, handle_path):
        prim = stage.GetPrimAtPath(path)

        if not prim.IsValid():
            raise RuntimeError(
                f"Missing grasp rigid body: {path}"
            )

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                f"Grasp prim is not a rigid body: {path}"
            )

    joint_path = "/World/RAVEN_AssistedGraspJoint"

    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)

    joint = UsdPhysics.FixedJoint.Define(
        stage,
        joint_path,
    )

    joint.CreateBody0Rel().SetTargets([
        Sdf.Path(hand_path)
    ])

    joint.CreateBody1Rel().SetTargets([
        Sdf.Path(handle_path)
    ])

    joint.CreateExcludeFromArticulationAttr(True)
    joint.CreateCollisionEnabledAttr(False)

    joint.CreateLocalPos0Attr(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    joint.CreateLocalRot0Attr(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )

    p = relative_p[0].detach().cpu().tolist()
    q = relative_q[0].detach().cpu().tolist()

    joint.CreateLocalPos1Attr(
        Gf.Vec3f(*map(float, p))
    )

    joint.CreateLocalRot1Attr(
        Gf.Quatf(
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        )
    )

    joint.CreateJointEnabledAttr(True)

    print("[RAVEN] Physics-assisted grasp attached.")
    return joint_path


# ---------------------------------------------------------------------
# Actor generation
# ---------------------------------------------------------------------

def create_material(stage, name, colour):
    path = f"/World/Looks/{name}"

    material = UsdShade.Material.Define(
        stage,
        path,
    )

    shader = UsdShade.Shader.Define(
        stage,
        path + "/Shader",
    )

    shader.CreateIdAttr("UsdPreviewSurface")

    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(Gf.Vec3f(*colour))

    shader.CreateInput(
        "emissiveColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(
        Gf.Vec3f(
            *(component * 0.3 for component in colour)
        )
    )

    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(0.9)

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    return material


def create_actor(stage):
    root = UsdGeom.Xform.Define(
        stage,
        ACTOR_PATH,
    )

    pink = create_material(
        stage,
        "RAVEN_Actor_Pink",
        (0.98, 0.03, 0.75),
    )

    grey = create_material(
        stage,
        "RAVEN_Actor_Grey",
        (0.16, 0.19, 0.25),
    )

    def capsule(name, centre, radius, height, material):
        shape = UsdGeom.Capsule.Define(
            stage,
            ACTOR_PATH + "/" + name,
        )

        shape.CreateRadiusAttr(radius)
        shape.CreateHeightAttr(height)
        shape.CreateAxisAttr("Z")

        UsdGeom.XformCommonAPI(shape).SetTranslate(
            Gf.Vec3d(*centre)
        )

        UsdShade.MaterialBindingAPI.Apply(
            shape.GetPrim()
        ).Bind(material)

    capsule("Torso", (0, 0, 1.10), 0.24, 0.70, pink)
    capsule("LeftLeg", (0, 0.12, 0.43), 0.075, 0.70, grey)
    capsule("RightLeg", (0, -0.12, 0.43), 0.075, 0.70, grey)
    capsule("LeftArm", (0, 0.31, 1.09), 0.065, 0.65, pink)
    capsule("RightArm", (0, -0.31, 1.09), 0.065, 0.65, pink)

    head = UsdGeom.Sphere.Define(
        stage,
        ACTOR_PATH + "/Head",
    )

    head.CreateRadiusAttr(0.14)

    UsdGeom.XformCommonAPI(head).SetTranslate(
        Gf.Vec3d(0, 0, 1.74)
    )

    UsdShade.MaterialBindingAPI.Apply(
        head.GetPrim()
    ).Bind(pink)

    return root


def move_actor(root, lateral):
    UsdGeom.XformCommonAPI(root).SetTranslate(
        Gf.Vec3d(ACTOR_X, lateral, 0.0)
    )


# ---------------------------------------------------------------------
# G1 RGB-D perception
# ---------------------------------------------------------------------

def detect_actor(camera, tripod, robot, torso_body):
    data = camera.data
    rgb = data.output["rgb"][0, ..., :3]

    if rgb.dtype != torch.uint8:
        rgb = (
            (rgb * 255.0).clamp(0, 255).to(torch.uint8)
            if float(torch.nan_to_num(rgb).max()) <= 1.0
            else rgb.clamp(0, 255).to(torch.uint8)
        )

    r = rgb[..., 0].float()
    g = rgb[..., 1].float()
    b = rgb[..., 2].float()

    mask = (
        (r > 90)
        & (b > 65)
        & (r > g * 1.5)
        & (b > g * 1.3)
    )

    depth = data.output["distance_to_image_plane"][0]

    if depth.ndim == 3:
        depth = depth[..., 0]

    valid = (
        mask
        & torch.isfinite(depth)
        & (depth > 0.2)
        & (depth < 15.0)
    )

    pixels = torch.nonzero(valid, as_tuple=False)

    if pixels.shape[0] < MIN_ACTOR_PIXELS:
        return None, rgb, "no valid actor pixels"

    rows = pixels[:, 0]
    cols = pixels[:, 1]

    u = cols.float().mean()
    v = rows.float().mean()
    z = depth[rows, cols].median()

    k = data.intrinsic_matrices[0]

    if not bool(torch.isfinite(k).all()):
        return None, rgb, "invalid camera intrinsics"

    fx = k[0, 0]
    fy = k[1, 1]
    cx = k[0, 2]
    cy = k[1, 2]

    if fx.item() <= 1.0 or fy.item() <= 1.0:
        return None, rgb, "invalid focal length"

    torso = robot.data.body_pose_w[0, torso_body]
    tripod_root = tripod.data.root_pose_w[0]

    if not bool(torch.isfinite(torso).all()):
        return None, rgb, "invalid torso transform"

    if not bool(torch.isfinite(tripod_root).all()):
        return None, rgb, "invalid tripod transform"

    torso_q = torso[3:7].unsqueeze(0)

    mount = torch.tensor(
        D435_MOUNT_POS,
        device=torso.device,
        dtype=torso.dtype,
    ).unsqueeze(0)

    camera_pos = torso[:3] + quat_apply(
        torso_q,
        mount,
    )[0]

    # Camera optical coordinates:
    # +X image-right, +Y image-down, +Z optical-forward.
    # Convert to G1 forward +X, left +Y, up +Z.

    point_local = torch.stack((
        z,
        -(u - cx) * z / fx,
        -(v - cy) * z / fy,
    )).to(
        device=torso.device,
        dtype=torso.dtype,
    )

    point_world = camera_pos + quat_apply(
        torso_q,
        point_local.unsqueeze(0),
    )[0]

    inverse_tripod_q = tripod_root[3:7].clone()
    inverse_tripod_q[1:] *= -1.0

    relative = quat_apply(
        inverse_tripod_q.unsqueeze(0),
        (point_world - tripod_root[:3]).unsqueeze(0),
    )[0]

    if not bool(torch.isfinite(relative).all()):
        return None, rgb, "invalid reconstructed position"

    if float(relative[0]) < 0.3:
        return None, rgb, "target behind tripod"

    bearing = math.degrees(math.atan2(
        float(relative[1]),
        float(relative[0]),
    ))

    if not math.isfinite(bearing):
        return None, rgb, "invalid bearing"

    if abs(bearing) > 60:
        return None, rgb, "bearing outside sector"

    return {
        "bearing_deg": bearing,
        "u": float(u),
        "v": float(v),
        "depth_m": float(z),
        "pixels": int(pixels.shape[0]),
        "bbox": (
            int(cols.min()),
            int(rows.min()),
            int(cols.max()),
            int(rows.max()),
        ),
    }, rgb, "ok"


def save_frame(path, rgb, detection, label):
    if not PIL_AVAILABLE:
        return

    image = Image.fromarray(
        rgb.detach().cpu().numpy()
    )

    draw = ImageDraw.Draw(image)

    if detection is not None:
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

    draw.text(
        (12, 10),
        label,
        fill=(255, 255, 255),
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )

    image.save(path)


# ---------------------------------------------------------------------
# Main robot control
# ---------------------------------------------------------------------

def run(sim, scene, actor_root):
    robot = scene["robot"]
    tripod = scene["tripod"]
    camera = scene["g1_rgbd"]

    dt = sim.get_physics_dt()

    rn = {
        name: i
        for i, name in enumerate(robot.data.joint_names)
    }

    rb = {
        name: i
        for i, name in enumerate(robot.data.body_names)
    }

    tn = {
        name: i
        for i, name in enumerate(tripod.data.joint_names)
    }

    tb = {
        name: i
        for i, name in enumerate(tripod.data.body_names)
    }

    arm_ids = [rn[name] for name in ARM_NAMES]
    right_arm_ids = [rn[name] for name in RIGHT_ARM_NAMES]
    ik_ids = [rn[name] for name in IK_NAMES]
    finger_ids = [rn[name] for name in HAND_CLOSE]

    right_finger_ids = [
        rn[name.replace("left_", "right_", 1)]
        for name in HAND_CLOSE
        if name.replace("left_", "right_", 1) in rn
    ]

    waist_id = rn["waist_yaw_joint"]
    elbow_id = rn["left_elbow_joint"]

    hand_body = rb["left_base_link"]
    elbow_body = rb["left_elbow_link"]
    torso_body = rb["torso_link"]

    hand_jac_id = (
        hand_body - 1 if robot.is_fixed_base else hand_body
    )

    elbow_jac_id = (
        elbow_body - 1 if robot.is_fixed_base else elbow_body
    )

    torso_jac_id = (
        torso_body - 1 if robot.is_fixed_base else torso_body
    )

    handle_body = tb["control_handle_link"]
    pan_body = tb["pan_link"]

    pan_id = tn["pan_joint"]
    tilt_id = tn["tilt_joint"]

    # -------------------------------------------------------------
    # FIXED: explicit neutral startup pose
    # -------------------------------------------------------------

    init = robot.data.default_joint_pos.clone()

    init[:, arm_ids] = torch.tensor(
        LEFT_ARM_STAND,
        device=sim.device,
        dtype=init.dtype,
    ).unsqueeze(0)

    init[:, right_arm_ids] = torch.tensor(
        RIGHT_ARM_STAND,
        device=sim.device,
        dtype=init.dtype,
    ).unsqueeze(0)

    init[:, finger_ids] = 0.0

    if right_finger_ids:
        init[:, right_finger_ids] = 0.0

    arm_init = init[:, arm_ids].clone()
    hand_init = init[:, finger_ids].clone()

    arm_grasp = torch.tensor(
        ARM_CALIBRATED,
        device=sim.device,
        dtype=init.dtype,
    ).unsqueeze(0)

    finger_closed = torch.tensor(
        list(HAND_CLOSE.values()),
        device=sim.device,
        dtype=init.dtype,
    ).unsqueeze(0)

    # Set both actual joint state AND actuator targets before the loop.
    robot.write_joint_state_to_sim(
        init,
        torch.zeros_like(init),
    )

    robot.set_joint_position_target(init)

    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(dt)

    print("[RAVEN] Neutral G1 startup pose applied.")

    # -------------------------------------------------------------
    # IK setup
    # -------------------------------------------------------------

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.04},
        ),
        num_envs=1,
        device=sim.device,
    )

    ik_command = torch.zeros(
        (1, ik.action_dim),
        device=sim.device,
    )

    joint_limits = robot.data.joint_pos_limits[:, ik_ids, :]

    waist_home = float(init[0, waist_id])
    waist_limit = math.radians(WAIST_YAW_LIMIT_DEG)

    elbow_ik_idx = IK_NAMES.index("left_elbow_joint")
    elbow_floor = math.radians(ELBOW_WORKSPACE_MIN_DEG)

    grip_pos = None
    grip_quat = None
    grip_attached = False

    elbow_baseline = None
    elbow_goal = None
    elbow_prep_pan = None

    posture_reference = None
    posture_halt = False
    prep_reported = False

    prep_log_second = -1

    pan_zero = None
    previous_pan = None

    filtered_velocity = torch.zeros(
        (1,),
        device=sim.device,
    )

    reference_bearing = None
    actor_motion_start_time = None

    desired_pan = 0.0
    requested_pan = 0.0

    last_detection_time = -100.0
    latest_detection = None

    detected_frames = 0
    camera_frames = 0

    error_sum = 0.0
    error_count = 0

    pan_min = float("inf")
    pan_max = -float("inf")

    last_phase = None
    last_log_second = -1
    last_image_time = -100.0
    last_warning_second = -1

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "raven_actor_tracking.csv"

    fields = (
        "time_s",
        "detected",
        "pixels",
        "image_bearing_deg",
        "desired_pan_deg",
        "requested_pan_deg",
        "actual_pan_deg",
        "tracking_error_deg",
        "pan_velocity_deg_s",
        "waist_yaw_deg",
        "elbow_deg",
        "grasp_error_m",
        "actor_y_ground_truth_m",
    )

    def elbow_gradient():
        torso = robot.data.body_pose_w[:, torso_body]
        elbow = robot.data.body_pose_w[:, elbow_body]

        offset = elbow[:, :3] - torso[:, :3]

        local_y = torch.tensor(
            [[0.0, 1.0, 0.0]],
            device=sim.device,
        )

        outward = quat_apply(
            torso[:, 3:7],
            local_y,
        )

        clearance = torch.sum(
            offset * outward,
            dim=-1,
        )

        jac = robot.root_physx_view.get_jacobians()

        elbow_j = jac[
            :, elbow_jac_id, :, ik_ids
        ]

        torso_j = jac[
            :, torso_jac_id, :, ik_ids
        ]

        linear = torch.einsum(
            "bi,bij->bj",
            outward,
            elbow_j[:, :3, :] - torso_j[:, :3, :],
        )

        angular = torch.einsum(
            "bi,bij->bj",
            torch.cross(outward, offset, dim=-1),
            torso_j[:, 3:, :],
        )

        return clearance, linear + angular

    def ik_target(
        world_p,
        world_q,
        prepare_elbow=False,
        hold_posture=False,
    ):
        base = robot.data.root_pose_w
        hand = robot.data.body_pose_w[:, hand_body]

        target_p, target_q = subtract_frame_transforms(
            base[:, :3],
            base[:, 3:7],
            world_p,
            world_q,
        )

        current_p, current_q = subtract_frame_transforms(
            base[:, :3],
            base[:, 3:7],
            hand[:, :3],
            hand[:, 3:7],
        )

        ik_command[:, :3] = target_p
        ik_command[:, 3:7] = target_q

        ik.set_command(ik_command)

        hand_j = robot.root_physx_view.get_jacobians()[
            :, hand_jac_id, :, ik_ids
        ]

        q = robot.data.joint_pos[:, ik_ids]

        proposed = ik.compute(
            current_p,
            current_q,
            hand_j,
            q,
        )

        if prepare_elbow and elbow_goal is not None:
            hand_error = torch.linalg.vector_norm(
                hand[:, :3] - world_p,
                dim=-1,
            )

            if float(hand_error[0]) < ELBOW_HAND_ERROR_CUTOFF_M:
                clearance, clearance_j = elbow_gradient()

                error = torch.clamp(
                    elbow_goal - clearance,
                    min=0.0,
                    max=0.06,
                )

                if float(error[0]) > 0.001:
                    inverse = torch.linalg.pinv(
                        hand_j,
                        rtol=1e-4,
                    )

                    identity = torch.eye(
                        len(ik_ids),
                        device=sim.device,
                    ).unsqueeze(0)

                    null = identity - torch.bmm(
                        inverse,
                        hand_j,
                    )

                    gradient = torch.bmm(
                        null,
                        clearance_j.unsqueeze(-1),
                    ).squeeze(-1)

                    magnitude = torch.sum(
                        gradient.square(),
                        dim=-1,
                        keepdim=True,
                    )

                    correction = (
                        ELBOW_NULLSPACE_GAIN
                        * error.unsqueeze(-1)
                        * gradient
                        / (magnitude + ELBOW_DLS_REGULARIZER)
                    )

                    correction = correction.clamp(
                        -ELBOW_NULLSPACE_MAX_STEP_RAD,
                        ELBOW_NULLSPACE_MAX_STEP_RAD,
                    )

                    proposed += correction

        if hold_posture and posture_reference is not None:
            inverse = torch.linalg.pinv(
                hand_j,
                rtol=1e-4,
            )

            identity = torch.eye(
                len(ik_ids),
                device=sim.device,
                dtype=hand_j.dtype,
            ).unsqueeze(0)

            null = identity - torch.bmm(
                inverse,
                hand_j,
            )

            weights = torch.ones_like(q)
            weights[:, 0] = POSTURE_WEIGHT_WAIST
            weights[:, elbow_ik_idx] = POSTURE_WEIGHT_ELBOW

            correction = torch.bmm(
                null,
                (
                    weights * (posture_reference - q)
                ).unsqueeze(-1),
            ).squeeze(-1)

            correction = (
                POSTURE_HOLD_GAIN * correction
            ).clamp(
                -POSTURE_MAX_STEP_RAD,
                POSTURE_MAX_STEP_RAD,
            )

            proposed += correction

        max_step = torch.full_like(
            q,
            MAX_ARM_STEP,
        )

        max_step[:, 0] = MAX_WAIST_STEP

        next_q = q + (
            proposed - q
        ).clamp(
            -max_step,
            max_step,
        )

        next_q = torch.maximum(
            torch.minimum(
                next_q,
                joint_limits[..., 1],
            ),
            joint_limits[..., 0],
        )

        next_q[:, 0] = next_q[:, 0].clamp(
            waist_home - waist_limit,
            waist_home + waist_limit,
        )

        next_q[:, elbow_ik_idx] = (
            next_q[:, elbow_ik_idx].clamp(
                min=elbow_floor
            )
        )

        return next_q

    def hand_target(lead):
        handle = tripod.data.body_pose_w[:, handle_body]
        pivot = tripod.data.body_pose_w[:, pan_body, :3]

        rotation = yaw_quat(
            PAN_WORLD_SIGN * lead
        )

        p = pivot + quat_apply(
            rotation,
            handle[:, :3] - pivot,
        )

        q = quat_mul(
            rotation,
            handle[:, 3:7],
        )

        return combine_frame_transforms(
            p,
            q,
            grip_pos,
            grip_quat,
        )

    print("\n" + "=" * 72)
    print(" RAVEN - G1 RGB-D ACTOR TRACKING")
    print("=" * 72)
    print("Neutral pose -> Approach -> Grasp -> Tracking")
    print("20-degree total physical camera pan range")
    print("G1 actuator-driven, passive tripod pan")
    print("Temporary physics-assisted grasp")
    print(f"Output: {output}\n")

    step = 0
    status = "INCOMPLETE"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
        )
        writer.writeheader()

        while simulation_app.is_running():
            t = step * dt
            relative_time = t - PAN_START_S

            if (
                actor_motion_start_time is None
                or t < actor_motion_start_time
            ):
                actor_y = 0.0
            else:
                actor_y = (
                    args.actor_amplitude_m
                    * math.sin(
                        2.0 * math.pi
                        * (t - actor_motion_start_time)
                        / args.actor_period_s
                    )
                )

            move_actor(actor_root, actor_y)

            command = init.clone()
            phase = "APPROACH"

            # -----------------------------------------------------
            # Calibrated approach
            # -----------------------------------------------------

            if t < MOVE_S:
                a = smoothstep(t / MOVE_S)

                command[:, arm_ids] = (
                    arm_init
                    + a * (arm_grasp - arm_init)
                )

                command[:, finger_ids] = hand_init

            elif t < MOVE_S + PRE_CLOSE_S:
                phase = "PRE_CLOSE"
                command[:, arm_ids] = arm_grasp
                command[:, finger_ids] = hand_init

            elif t < MOVE_S + PRE_CLOSE_S + CLOSE_S:
                phase = "CLOSING"

                a = smoothstep(
                    (t - MOVE_S - PRE_CLOSE_S)
                    / CLOSE_S
                )

                command[:, arm_ids] = arm_grasp

                command[:, finger_ids] = (
                    hand_init
                    + a * (finger_closed - hand_init)
                )

            elif t < GRASP_READY_S:
                phase = "LOCKED_SETTLE"
                command[:, arm_ids] = arm_grasp
                command[:, finger_ids] = finger_closed

            else:
                command[:, finger_ids] = finger_closed

                if not grip_attached:
                    handle = tripod.data.body_pose_w[
                        :, handle_body
                    ]

                    hand = robot.data.body_pose_w[
                        :, hand_body
                    ]

                    grip_pos, grip_quat = subtract_frame_transforms(
                        handle[:, :3],
                        handle[:, 3:7],
                        hand[:, :3],
                        hand[:, 3:7],
                    )

                    create_assisted_grasp_joint(
                        "/World/envs/env_0/Robot/left_base_link",
                        "/World/envs/env_0/Tripod/control_handle_link",
                        grip_pos,
                        grip_quat,
                    )

                    grip_attached = True
                    ik.reset()

                    print("[RAVEN] Calibrated grasp established.")

                # -------------------------------------------------
                # Elbow-out preparation
                # -------------------------------------------------

                if t < ELBOW_PREP_END_S:
                    phase = "ELBOW_OUT"

                    if elbow_baseline is None:
                        elbow_baseline, _ = elbow_gradient()
                        elbow_goal = (
                            elbow_baseline + ELBOW_OUT_GOAL_M
                        )

                        elbow_prep_pan = (
                            tripod.data.joint_pos[:, pan_id].clone()
                        )

                        print("[RAVEN] Elbow preparation started.")

                    drift = math.degrees(
                        float(
                            tripod.data.joint_pos[0, pan_id]
                            - elbow_prep_pan[0]
                        )
                    )

                    if abs(drift) > PAN_LOCK_DRIFT_LIMIT_DEG:
                        posture_halt = True

                    p, q = hand_target(
                        torch.zeros(
                            (1,),
                            device=sim.device,
                        )
                    )

                    command[:, ik_ids] = ik_target(
                        p,
                        q,
                        prepare_elbow=not posture_halt,
                    )

                    if posture_halt:
                        command[:, ik_ids] = (
                            robot.data.joint_pos[:, ik_ids].clone()
                        )

                    second = int(t - GRASP_READY_S)

                    if second != prep_log_second:
                        clearance, _ = elbow_gradient()

                        achieved = float(
                            (clearance - elbow_baseline)[0]
                        ) * 100.0

                        print(
                            f"[ELBOW] outward={achieved:+.2f}cm "
                            f"| panDrift={drift:+.2f}deg"
                        )

                        prep_log_second = second

                # -------------------------------------------------
                # Release passive tripod pan
                # -------------------------------------------------

                elif t < PAN_RELEASE_S:
                    phase = "SOFT_RELEASE"

                    if not prep_reported:
                        clearance, _ = elbow_gradient()

                        achieved = float(
                            (clearance - elbow_baseline)[0]
                        ) * 100.0

                        if achieved < 0.5 or posture_halt:
                            status = "POSTURE_PREPARATION_FAILED"
                            print(
                                "[RAVEN] Elbow preparation failed."
                            )
                            break

                        posture_reference = (
                            robot.data.joint_pos[:, ik_ids].clone()
                        )

                        posture_reference[:, 0] = waist_home

                        posture_reference[:, elbow_ik_idx] = (
                            math.radians(ELBOW_PREFERRED_DEG)
                        )

                        prep_reported = True

                        print(
                            f"[RAVEN] Elbow preparation complete: "
                            f"{achieved:+.2f}cm"
                        )

                    a = smoothstep(
                        (t - ELBOW_PREP_END_S)
                        / RELEASE_S
                    )

                    stiffness = (
                        PAN_LOCK_STIFFNESS * (1.0 - a)
                    )

                    damping = (
                        PAN_LOCK_DAMPING * (1.0 - a)
                        + PAN_FREE_DAMPING * a
                    )

                    tripod.write_joint_stiffness_to_sim(
                        torch_column(stiffness, sim.device),
                        joint_ids=[pan_id],
                    )

                    tripod.write_joint_damping_to_sim(
                        torch_column(damping, sim.device),
                        joint_ids=[pan_id],
                    )

                    p, q = hand_target(
                        torch.zeros(
                            (1,),
                            device=sim.device,
                        )
                    )

                    command[:, ik_ids] = ik_target(
                        p,
                        q,
                        hold_posture=True,
                    )

                else:
                    phase = (
                        "FREE_SETTLE"
                        if t < PAN_START_S
                        else "ACTOR_TRACKING"
                    )

                    tripod.write_joint_stiffness_to_sim(
                        torch_column(0.0, sim.device),
                        joint_ids=[pan_id],
                    )

                    tripod.write_joint_damping_to_sim(
                        torch_column(
                            PAN_FREE_DAMPING,
                            sim.device,
                        ),
                        joint_ids=[pan_id],
                    )

                    lead = torch.zeros(
                        (1,),
                        device=sim.device,
                    )

                    # ---------------------------------------------
                    # RGB-D physical actor tracking
                    # ---------------------------------------------

                    if t >= PAN_START_S:
                        if pan_zero is None:
                            pan_zero = (
                                tripod.data.joint_pos[:, pan_id].clone()
                            )

                            bounds_rad = math.radians(
                                PAN_PHYSICAL_LIMIT_DEG
                            )

                            bounds = torch.stack(
                                (
                                    pan_zero - bounds_rad,
                                    pan_zero + bounds_rad,
                                ),
                                dim=-1,
                            ).unsqueeze(1)

                            tripod.write_joint_position_limit_to_sim(
                                bounds,
                                joint_ids=[pan_id],
                            )

                            previous_pan = pan_zero.clone()

                            print(
                                "[RAVEN] Physical pan active. "
                                "Waiting for RGB-D actor detection."
                            )

                        actual_rad = (
                            tripod.data.joint_pos[:, pan_id]
                            - pan_zero
                        )

                        actual_deg = math.degrees(
                            float(actual_rad[0])
                        )

                        current_pan = (
                            tripod.data.joint_pos[:, pan_id].clone()
                        )

                        raw_velocity = (
                            current_pan - previous_pan
                        ) / dt

                        previous_pan = current_pan

                        filtered_velocity = (
                            0.95 * filtered_velocity
                            + 0.05 * raw_velocity
                        )

                        velocity_deg = math.degrees(
                            float(filtered_velocity[0])
                        )

                        if step % TRACKING_STEPS == 0:
                            camera_frames += 1

                            (
                                latest_detection,
                                rgb,
                                vision_status,
                            ) = detect_actor(
                                camera,
                                tripod,
                                robot,
                                torso_body,
                            )

                            if latest_detection is not None:
                                detected_frames += 1
                                last_detection_time = t

                                bearing = latest_detection[
                                    "bearing_deg"
                                ]

                                if reference_bearing is None:
                                    reference_bearing = bearing
                                    actor_motion_start_time = t + 1.0

                                    print(
                                        "[VISION] Valid reference bearing: "
                                        f"{bearing:+.2f}deg"
                                    )

                                    print(
                                        "[VISION] Actor begins moving in 1s."
                                    )

                                desired_pan = max(
                                    -args.sweep_deg,
                                    min(
                                        args.sweep_deg,
                                        bearing - reference_bearing,
                                    ),
                                )

                            else:
                                if (
                                    t - last_detection_time
                                    > NO_DETECTION_RETURN_S
                                ):
                                    desired_pan = 0.0

                                second = int(relative_time)

                                if second != last_warning_second:
                                    print(
                                        "[VISION] Rejected detection: "
                                        f"{vision_status}"
                                    )
                                    last_warning_second = second

                            if t - last_image_time >= 5.0:
                                label = (
                                    f"G1 RGB-D {relative_time:.1f}s "
                                    f"pan={desired_pan:+.1f} "
                                    f"{'DETECTED' if latest_detection else 'NOT SEEN'}"
                                )

                                save_frame(
                                    output
                                    / f"g1_rgbd_{int(relative_time):03d}.png",
                                    rgb,
                                    latest_detection,
                                    label,
                                )

                                last_image_time = t

                        # Never send NaN or Inf to the G1 controller.
                        if (
                            not math.isfinite(desired_pan)
                            or not math.isfinite(requested_pan)
                        ):
                            desired_pan = 0.0
                            requested_pan = 0.0

                            print(
                                "[VISION] Invalid pan command blocked."
                            )

                        max_delta = PAN_SLEW_DEG_S * dt

                        delta = (
                            desired_pan - requested_pan
                        )

                        requested_pan += max(
                            -max_delta,
                            min(max_delta, delta),
                        )

                        lead = (
                            PAN_KP
                            * (
                                math.radians(requested_pan)
                                - actual_rad
                            )
                            - PAN_KD * filtered_velocity
                        )

                        lead = lead.clamp(
                            -math.radians(MAX_LEAD_DEG),
                            math.radians(MAX_LEAD_DEG),
                        )

                        waist_deg = math.degrees(
                            float(robot.data.joint_pos[0, waist_id])
                            - waist_home
                        )

                        elbow_deg = math.degrees(
                            float(robot.data.joint_pos[0, elbow_id])
                        )

                        handle = tripod.data.body_pose_w[
                            :, handle_body
                        ]

                        hand = robot.data.body_pose_w[
                            :, hand_body
                        ]

                        expected_p, _ = combine_frame_transforms(
                            handle[:, :3],
                            handle[:, 3:7],
                            grip_pos,
                            grip_quat,
                        )

                        grasp_error = float(
                            torch.linalg.vector_norm(
                                hand[:, :3] - expected_p,
                                dim=-1,
                            )[0]
                        )

                        if (
                            elbow_deg < ELBOW_EMERGENCY_MIN_DEG
                            or abs(waist_deg) > WAIST_YAW_LIMIT_DEG + 2.0
                            or abs(actual_deg) > PAN_PHYSICAL_LIMIT_DEG + 1.0
                            or grasp_error > MAX_HAND_POSITION_ERROR_M
                        ):
                            status = "SAFETY_ABORT"

                            print(
                                "[RAVEN] Safety abort: "
                                f"pan={actual_deg:+.2f} "
                                f"elbow={elbow_deg:.1f} "
                                f"waist={waist_deg:+.1f}"
                            )
                            break

                        error = (
                            desired_pan - actual_deg
                        )

                        if latest_detection is not None:
                            error_sum += abs(error)
                            error_count += 1

                            pan_min = min(
                                pan_min,
                                actual_deg,
                            )

                            pan_max = max(
                                pan_max,
                                actual_deg,
                            )

                        if step % TRACKING_STEPS == 0:
                            writer.writerow({
                                "time_s": round(relative_time, 3),
                                "detected": latest_detection is not None,
                                "pixels": (
                                    latest_detection["pixels"]
                                    if latest_detection else ""
                                ),
                                "image_bearing_deg": (
                                    round(
                                        latest_detection["bearing_deg"],
                                        3,
                                    )
                                    if latest_detection else ""
                                ),
                                "desired_pan_deg": round(desired_pan, 3),
                                "requested_pan_deg": round(requested_pan, 3),
                                "actual_pan_deg": round(actual_deg, 3),
                                "tracking_error_deg": round(error, 3),
                                "pan_velocity_deg_s": round(velocity_deg, 3),
                                "waist_yaw_deg": round(waist_deg, 3),
                                "elbow_deg": round(elbow_deg, 3),
                                "grasp_error_m": round(grasp_error, 6),
                                "actor_y_ground_truth_m": round(actor_y, 3),
                            })

                        second = int(relative_time)

                        if second != last_log_second:
                            print(
                                f"[TRACK] t={relative_time:5.1f}s "
                                f"| target={desired_pan:+5.2f}deg "
                                f"| pan={actual_deg:+5.2f}deg "
                                f"| elbow={elbow_deg:5.1f}deg "
                                f"| detected={latest_detection is not None}"
                            )

                            last_log_second = second

                        if (
                            relative_time
                            >= args.tracking_duration_s
                        ):
                            status = "COMPLETED"
                            break

                    p, q = hand_target(lead)

                    command[:, ik_ids] = ik_target(
                        p,
                        q,
                        hold_posture=True,
                    )

            if phase != last_phase:
                print(
                    f"[RAVEN] Phase -> {phase}"
                )
                last_phase = phase

            robot.set_joint_position_target(command)

            tripod.set_joint_position_target(
                torch_column(0.0, sim.device),
                joint_ids=[tilt_id],
            )

            if t < PAN_RELEASE_S:
                tripod.set_joint_position_target(
                    torch_column(0.0, sim.device),
                    joint_ids=[pan_id],
                )

            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt)

            step += 1

        stream.flush()

    if reference_bearing is None:
        status = "NO_VALID_BEARING"

    summary = {
        "experiment": "RAVEN G1 RGB-D physical actor tracking",
        "status": status,
        "tracking_duration_s": args.tracking_duration_s,
        "camera_frames": camera_frames,
        "detected_frames": detected_frames,
        "reference_bearing_deg": reference_bearing,
        "actor_motion_started": actor_motion_start_time is not None,
        "detection_rate": (
            detected_frames / camera_frames
            if camera_frames else 0.0
        ),
        "mean_absolute_tracking_error_deg": (
            error_sum / error_count
            if error_count else None
        ),
        "observed_pan_min_deg": (
            pan_min if error_count else None
        ),
        "observed_pan_max_deg": (
            pan_max if error_count else None
        ),
        "assisted_grasp": True,
        "csv": str(csv_path),
    }

    summary_path = (
        output / "raven_actor_tracking.summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"\n[RAVEN] Status: {status}")
    print(
        f"[RAVEN] Valid detections: "
        f"{detected_frames}/{camera_frames}"
    )
    print(f"[RAVEN] CSV: {csv_path}")
    print(f"[RAVEN] Summary: {summary_path}")


# ---------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------

def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=DT,
            device="cpu",
            use_fabric=False,
        )
    )

    sim.set_camera_view(
        eye=(2.1, 1.8, 1.55),
        target=(0.35, 0.10, 0.85),
    )

    scene = InteractiveScene(
        SceneCfg(
            num_envs=1,
            env_spacing=2.0,
        )
    )

    actor = create_actor(
        omni.usd.get_context().get_stage()
    )

    move_actor(actor, 0.0)

    sim.reset()

    try:
        run(sim, scene, actor)
    finally:
        stage = omni.usd.get_context().get_stage()
        joint = "/World/RAVEN_AssistedGraspJoint"

        if stage and stage.GetPrimAtPath(joint).IsValid():
            stage.RemovePrim(joint)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
