"""RAVEN interactive G1/BrainCo hand-to-tripod calibration (Windows, Isaac Lab).

RUN in the same Python environment as the validated G1 tracker; no baseline
files are changed. Keys are read from the POWERSHELL CONSOLE (not Isaac window).

Hand is positioned by 7-DOF ARM-ONLY damped least squares IK; waist yaw is locked.
finger actuators can close/open. No physics-assisted FixedJoint is created.
Press P to save the *measured* hand-to-handle pose and joint configurations.

IMPORTANT: This calibrates a visual/kinematic grip, not load-bearing contact.
A subsequent separate experiment must validate collision/grasp and tracking.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import msvcrt  # Windows console, available on user's laptop.
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRACKER = SCRIPT_DIR / "raven_g1_actor_tracking.py"
if not TRACKER.is_file():
    raise SystemExit(f"Cannot locate original controller: {TRACKER}")

# Import the original's scene, assets and IK implementation; importing does NOT
# execute its tracking loop or create its assisted grasp joint.
spec = importlib.util.spec_from_file_location("raven_manual_grasp_base", TRACKER)
if spec is None or spec.loader is None:
    raise SystemExit("Unable to import validated tracker")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

import torch

# Change only these initial values if desired. Root X moves toward the handle.
INITIAL_ROOT_FORWARD_M = 0.10
APPROACH_FRACTION = 0.80  # 80% of the previously calibrated arm approach.
APPROACH_DURATION_S = 3.5
POST_APPROACH_SETTLE_S = 0.5
FINGER_RAMP_S = 0.8
DEFAULT_FINE_TRANSLATION_M = 0.005
DEFAULT_COARSE_TRANSLATION_M = 0.020
DEFAULT_FINE_ROTATION_DEG = 1.0
DEFAULT_COARSE_ROTATION_DEG = 4.0
BASE_JOG_M = 0.005
IK_MAX_ARM_STEP_RAD = 0.012
IK_MAX_WAIST_STEP_RAD = 0.007

OUTPUT_DIR = Path.cwd() / "raven_training_results" / "manual_grasp_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")


def as_list(t):
    return [float(v) for v in t.detach().cpu().reshape(-1).tolist()]


def safe_json_values(t):
    vals = as_list(t)
    return [v if math.isfinite(v) else None for v in vals]


def axis_angle(axis, radians, device):
    h = radians * 0.5
    c, s = math.cos(h), math.sin(h)
    return torch.tensor([[c, axis[0]*s, axis[1]*s, axis[2]*s]],
                        device=device, dtype=torch.float32)


def operate(sim, scene):
    robot, tripod = scene["robot"], scene["tripod"]
    dt = float(sim.get_physics_dt())
    rn = {n: i for i, n in enumerate(robot.data.joint_names)}
    rb = {n: i for i, n in enumerate(robot.data.body_names)}
    tb = {n: i for i, n in enumerate(tripod.data.body_names)}
    tn = {n: i for i, n in enumerate(tripod.data.joint_names)}

    arm_ids = [rn[n] for n in base.ARM_NAMES]
    right_arm_ids = [rn[n] for n in base.RIGHT_ARM_NAMES]
    # Exclude waist_yaw_joint so W/S cannot rotate the torso.
    ik_ids = list(arm_ids)
    finger_ids = [rn[n] for n in base.HAND_CLOSE]
    elbow_id = rn["left_elbow_joint"]
    hand_body_id = rb["left_base_link"]
    handle_body_id = tb["control_handle_link"]
    hand_jac_id = hand_body_id - 1 if robot.is_fixed_base else hand_body_id
    elbow_ik_index = base.ARM_NAMES.index("left_elbow_joint")
    pan_id, tilt_id = tn["pan_joint"], tn["tilt_joint"]

    current = robot.data.default_joint_pos.clone()
    current[:, arm_ids] = torch.tensor([base.LEFT_ARM_STAND],