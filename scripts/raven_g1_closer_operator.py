
"""RAVEN experimental operator stance: G1 10 cm nearer the tripod.

Runs a copy of the validated G1 tracking function in memory, with:
  - robot spawn moved +0.10 m on world X; tripod/actor remain unchanged
  - approach retargeted by IK to the VERIFIED handle-relative wrist pose
  - neutral-to-grasp approach and finger closure shortened
  - elbow-out preparation changed to a downward nullspace objective
  - safety abort if wrist misses the calibrated pre-grasp pose
  - unchanged baseline controller, assisted grasp and RL recorder on disk

The pose is for the exact validated G1 assets and tripod configuration;
never apply this script to a different robot or grasp without recalibration.
This is a simulation experiment, not physical-robot control.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACKER = HERE / "raven_g1_actor_tracking.py"
RECORDER = HERE / "raven_g1_rl_data_capture.py"
EXPECTED_SHA256 = "2b7da3c47a8614b690bb20f2f461d0bb93a90a90b9f0c95a3d6c6b4648ecd038"

if not TRACKER.is_file() or not RECORDER.is_file():
    raise SystemExit("Missing original actor tracker or 120 Hz recorder in scripts/")

sha = hashlib.sha256(TRACKER.read_bytes()).hexdigest()
if sha != EXPECTED_SHA256:
    raise SystemExit(
        "Validated baseline code differs from the tested version. "
        f"Expected {EXPECTED_SHA256}, found {sha}. No changes applied."
    )

spec = importlib.util.spec_from_file_location("raven_closer_recorder", RECORDER)
if spec is None or spec.loader is None:
    raise SystemExit(f"Unable to import {RECORDER}")

rec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rec
spec.loader.exec_module(rec)

base = rec.base

import torch

CLOSER_BY_M = 0.10

# Previously validated pose of the wrist relative to the tripod handle.
# Preserve this offset to prevent forcing the wrist inside the handle.
GRIP_LOCAL_POS = (-0.09520331, +0.04216317, +0.04810547)
GRIP_LOCAL_QUAT_WXYZ = (
    -0.01174326, -0.69758332, +0.70176734, -0.14409096
)

MAX_PREGRASP_POSITION_ERROR_M = 0.035
MAX_PREGRASP_ORIENTATION_ERROR_DEG = 18.0

OUT = (
    Path.cwd()
    / "raven_training_results"
    / "closer_operator"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)

rec.OUT = OUT
base.args.output_dir = str(OUT)

# Faster grasp preparation.
base.MOVE_S = 4.5
base.PRE_CLOSE_S = 0.5
base.CLOSE_S = 2.5
base.GRASP_SETTLE_S = 0.75
base.ELBOW_PREP_S = 4.0
base.RELEASE_S = 1.5
base.FREE_SETTLE_S = 0.75

base.GRASP_READY_S = (
    base.MOVE_S + base.PRE_CLOSE_S
    + base.CLOSE_S + base.GRASP_SETTLE_S
)
base.ELBOW_PREP_END_S = base.GRASP_READY_S + base.ELBOW_PREP_S
base.PAN_RELEASE_S = base.ELBOW_PREP_END_S + base.RELEASE_S
base.PAN_START_S = base.PAN_RELEASE_S + base.FREE_SETTLE_S

# Interpreted by the experimental helper as an elbow-DOWN goal.
base.ELBOW_OUT_GOAL_M = 0.025
base.ELBOW_PREFERRED_DEG = 105.0

# Move the robot spawn without changing the tripod.
original_scene_cfg = base.SceneCfg

def closer_scene_cfg(*args, **kwargs):
    cfg = original_scene_cfg(*args, **kwargs)
    xyz = cfg.robot.init_state.pos
    cfg.robot.init_state.pos = (
        xyz[0] + CLOSER_BY_M, xyz[1], xyz[2]
    )
    print(f"[CLOSER] G1 spawn {xyz} -> {cfg.robot.init_state.pos}")
    return cfg

base.SceneCfg = closer_scene_cfg

# Modify a copy of the validated run function in memory.
source_tree = ast.parse(
    TRACKER.read_text(encoding="utf-8"), filename=str(TRACKER)
)

runs = [
    n for n in source_tree.body
    if isinstance(n, ast.FunctionDef) and n.name == "run"
]
if len(runs) != 1:
    raise SystemExit("Expected exactly one run() in validated tracker")

run_fn = runs[0]

# Change the elbow nullspace preference from outward to downward.
helpers = [
    (i, n) for i, n in enumerate(run_fn.body)
    if isinstance(n, ast.FunctionDef)
    and n.name == "elbow_gradient"
]
if len(helpers) != 1:
    raise SystemExit("Expected elbow_gradient() helper")

run_fn.body[helpers[0][0]] = ast.parse('''
def elbow_gradient():
    torso = robot.data.body_pose_w[:, torso_body]
    elbow = robot.data.body_pose_w[:, elbow_body]
    downward_clearance = torso[:, 2] - elbow[:, 2]
    jac = robot.root_physx_view.get_jacobians()
    elbow_j = jac[:, elbow_jac_id, :, ik_ids]
    torso_j = jac[:, torso_jac_id, :, ik_ids]
    return downward_clearance, torso_j[:, 2, :] - elbow_j[:, 2, :]
''').body[0]

# Determine where the hand should go relative to the stationary handle.
start_block = ast.parse('''
initial_hand = robot.data.body_pose_w[:, hand_body].clone()
initial_hand_p = initial_hand[:, :3].clone()
initial_hand_q = initial_hand[:, 3:7].clone()

initial_handle = tripod.data.body_pose_w[:, handle_body]
local_grip_p = torch.tensor(
    [GRIP_LOCAL_POS],
    dtype=initial_hand_p.dtype,
    device=sim.device
)
local_grip_q = torch.tensor(
    [GRIP_LOCAL_QUAT_WXYZ],
    dtype=initial_hand_q.dtype,
    device=sim.device
)

pregrasp_world_p, pregrasp_world_q = combine_frame_transforms(
    initial_handle[:, :3],
    initial_handle[:, 3:7],
    local_grip_p,
    local_grip_q
)

if torch.linalg.vector_norm(
    initial_handle[0, :3] - torch.tensor(
        [0.450, 0.140, 0.910], device=sim.device
    )
) > 0.025:
    raise RuntimeError(
        "Tripod handle has moved from validated calibration"
    )

print(
    "[CLOSER] World wrist pre-grasp goal:",
    pregrasp_world_p[0].tolist()
)
''').body

neutral_points = [
    i for i, n in enumerate(run_fn.body)
    if isinstance(n, ast.Expr)
    and "Neutral G1 startup pose applied." in ast.unparse(n)
]
if len(neutral_points) != 1:
    raise SystemExit("Couldn't locate neutral-pose calibration point")

run_fn.body[
    neutral_points[0]+1:neutral_points[0]+1
] = start_block

# Locate the physics control loop.
loops = [
    n for n in ast.walk(run_fn)
    if isinstance(n, ast.While)
    and "simulation_app.is_running()" in ast.unparse(n.test)
]
if len(loops) != 1:
    raise SystemExit("Expected one simulation control loop")

loop = loops[0]

step_positions = [
    i for i, n in enumerate(loop.body)
    if isinstance(n, ast.Assign)
    and any(
        isinstance(x, ast.Name) and x.id == "t"
        for x in n.targets
    )
]
if len(step_positions) != 1:
    raise SystemExit("Cannot locate simulation timestamp")

loop.body[
    step_positions[0]+1:step_positions[0]+1
] = ast.parse('''
elbow_floor = math.radians(
    15.0 if t < GRASP_READY_S
    else ELBOW_WORKSPACE_MIN_DEG
)
''').body

# Re-solve the approach using feedback IK rather than moving fixed joint
# angles, which would put the hand too close to the handle after relocation.
pregrasp_override = ast.parse('''
if t < GRASP_READY_S:
    alpha = smoothstep(t / MOVE_S)
    target_p = initial_hand_p + alpha * (
        pregrasp_world_p - initial_hand_p
    )

    hemisphere = torch.sign(
        (initial_hand_q * pregrasp_world_q).sum(
            -1, keepdim=True
        )
    )
    hemisphere = torch.where(
        hemisphere == 0,
        torch.ones_like(hemisphere),
        hemisphere
    )
    target_q = (
        (1.0 - alpha) * initial_hand_q
        + alpha * pregrasp_world_q * hemisphere
    )
    target_q = torch.nn.functional.normalize(
        target_q, dim=-1
    )

    command[:, ik_ids] = ik_target(target_p, target_q)
''').body

phase_points = [
    i for i, n in enumerate(loop.body)
    if isinstance(n, ast.If)
    and "phase != last_phase" in ast.unparse(n.test)
]
if len(phase_points) != 1:
    raise SystemExit("Cannot locate pre-actuation target point")

loop.body[
    phase_points[0]:phase_points[0]
] = pregrasp_override

# Abort if the wrist has not reached its expected grasp pose.
grips = [
    n for n in ast.walk(run_fn)
    if isinstance(n, ast.If)
    and ast.unparse(n.test) == "not grip_attached"
]
if len(grips) != 1:
    raise SystemExit("Cannot locate assisted grasp attachment")

grip = grips[0]

hand_points = [
    i for i, n in enumerate(grip.body)
    if isinstance(n, ast.Assign)
    and any(
        isinstance(x, ast.Name) and x.id == "hand"
        for x in n.targets
    )
]
if len(hand_points) != 1:
    raise SystemExit("Cannot locate live hand pose before grasp")

grip.body[
    hand_points[0]+1:hand_points[0]+1
] = ast.parse('''
position_error_m = float(torch.linalg.vector_norm(
    hand[:, :3] - pregrasp_world_p, dim=-1
)[0])

orientation_dot = abs(float(
    (hand[:, 3:7] * pregrasp_world_q).sum()
))
orientation_error_deg = math.degrees(
    2.0 * math.acos(
        min(1.0, max(0.0, orientation_dot))
    )
)

print(
    f"[CLOSER] Pre-grasp wrist error: "
    f"{position_error_m*1000:.1f} mm, "
    f"{orientation_error_deg:.1f} deg"
)

if (
    position_error_m > MAX_PREGRASP_POSITION_ERROR_M
    or orientation_error_deg > MAX_PREGRASP_ORIENTATION_ERROR_DEG
):
    status = "PREGRASP_ALIGNMENT_FAILED"
    print(
        "[CLOSER] Refusing assisted grasp: "
        "incorrect wrist alignment"
    )
    break
''').body

for n in ast.walk(run_fn):
    if isinstance(n, ast.Constant) and isinstance(n.value, str):
        n.value = n.value.replace(
            "[ELBOW] outward=", "[ELBOW] downward="
        )

program = ast.fix_missing_locations(
    ast.Module(body=[run_fn], type_ignores=[])
)

ns = dict(base.__dict__)
ns.update({
    "GRIP_LOCAL_POS": GRIP_LOCAL_POS,
    "GRIP_LOCAL_QUAT_WXYZ": GRIP_LOCAL_QUAT_WXYZ,
    "MAX_PREGRASP_POSITION_ERROR_M":
        MAX_PREGRASP_POSITION_ERROR_M,
    "MAX_PREGRASP_ORIENTATION_ERROR_DEG":
        MAX_PREGRASP_ORIENTATION_ERROR_DEG,
})

exec(compile(program, str(TRACKER), "exec"), ns)
closer_run = ns["run"]

# Preserve the full 120 Hz recorder and add a compact posture log.
POSTURE = OUT / "raven_closer_operator_posture.csv"

def monitored_run(sim, scene, actor_root):
    robot = scene["robot"]
    tripod = scene["tripod"]
    original_update = scene.update

    rb = {
        name: i for i, name in enumerate(
            robot.data.body_names
        )
    }
    tb = {
        name: i for i, name in enumerate(
            tripod.data.body_names
        )
    }

    j = robot.data.joint_names.index(
        "left_elbow_joint"
    )
    pan = tripod.data.joint_names.index("pan_joint")
    tilt = tripod.data.joint_names.index("tilt_joint")

    tick = 0

    with POSTURE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        columns = (
            "time_s",
            "root_to_handle_xy_m",
            "elbow_below_torso_m",
            "elbow_joint_deg",
            "pan_deg",
            "tilt_deg",
            "hand_to_handle_m",
        )

        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()

        def record_update(*args, **kwargs):
            nonlocal tick

            result = original_update(*args, **kwargs)
            tick += 1

            if tick % base.TRACKING_STEPS == 0:
                root = robot.data.root_pose_w[0, :3]

                handle = tripod.data.body_pose_w[
                    0, tb["control_handle_link"], :3
                ]
                elbow = robot.data.body_pose_w[
                    0, rb["left_elbow_link"], :3
                ]
                torso = robot.data.body_pose_w[
                    0, rb["torso_link"], :3
                ]
                hand = robot.data.body_pose_w[
                    0, rb["left_base_link"], :3
                ]

                writer.writerow({
                    "time_s": round(
                        tick * sim.get_physics_dt(), 4
                    ),
                    "root_to_handle_xy_m": round(
                        float(torch.linalg.vector_norm(
                            (root - handle)[:2]
                        )), 5
                    ),
                    "elbow_below_torso_m": round(
                        float(torso[2] - elbow[2]), 5
                    ),
                    "elbow_joint_deg": round(
                        math.degrees(float(
                            robot.data.joint_pos[0, j]
                        )), 3
                    ),
                    "pan_deg": round(
                        math.degrees(float(
                            tripod.data.joint_pos[0, pan]
                        )), 3
                    ),
                    "tilt_deg": round(
                        math.degrees(float(
                            tripod.data.joint_pos[0, tilt]
                        )), 3
                    ),
                    "hand_to_handle_m": round(
                        float(torch.linalg.vector_norm(
                            hand - handle
                        )), 5
                    ),
                })
            return result

        scene.update = record_update

        try:
            return closer_run(sim, scene, actor_root)
        finally:
            scene.update = original_update
            f.flush()

rec.original_run = monitored_run
base.run = rec.wrap_run

(OUT / "raven_closer_operator.config.json").write_text(
    json.dumps({
        "experiment":
            "G1 10-cm closer operator stance with IK-retargeted grasp",
        "robot_forward_shift_m": CLOSER_BY_M,
        "tripod_unchanged": True,
        "validated_tracker_sha256": sha,
        "grip_local_pos_m": GRIP_LOCAL_POS,
        "grip_local_quat_wxyz": GRIP_LOCAL_QUAT_WXYZ,
        "time_to_tracking_s": base.PAN_START_S,
        "pregrasp_pos_threshold_m":
            MAX_PREGRASP_POSITION_ERROR_M,
        "pregrasp_ori_threshold_deg":
            MAX_PREGRASP_ORIENTATION_ERROR_DEG,
        "assisted_grasp_unchanged": True,
        "original_scripts_unchanged": True,
    }, indent=2),
    encoding="utf-8"
)

if __name__ == "__main__":
    print("[CLOSER] G1 root +10 cm toward tripod")
    print("[CLOSER] Re-solving grasp with IK")
    print("[CLOSER] Output:", OUT)

    try:
        base.main()
    finally:
        base.simulation_app.close()
