
"""RAVEN: experimental faster grasp and elbow-down IK posture.

Requires, beside this file:
  raven_g1_actor_tracking.py     (unchanged validated baseline)
  raven_g1_rl_data_capture.py    (unchanged 120 Hz logger)

This script applies strictly isolated runtime changes to the validated controller:
  * shorter staged grasp and pan-release timings
  * an elbow-DOWN nullspace objective rather than elbow-OUT
  * a slightly less flexed elbow preference during tracking

It does NOT flip the elbow joint sign, change the calibrated grip/hand pose,
change camera calibration, adjust PD gains, or remove the grasp safety checks.
The altered experiment uses a separate result directory. The original
controller is never overwritten on disk.

This is an unvalidated physical ergonomics experiment, NOT an RL policy.
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
EXPECTED_TRACKER_SHA256 = (
    "2b7da3c47a8614b690bb20f2f461d0bb93a90a90b9f0c95a3d6c6b4648ecd038"
)
for required in (TRACKER, RECORDER):
    if not required.is_file():
        raise SystemExit(f"Missing required baseline script: {required}")

actual_sha = hashlib.sha256(TRACKER.read_bytes()).hexdigest()
if actual_sha != EXPECTED_TRACKER_SHA256:
    raise SystemExit(
        "Validated tracker differs from the recorded baseline. Refusing "
        "to patch an unfamiliar implementation. Original SHA256: " + actual_sha
    )

# Import the validated 120 Hz recorder, which launches Isaac Sim and imports
# the unchanged original tracker. Its own wrappers will still capture all
# G1 actions and post-physics state transitions.
spec = importlib.util.spec_from_file_location("raven_ergo_recorder", RECORDER)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot load recorder {RECORDER}")
rec = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rec
spec.loader.exec_module(rec)
base = rec.base

OUT = (
    Path.cwd()
    / "raven_training_results"
    / "ergonomic_fast_grasp"
    / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
)
OUT.mkdir(parents=True, exist_ok=False)
rec.OUT = OUT
base.args.output_dir = str(OUT)

# Baseline: 8 + 2 + 6 + 2 + 6 + 2.5 + 1 = 27.5 s.
# Experiment: 4.5 + .5 + 2.5 + .75 + 4 + 1.5 + .75 = 14.5 s.
# Slowing / speeding these phases never alters physics (120 Hz).
base.MOVE_S = 4.5
base.PRE_CLOSE_S = 0.5
base.CLOSE_S = 2.5
base.GRASP_SETTLE_S = 0.75
base.ELBOW_PREP_S = 4.0
base.RELEASE_S = 1.5
base.FREE_SETTLE_S = 0.75
base.GRASP_READY_S = (
    base.MOVE_S + base.PRE_CLOSE_S + base.CLOSE_S + base.GRASP_SETTLE_S
)
base.ELBOW_PREP_END_S = base.GRASP_READY_S + base.ELBOW_PREP_S
base.PAN_RELEASE_S = base.ELBOW_PREP_END_S + base.RELEASE_S
base.PAN_START_S = base.PAN_RELEASE_S + base.FREE_SETTLE_S

# The prior 2-cm OUTWARD goal is replaced by a 3-cm DOWNWARD goal.
# This changes only the nullspace objective and its scalar reference; IK
# continues to prioritise the calibrated grip position and orientation.
base.ELBOW_OUT_GOAL_M = 0.03
base.ELBOW_PREFERRED_DEG = 105.0  # previous: 110 degrees

# Compile a copy of the existing run() function in memory, replacing ONLY
# its nested elbow-gradient helper. Avoid duplicating/reimplementing its
# extensive validated tracking, grasp, IK, perception and safety logic.
tracker_ast = ast.parse(TRACKER.read_text(encoding="utf-8"), filename=str(TRACKER))
run_nodes = [
    node for node in tracker_ast.body
    if isinstance(node, ast.FunctionDef) and node.name == "run"
]
if len(run_nodes) != 1:
    raise SystemExit("Expected exactly one validated run() function")
run_fn = run_nodes[0]
old_helpers = [
    (index, node) for index, node in enumerate(run_fn.body)
    if isinstance(node, ast.FunctionDef) and node.name == "elbow_gradient"
]
if len(old_helpers) != 1:
    raise SystemExit("Expected exactly one elbow_gradient() helper")

# Use measured link Jacobians for the signed downward direction: increasing
# (torso_z - elbow_z) corresponds to lowering the elbow relative to the torso.
# The existing IK projects this gradient into the end-effector nullspace.
helper = ast.parse('''
def elbow_gradient():
    torso = robot.data.body_pose_w[:, torso_body]
    elbow = robot.data.body_pose_w[:, elbow_body]
    clearance = torso[:, 2] - elbow[:, 2]
    jac = robot.root_physx_view.get_jacobians()
    elbow_j = jac[:, elbow_jac_id, :, ik_ids]
    torso_j = jac[:, torso_jac_id, :, ik_ids]
    downward_gradient = torso_j[:, 2, :] - elbow_j[:, 2, :]
    return clearance, downward_gradient
''').body[0]
run_fn.body[old_helpers[0][0]] = helper

# Keep console messages accurate after changing the nullspace objective.
for node in ast.walk(run_fn):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = (
            node.value.replace("[ELBOW] outward=", "[ELBOW] downward=")
            .replace("Elbow-out preparation", "Elbow-down preparation")
        )

module = ast.fix_missing_locations(ast.Module(body=[run_fn], type_ignores=[]))
exec(compile(module, str(TRACKER), "exec"), base.__dict__)
ergonomic_run = base.run

# Add a compact 10 Hz posture diagnostic alongside the recorder's 120 Hz JSONL.
# The diagnostic wraps the original scene.update AFTER each physics step.
POSTURE_CSV = OUT / "raven_elbow_posture_10hz.csv"


def run_with_posture_log(sim, scene, actor_root):
    robot = scene["robot"]
    tripod = scene["tripod"]
    body_names = list(robot.data.body_names)
    joint_names = list(robot.data.joint_names)
    elbow_body = body_names.index("left_elbow_link")
    hand_body = body_names.index("left_base_link")
    torso_body = body_names.index("torso_link")
    elbow_joint = joint_names.index("left_elbow_joint")
    pan_joint = list(tripod.data.joint_names).index("pan_joint")
    tilt_joint = list(tripod.data.joint_names).index("tilt_joint")
    original_update = scene.update
    tick = 0
    fields = (
        "time_s", "phase", "elbow_height_m", "hand_height_m",
        "torso_height_m", "elbow_minus_hand_m", "elbow_minus_torso_m",
        "elbow_joint_deg", "tripod_pan_deg", "tripod_tilt_deg"
    )
    with POSTURE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        def update_and_monitor(*args, **kwargs):
            nonlocal tick
            result = original_update(*args, **kwargs)
            tick += 1
            if tick % base.TRACKING_STEPS == 0:
                t = tick * base.DT
                eb = float(robot.data.body_pose_w[0, elbow_body, 2])
                hand = float(robot.data.body_pose_w[0, hand_body, 2])
                torso = float(robot.data.body_pose_w[0, torso_body, 2])
                if t < base.MOVE_S:
                    phase = "approach"
                elif t < base.GRASP_READY_S:
                    phase = "close_and_settle"
                elif t < base.ELBOW_PREP_END_S:
                    phase = "elbow_down"
                elif t < base.PAN_START_S:
                    phase = "pan_release"
                else:
                    phase = "tracking"
                writer.writerow({
                    "time_s": round(t, 4),
                    "phase": phase,
                    "elbow_height_m": round(eb, 5),
                    "hand_height_m": round(hand, 5),
                    "torso_height_m": round(torso, 5),
                    "elbow_minus_hand_m": round(eb-hand, 5),
                    "elbow_minus_torso_m": round(eb-torso, 5),
                    "elbow_joint_deg": round(math.degrees(float(
                        robot.data.joint_pos[0, elbow_joint])), 3),
                    "tripod_pan_deg": round(math.degrees(float(
                        tripod.data.joint_pos[0, pan_joint])), 3),
                    "tripod_tilt_deg": round(math.degrees(float(
                        tripod.data.joint_pos[0, tilt_joint])), 3),
                })
            return result

        scene.update = update_and_monitor
        try:
            return ergonomic_run(sim, scene, actor_root)
        finally:
            scene.update = original_update
            stream.flush()


# The 120 Hz recorder's wrapper calls rec.original_run. Route it through our
# experimental IK and posture monitor, and preserve rec.wrap_run as entrypoint.
rec.original_run = run_with_posture_log
base.run = rec.wrap_run

config = {
    "experiment": "RAVEN faster grasp / downward-elbow posture ablation",
    "validated_tracker_sha256": actual_sha,
    "baseline_unmodified_on_disk": True,
    "physics_hz": base.PHYSICS_HZ,
    "grasp_calibration_unchanged": True,
    "assisted_fixed_joint_unchanged": True,
    "pd_controller_unchanged": True,
    "calibrated_camera_unchanged": True,
    "pose_optimisation": "IK hand pose constrained; elbow-down nullspace objective",
    "elbow_downward_goal_cm": base.ELBOW_OUT_GOAL_M * 100,
    "preferred_elbow_deg": base.ELBOW_PREFERRED_DEG,
    "timings_s": {
        "approach": base.MOVE_S,
        "pre_close": base.PRE_CLOSE_S,
        "close_fingers": base.CLOSE_S,
        "grasp_settle": base.GRASP_SETTLE_S,
        "elbow_prep": base.ELBOW_PREP_S,
        "soft_pan_release": base.RELEASE_S,
        "free_settle": base.FREE_SETTLE_S,
        "time_to_tracking": base.PAN_START_S,
    },
    "posture_log": str(POSTURE_CSV),
    "notes": (
        "A lower elbow is a soft nullspace objective, not a forced inverted "
        "elbow joint. Collision clearance and grasp safety still require "
        "experimental validation."
    ),
}
(OUT / "raven_ergonomic_fast_grasp.config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

if __name__ == "__main__":
    print("[RAVEN ERGO] Independent faster-grasp / elbow-down experiment")
    print(f"[RAVEN ERGO] Time to tracking: {base.PAN_START_S:.2f}s "
          "(baseline 27.50s)")
    print("[RAVEN ERGO] Calibrated grip, IK hand constraint and safety retained")
    print("[RAVEN ERGO] This does not invert the elbow joint")
    print("[RAVEN ERGO] Output:", OUT)
    try:
        base.main()
    finally:
        base.simulation_app.close()
