
"""RAVEN: closer-stance closed-grip alignment and locked-pan stability test.

Requires the EXISTING experimental scripts alongside this file:
  raven_g1_closer_stable_grasp.py
  raven_g1_closer_operator.py
  raven_g1_rl_data_capture.py
  raven_g1_actor_tracking.py

Preserves the validated tracker on disk and keeps G1 +10 cm nearer the tripod.
Retargets continuously to the LIVE handle pose using the closed-grasp
hand/handle offset measured from the prior successful 120 Hz baseline.
Requires a <=12 mm, <=8 deg pre-attachment match and low joint speed.
Immediately after attachment, holds the achieved joint position targets
instead of issuing an IK correction against the locked tripod.

This is an experimental assisted-grasp configuration, not learned control.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "raven_g1_closer_stable_grasp.py"
if not SOURCE.is_file():
    raise SystemExit(f"Save the prior stable-grasp experiment first: {SOURCE}")

spec = importlib.util.spec_from_file_location("raven_alignment_parent", SOURCE)
if spec is None or spec.loader is None:
    raise SystemExit("Could not load stable-grasp experiment")
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)

closer, base, rec = parent.closer, parent.base, parent.rec

# Calibrated against the successfully CLOSED original grasp at t=18 s.
# This is LEFT BASE LINK relative to the handle: it is NOT a fingertip pose.
CLOSED_GRIP_LOCAL_POS = (-0.0882, +0.0413, +0.0566)
CLOSED_GRIP_LOCAL_QUAT_WXYZ = (-0.005, -0.700, +0.699, -0.147)
MAX_POSITION_ERROR_M = 0.012
MAX_ORIENTATION_ERROR_DEG = 8.0
MAX_IK_JOINT_SPEED_RAD_S = 0.40
MAX_PREGRASP_PAN_SPEED_DEG_S = 1.5

# Still faster than the original 27.5s preparation, but allow the controller
# to reach a stable hand pose before the joint is physically attached.
base.MOVE_S = 6.0
base.PRE_CLOSE_S = 1.0
base.CLOSE_S = 3.0
base.GRASP_SETTLE_S = 2.0
base.ELBOW_PREP_S = 0.75
base.RELEASE_S = 2.0
base.FREE_SETTLE_S = 1.0
base.GRASP_READY_S = (base.MOVE_S + base.PRE_CLOSE_S
                      + base.CLOSE_S + base.GRASP_SETTLE_S)
base.ELBOW_PREP_END_S = base.GRASP_READY_S + base.ELBOW_PREP_S
base.PAN_RELEASE_S = base.ELBOW_PREP_END_S + base.RELEASE_S
base.PAN_START_S = base.PAN_RELEASE_S + base.FREE_SETTLE_S

OUT = (Path.cwd() / "raven_training_results" / "closer_grasp_alignment"
       / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
OUT.mkdir(parents=True, exist_ok=False)
rec.OUT = OUT
closer.OUT = OUT
closer.POSTURE = OUT / "raven_closer_operator_posture.csv"
base.args.output_dir = str(OUT)

run_fn = parent.run_fn
loops = [node for node in ast.walk(run_fn)
         if isinstance(node, ast.While)
         and "simulation_app.is_running()" in ast.unparse(node.test)]
if len(loops) != 1:
    raise SystemExit("Unexpected simulation control-loop structure")
loop = loops[0]

# Avoid abruptly increasing the elbow clamp at the instant of grasp.
clamps = [(i, node) for i, node in enumerate(loop.body)
          if isinstance(node, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == "elbow_floor"
                  for t in node.targets)]
if len(clamps) != 1:
    raise SystemExit("Could not locate experimental elbow-floor assignment")
loop.body[clamps[0][0]:clamps[0][0]+1] = ast.parse('''
elbow_floor = math.radians(ELBOW_WORKSPACE_MIN_DEG)
''').body

# Follow the CURRENT handle pose instead of its initial position.
reaches = [(i, node) for i, node in enumerate(loop.body)
           if isinstance(node, ast.If)
           and ast.unparse(node.test) == "t < GRASP_READY_S"]
if len(reaches) != 1:
    raise SystemExit("Could not locate IK approach override")
loop.body[reaches[0][0]:reaches[0][0]+1] = ast.parse('''
if t < GRASP_READY_S:
    handle_now = tripod.data.body_pose_w[:, handle_body]
    desired_hand_p, desired_hand_q = combine_frame_transforms(
        handle_now[:, :3], handle_now[:, 3:7], local_grip_p, local_grip_q
    )
    alpha = smoothstep(t / MOVE_S)
    target_p = initial_hand_p + alpha * (desired_hand_p - initial_hand_p)
    hemisphere = torch.sign(
        (initial_hand_q * desired_hand_q).sum(-1, keepdim=True)
    )
    hemisphere = torch.where(hemisphere == 0,
                             torch.ones_like(hemisphere), hemisphere)
    target_q = torch.nn.functional.normalize(
        (1.0-alpha) * initial_hand_q
        + alpha * desired_hand_q * hemisphere, dim=-1
    )
    command[:, ik_ids] = ik_target(target_p, target_q)
''').body

# Require accurate and stationary alignment before attaching a fixed joint.
grips = [n for n in ast.walk(run_fn)
         if isinstance(n, ast.If) and ast.unparse(n.test) == "not grip_attached"]
if len(grips) != 1:
    raise SystemExit("Could not locate assisted grasp attachment")
grip = grips[0]
hand_idx = next((i for i, n in enumerate(grip.body)
                 if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "hand"
                         for t in n.targets)), None)
grip_idx = next((i for i, n in enumerate(grip.body)
                 if isinstance(n, ast.Assign)
                 and "grip_pos, grip_quat" in ast.unparse(n)), None)
if hand_idx is None or grip_idx is None or grip_idx <= hand_idx:
    raise SystemExit("Unexpected grasp alignment/attachment structure")

check = ast.parse('''
current_grip_p, current_grip_q = combine_frame_transforms(
    handle[:, :3], handle[:, 3:7], local_grip_p, local_grip_q
)
position_error_m = float(torch.linalg.vector_norm(
    hand[:, :3] - current_grip_p, dim=-1)[0])
orientation_dot = abs(float((hand[:, 3:7] * current_grip_q).sum()))
orientation_error_deg = math.degrees(
    2.0 * math.acos(min(1.0, max(0.0, orientation_dot))))
joint_speed = float(robot.data.joint_vel[:, ik_ids].abs().max())
pan_speed = math.degrees(float(tripod.data.joint_vel[0, pan_id]))
print(f"[GRIP CHECK] Error={position_error_m*1000:.1f} mm, "
      f"orientation={orientation_error_deg:.1f} deg, "
      f"max joint speed={joint_speed:.3f} rad/s, "
      f"pan speed={pan_speed:+.2f} deg/s")
if (position_error_m > MAX_POSITION_ERROR_M
        or orientation_error_deg > MAX_ORIENTATION_ERROR_DEG
        or joint_speed > MAX_IK_JOINT_SPEED_RAD_S
        or abs(pan_speed) > MAX_PREGRASP_PAN_SPEED_DEG_S):
    status = "PREGRASP_ALIGNMENT_FAILED"
    print("[GRIP CHECK] No assisted joint attached: "
          "hand or tripod not yet aligned/stationary.")
    break
''').body
grip.body = grip.body[:hand_idx+1] + check + grip.body[grip_idx:]

# Hold the achieved joint positions rather than moving the arm against
# the still-locked tripod immediately after attachment.
prep = [n for n in ast.walk(run_fn)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "t < ELBOW_PREP_END_S"]
if len(prep) != 1:
    raise SystemExit("Could not locate stable-grasp hold phase")
prep[0].body = ast.parse('''
phase = "GRASP_HOLD"
if elbow_prep_pan is None:
    elbow_prep_pan = tripod.data.joint_pos[:, pan_id].clone()
    posture_reference = robot.data.joint_pos[:, ik_ids].clone()
    print("[GRIP CHECK] Assisted joint attached; holding achieved IK targets.")
drift = math.degrees(float(
    tripod.data.joint_pos[0, pan_id] - elbow_prep_pan[0]
))
if abs(drift) > PAN_LOCK_DRIFT_LIMIT_DEG:
    status = "PRE_RELEASE_PAN_DRIFT"
    print(f"[GRIP CHECK] Stop: locked pan drift={drift:+.2f} deg "
          f"(limit {PAN_LOCK_DRIFT_LIMIT_DEG:.2f} deg)")
    break
command[:, ik_ids] = posture_reference
''').body

# Compile the experiment in memory, leaving the original controller intact.
ns = dict(parent.ns)
for name in (
    "MOVE_S", "PRE_CLOSE_S", "CLOSE_S", "GRASP_SETTLE_S", "GRASP_READY_S",
    "ELBOW_PREP_S", "ELBOW_PREP_END_S", "PAN_RELEASE_S", "PAN_START_S",
    "RELEASE_S", "FREE_SETTLE_S",
):
    ns[name] = getattr(base, name)
ns.update({
    "GRIP_LOCAL_POS": CLOSED_GRIP_LOCAL_POS,
    "GRIP_LOCAL_QUAT_WXYZ": CLOSED_GRIP_LOCAL_QUAT_WXYZ,
    "MAX_POSITION_ERROR_M": MAX_POSITION_ERROR_M,
    "MAX_ORIENTATION_ERROR_DEG": MAX_ORIENTATION_ERROR_DEG,
    "MAX_IK_JOINT_SPEED_RAD_S": MAX_IK_JOINT_SPEED_RAD_S,
    "MAX_PREGRASP_PAN_SPEED_DEG_S": MAX_PREGRASP_PAN_SPEED_DEG_S,
})
module = ast.fix_missing_locations(ast.Module(body=[run_fn], type_ignores=[]))
exec(compile(module, str(closer.TRACKER), "exec"), ns)
closer.closer_run = ns["run"]

(OUT / "raven_closer_alignment.config.json").write_text(json.dumps({
    "experiment": "Closer G1: live-handle closed-grip alignment and pan hold",
    "g1_closer_by_m": closer.CLOSER_BY_M,
    "calibration_pose": "Baseline measured closed grasp around t=18s",
    "grip_local_position_m": CLOSED_GRIP_LOCAL_POS,
    "grip_local_quaternion_wxyz": CLOSED_GRIP_LOCAL_QUAT_WXYZ,
    "max_position_error_m": MAX_POSITION_ERROR_M,
    "max_orientation_error_deg": MAX_ORIENTATION_ERROR_DEG,
    "max_ik_joint_speed_rad_s": MAX_IK_JOINT_SPEED_RAD_S,
    "grasp_ready_s": base.GRASP_READY_S,
    "tracking_starts_s": base.PAN_START_S,
    "baseline_controller_source_unchanged": True,
    "monitoring": "120Hz recorder and 10Hz posture log",
    "output": str(OUT),
}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    print("[GRIP CHECK] +10 cm closer G1; closed-grip target follows live handle")
    print("[GRIP CHECK] Strict alignment and velocity check before attachment")
    print("[GRIP CHECK] No elbow preparation or altered pan-drift threshold")
    print(f"[GRIP CHECK] Tracking planned from {base.PAN_START_S:.2f}s")
    print("[GRIP CHECK] Results:", OUT)
    try:
        base.main()
    finally:
        base.simulation_app.close()
