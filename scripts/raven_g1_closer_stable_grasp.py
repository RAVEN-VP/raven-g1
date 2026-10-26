
"""RAVEN: closer stance, fast grasp, NO loaded post-grasp elbow adjustment.

Safe follow-up to raven_g1_closer_operator.py after its elbow-down IK
nullspace attempt caused +7.49 deg tripod pan drift while locked.

Preserves the existing 10 cm closer stance, IK-targeted approach, original
handle-relative wrist pose, assisted grasp, pan controller, and 120 Hz logging.
Replaces ONLY the dangerous post-grasp elbow-preparation phase with a short
IK hand-pose hold, followed by the established gradual pan release.

Requires the previously saved raven_g1_closer_operator.py alongside it.
Do not change validated raven_g1_actor_tracking.py or rl_data_capture.py.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "raven_g1_closer_operator.py"
if not SOURCE.exists():
    raise SystemExit(f"Missing earlier closer-stance experiment: {SOURCE}")

# Import existing closer-stance/recording setup without running the experiment.
spec = importlib.util.spec_from_file_location("raven_closer_stable_base", SOURCE)
if spec is None or spec.loader is None:
    raise SystemExit(f"Unable to import {SOURCE}")

closer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = closer
spec.loader.exec_module(closer)

base, rec = closer.base, closer.rec

# Brief post-grasp hold, instead of the previous 4-second elbow preparation.
base.ELBOW_PREP_S = 0.75
base.ELBOW_PREP_END_S = base.GRASP_READY_S + base.ELBOW_PREP_S
base.PAN_RELEASE_S = base.ELBOW_PREP_END_S + base.RELEASE_S
base.PAN_START_S = base.PAN_RELEASE_S + base.FREE_SETTLE_S

# The prior experiment already modified the validated tracking function
# in memory to retarget the grip after moving G1 closer.
run_fn = closer.run_fn

prep = [
    node for node in ast.walk(run_fn)
    if isinstance(node, ast.If)
    and ast.unparse(node.test) == "t < ELBOW_PREP_END_S"
]

if len(prep) != 1:
    raise SystemExit("Unexpected validated elbow-preparation control flow")

prep = prep[0]

if len(prep.orelse) != 1 or not isinstance(prep.orelse[0], ast.If):
    raise SystemExit("Expected original soft-release branch")

release = prep.orelse[0]

if ast.unparse(release.test) != "t < PAN_RELEASE_S":
    raise SystemExit("Unexpected soft-release control flow")

# Hold the calibrated grasp without forcing elbow movement.
# Abort if the tripod rotates too much before release.
prep.body = ast.parse('''
phase = "GRASP_HOLD"

if elbow_prep_pan is None:
    elbow_prep_pan = tripod.data.joint_pos[:, pan_id].clone()
    print(
        "[CLOSER SAFE] Grasp attached; "
        "holding pose without elbow adjustment."
    )

drift = math.degrees(float(
    tripod.data.joint_pos[0, pan_id] - elbow_prep_pan[0]
))

if abs(drift) > PAN_LOCK_DRIFT_LIMIT_DEG:
    status = "PRE_RELEASE_PAN_DRIFT"
    print(
        f"[CLOSER SAFE] Stop: pan drift {drift:+.2f}deg "
        f"exceeded {PAN_LOCK_DRIFT_LIMIT_DEG:.2f}deg before release."
    )
    break

p, q = hand_target(torch.zeros((1,), device=sim.device))
command[:, ik_ids] = ik_target(p, q)
''').body

# The original release required successful elbow repositioning.
# Replace that condition with the actual stable grasp posture.
report = [
    node for node in release.body
    if isinstance(node, ast.If)
    and ast.unparse(node.test) == "not prep_reported"
]

if len(report) != 1:
    raise SystemExit("Couldn't find elbow-preparation success gate")

report[0].body = ast.parse('''
posture_reference = robot.data.joint_pos[:, ik_ids].clone()
prep_reported = True

print(
    "[CLOSER SAFE] Stable grip: "
    "gradually releasing passive pan."
)
''').body

# Compile the changed tracking function only in memory.
# The original controller remains untouched on disk.
module = ast.fix_missing_locations(
    ast.Module(body=[run_fn], type_ignores=[])
)

ns = dict(closer.ns)
exec(compile(module, str(closer.TRACKER), "exec"), ns)

closer.closer_run = ns["run"]

# Existing monitored_run resolves closer.closer_run dynamically.
# This preserves the 10 Hz posture log and 120 Hz recorder.
config_file = closer.OUT / "raven_closer_stable.config.json"

config_file.write_text(
    json.dumps({
        "experiment": "Closer operator with stable grasp hold",
        "g1_forward_shift_m": closer.CLOSER_BY_M,
        "validated_tracker_unchanged": True,
        "calibrated_grip_unchanged": True,
        "assisted_grasp_unchanged": True,
        "ik_retargeted_approach_unchanged": True,
        "post_grasp_elbow_adjustment": "disabled",
        "pan_drift_limit_deg": base.PAN_LOCK_DRIFT_LIMIT_DEG,
        "grasp_ready_s": base.GRASP_READY_S,
        "time_to_tracking_s": base.PAN_START_S,
        "output": str(closer.OUT),
    }, indent=2),
    encoding="utf-8"
)

if __name__ == "__main__":
    print("[CLOSER SAFE] G1 +10 cm, fast grasp")
    print("[CLOSER SAFE] Elbow preparation DISABLED")
    print("[CLOSER SAFE] Time to tracking:", base.PAN_START_S, "s")
    print("[CLOSER SAFE] 120 Hz logging ON")
    print("[CLOSER SAFE] Results:", closer.OUT)

    try:
        base.main()
    finally:
        base.simulation_app.close()
