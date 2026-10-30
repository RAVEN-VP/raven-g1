import time
from pathlib import Path

import mujoco
import mujoco.viewer

MODEL_PATH = Path(
    "experiments/ctqi/models/g1_revo2_ctqi.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Start from the model's reference configuration.
data.qpos[:] = model.qpos0
data.qvel[:] = 0

mujoco.mj_forward(model, data)

controllers = []

print("\n========== PD CONTROLLER SETUP ==========")

for aid in range(model.nu):

    if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
        continue

    jid = int(model.actuator_trnid[aid, 0])

    if jid < 0:
        continue

    joint_type = model.jnt_type[jid]

    if joint_type not in (
        mujoco.mjtJoint.mjJNT_HINGE,
        mujoco.mjtJoint.mjJNT_SLIDE
    ):
        continue

    joint_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, jid
    )

    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])

    is_hand = any(
        finger in joint_name
        for finger in (
            "thumb", "index", "middle", "ring", "pinky"
        )
    )

    # Conservative initial gains.
    kp = 2.0 if is_hand else 50.0
    kd = 0.12 if is_hand else 5.0

    gear = float(model.actuator_gear[aid, 0])

    if abs(gear) < 1e-9:
        continue

    target = float(model.qpos0[qadr])

    controllers.append({
        "aid": aid,
        "qadr": qadr,
        "dadr": dadr,
        "target": target,
        "kp": kp,
        "kd": kd,
        "gear": gear,
        "name": joint_name,
    })

    print(
        f"{joint_name}: target={target:.3f}, "
        f"Kp={kp}, Kd={kd}, gear={gear}"
    )

print(f"\nControllers configured: {len(controllers)}")
print(f"Model actuators: {model.nu}")

def apply_controller():

    for c in controllers:

        error = c["target"] - data.qpos[c["qadr"]]
        velocity = data.qvel[c["dadr"]]

        torque = (
            c["kp"] * error
            - c["kd"] * velocity
        )

        control = torque / c["gear"]

        if model.actuator_ctrllimited[c["aid"]]:
            lower, upper = model.actuator_ctrlrange[c["aid"]]
            control = max(lower, min(upper, control))

        data.ctrl[c["aid"]] = control


print("\n========== LAUNCHING STABILISED VIEWER ==========")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        start = time.perf_counter()

        apply_controller()
        mujoco.mj_step(model, data)

        viewer.sync()

        elapsed = time.perf_counter() - start
        remaining = model.opt.timestep - elapsed

        if remaining > 0:
            time.sleep(remaining)

print("Viewer closed.")
