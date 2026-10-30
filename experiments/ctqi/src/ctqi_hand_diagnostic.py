"""Revo2 finger stability isolation test.

Runs with the G1 arms held at their initial configuration.
Does not move or grasp an object. Does not modify Experiment B.
"""

import mujoco
import numpy as np

from ctqi_stage0 import (
    apply_open_hand_pose,
    controlled_joints,
)

from ctqi_experiment_b import (
    MODEL_B,
    build_experiment_model,
    child_bodies,
)


def name(model, kind, index):
    return (
        mujoco.mj_id2name(model, kind, int(index))
        or f"unnamed_{index}"
    )


def describe_dof(model, index):
    if index >= model.nv:
        print(f"DOF {index} does not exist")
        return

    jid = int(model.dof_jntid[index])

    print(
        f"DOF {index}:",
        name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
    )


def initial_contacts(model, data, right_bodies):
    contacts = []

    for i in range(data.ncon):
        c = data.contact[i]

        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])

        if b1 not in right_bodies and b2 not in right_bodies:
            continue

        contacts.append((
            float(c.dist),
            name(model, mujoco.mjtObj.mjOBJ_BODY, b1),
            name(model, mujoco.mjtObj.mjOBJ_BODY, b2),
        ))

    contacts.sort(key=lambda x: x[0])

    print(
        "Initial right-hand contacts:",
        len(contacts)
    )

    for depth, b1, b2 in contacts[:12]:
        print(
            f"  separation={depth*1000:8.2f} mm | "
            f"{b1} <-> {b2}"
        )


def test(model, entries, mode, seconds=2.0):
    data = mujoco.MjData(model)

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    apply_open_hand_pose(model, data)

    mujoco.mj_forward(model, data)

    arms = [
        e for e in entries if not e["hand"]
    ]

    right = [
        e for e in entries
        if e["name"].startswith("right_")
        and e["hand"]
    ]

    left = [
        e for e in entries
        if e["name"].startswith("left_")
        and e["hand"]
    ]

    if len(arms) != 14 or len(right) != 11 or len(left) != 11:
        raise RuntimeError("Unexpected actuator layout")

    initial_fingers = np.array([
        data.qpos[e["q"]] for e in right
    ])

    max_accel = 0.0
    max_velocity = 0.0
    max_finger_deviation = 0.0

    dt = float(model.opt.timestep)

    print(f"\n===== TEST: {mode.upper()} =====")

    for step in range(round(seconds/dt)):
        t_before = float(data.time)

        data.ctrl[:] = 0.0

        # Same arm holding controller as Stage 0.
        # Arms do not receive a changing IK reference.
        for e in arms:
            error = (
                float(model.qpos0[e["q"]])
                - float(data.qpos[e["q"]])
            )

            torque = (
                float(data.qfrc_bias[e["d"]])
                + e["kp"]*error
                - e["kd"]*float(data.qvel[e["d"]])
            )

            data.ctrl[e["aid"]] = np.clip(
                torque/e["gear"],
                e["lo"], e["hi"]
            )

        # The left hand remains constrained open.
        for e in left:
            data.ctrl[e["aid"]] = 0.0

        for e in right:
            if mode == "passive":
                torque = 0.0
            else:
                # Deliberately soft, torque-limited hold.
                # No qfrc_bias compensation.
                error = (
                    e["ref"]
                    - float(data.qpos[e["q"]])
                )

                torque = (
                    0.06*error
                    - 0.008*float(data.qvel[e["d"]])
                )

                torque = float(np.clip(
                    torque, -0.015, 0.015
                ))

            data.ctrl[e["aid"]] = np.clip(
                torque/e["gear"],
                e["lo"], e["hi"]
            )

        mujoco.mj_step(model, data)

        accel = float(np.max(np.abs(data.qacc)))
        velocity = float(np.max(np.abs(data.qvel)))

        finger_values = np.array([
            data.qpos[e["q"]] for e in right
        ])

        finger_deviation = float(np.max(
            np.abs(finger_values-initial_fingers)
        ))

        max_accel = max(max_accel, accel)
        max_velocity = max(max_velocity, velocity)
        max_finger_deviation = max(
            max_finger_deviation,
            finger_deviation
        )

        # Detect MuJoCo numerical resets.
        # They must never be counted as valid trial data.
        time_reset = (
            data.time <= t_before
            or data.time > t_before+1.5*dt
        )

        unstable = (
            time_reset
            or not np.isfinite(data.qacc).all()
            or not np.isfinite(data.qvel).all()
            or accel > 50000.0
            or velocity > 150.0
        )

        if unstable:
            print(
                f"FAILED at step {step}, "
                f"time={t_before:.4f}s"
            )
            print(
                "Max acceleration:",
                round(max_accel, 2),
                "rad/s² or m/s²"
            )
            print(
                "Max velocity:",
                round(max_velocity, 2)
            )
            print(
                "Simulation time reset:",
                time_reset
            )

            dof = int(np.argmax(np.abs(data.qacc)))

            print("Largest acceleration DOF:")
            describe_dof(model, dof)

            return False

    print("PASSED:", seconds, "simulated seconds")
    print(
        "Max acceleration:",
        round(max_accel, 2)
    )
    print(
        "Max velocity:",
        round(max_velocity, 2)
    )
    print(
        "Max right-finger deviation:",
        round(max_finger_deviation, 4),
        "rad"
    )

    return True


def main():
    build_experiment_model(arm_only=False)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_B)
    )

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0

    apply_open_hand_pose(model, data)
    mujoco.mj_forward(model, data)

    print("\n========== REVO2 DIAGNOSTIC ==========")
    print("MuJoCo DOFs:", model.nv)
    print("Physics timestep:", model.opt.timestep)

    describe_dof(model, 44)

    right_hand = model.body(
        "right_hand_base_link"
    ).id

    right_bodies = child_bodies(
        model, right_hand
    )

    initial_contacts(
        model, data, right_bodies
    )

    entries = controlled_joints(model)

    passive_ok = test(
        model, entries, "passive"
    )

    soft_ok = test(
        model, entries, "soft_hold"
    )

    print("\n========== DIAGNOSIS ==========")

    if not passive_ok:
        print(
            "Passive hand is unstable. "
            "Investigate self-collisions, joint limits, "
            "mount geometry and inertial properties."
        )
    elif not soft_ok:
        print(
            "Passive hand is stable; soft actuation "
            "introduces instability. Investigate "
            "actuator gearing and torque control."
        )
    else:
        print(
            "Both configurations are stable. "
            "The previous high-gain/gravity-bias finger "
            "controller is the likely source. "
            "Next test controlled grasp actuation."
        )


if __name__ == "__main__":
    main()
