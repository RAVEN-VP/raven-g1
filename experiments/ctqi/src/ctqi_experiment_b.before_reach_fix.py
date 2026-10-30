"""CTQI Experiment B: physics-based manipulation feasibility.

G1 arms and right Revo2 fingers are dynamically controlled.
The left hand and lower-body stance remain constrained.

The first object is a cube for controlled contact and placement.
Task outcomes come from MuJoCo physics, not CTQI values.

This is a feasibility baseline, not final CTQI validation.
"""

import argparse
import csv
import heapq
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ctqi_stage0 import (
    STAGE0, HAND_OPEN_POSE, LOCK_JOINTS,
    apply_open_hand_pose,
    build as build_stage0,
    controlled_joints,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_B = ROOT / "models/g1_revo2_ctqi_b.xml"
OUT = ROOT / "results/experiment_b"
DT = 0.02

PHASES = [
    ("settle", 0.0, 0.8),
    ("reach", 0.8, 2.4),
    ("align", 2.4, 3.5),
    ("grip", 3.5, 4.5),
    ("hold", 4.5, 6.5),
    ("release", 6.5, 7.5),
    ("withdraw", 7.5, 8.5),
]


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3.0 - 2.0*x)


def phase_at(t):
    for name, start, end in PHASES:
        if start <= t < end:
            return name
    return "finished"


# ============================================================
# EXPERIMENT MODEL
# ============================================================

def build_experiment_model():

    build_stage0()

    tree = ET.parse(STAGE0)
    root = tree.getroot()

    equality = root.find("equality")

    if equality is None:
        raise RuntimeError("Stage 0 constraints missing")

    # Unlock ONLY the right Revo2 finger joints.
    # Preserve the left-hand and fixed-stance constraints.
    removed = 0

    for item in list(equality):
        if item.get("name", "").startswith(
            "stage0_open_right_"
        ):
            equality.remove(item)
            removed += 1

    if removed != 11:
        raise RuntimeError(
            f"Expected 11 right-hand locks, found {removed}"
        )

    locked_stance = sum(
        item.get("name", "").startswith("stage0_lock_")
        for item in equality
    )

    if locked_stance != 15:
        raise RuntimeError("Fixed stance was not preserved")

    # Controlled benchmark object, positioned on the table.
    cube = root.find(".//body[@name='teleop_cube']")

    if cube is None:
        raise RuntimeError("Cube body missing")

    cube_position = np.array([0.52, -0.14, 0.755])

    cube.set(
        "pos",
        " ".join(str(v) for v in cube_position)
    )

    root.set("model", "g1_revo2_ctqi_experiment_b")

    MODEL_B.parent.mkdir(parents=True, exist_ok=True)

    ET.indent(tree, space="  ")

    tree.write(
        MODEL_B,
        encoding="utf-8",
        xml_declaration=True,
    )

    return cube_position


# ============================================================
# RIGHT-ARM INVERSE KINEMATICS
# ============================================================

def solve_ik(model, seed_qpos, arms, body_id, target):

    data = mujoco.MjData(model)
    data.qpos[:] = seed_qpos

    right_arm = [
        e for e in arms
        if e["name"].startswith("right_")
    ]

    dofs = [e["d"] for e in right_arm]

    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))

    for _ in range(70):

        mujoco.mj_forward(model, data)

        error = target - data.xpos[body_id]

        if np.linalg.norm(error) < 0.003:
            break

        mujoco.mj_jacBody(
            model, data,
            jac_pos, jac_rot,
            body_id,
        )

        J = jac_pos[:, dofs]

        damping = 0.015

        dq = J.T @ np.linalg.solve(
            J @ J.T + damping*damping*np.eye(3),
            error,
        )

        for entry, change in zip(right_arm, dq):

            jid = int(
                model.actuator_trnid[entry["aid"], 0]
            )

            lower, upper = model.jnt_range[jid]

            data.qpos[entry["q"]] = np.clip(
                data.qpos[entry["q"]]
                + np.clip(change, -0.035, 0.035),
                lower + 0.02,
                upper - 0.02,
            )

    mujoco.mj_forward(model, data)

    residual = float(
        np.linalg.norm(
            target - data.xpos[body_id]
        )
    )

    targets = {
        e["name"]: float(data.qpos[e["q"]])
        for e in arms
    }

    return targets, residual


# ============================================================
# NOMINAL TELEOPERATION INTENT
# ============================================================

def plan_trajectory(model, data, arms, hand_id, cube_pos):

    initial_qpos = data.qpos.copy()
    initial_hand = data.xpos[hand_id].copy()

    pregrasp = cube_pos + np.array(
        [-0.19, 0.0, 0.05]
    )

    grasp = cube_pos + np.array(
        [-0.105, 0.0, 0.03]
    )

    lifted = grasp + np.array(
        [0.075, 0.0, 0.10]
    )

    placed = grasp + np.array(
        [0.085, 0.0, 0.0]
    )

    withdraw = placed + np.array(
        [-0.10, 0.0, 0.12]
    )

    keyframes = [
        (0.0, initial_hand),
        (0.8, initial_hand),
        (2.4, pregrasp),
        (3.5, grasp),
        (4.5, grasp),
        (6.5, lifted),
        (7.5, placed),
        (8.5, withdraw),
    ]

    times = np.arange(
        round(8.5 / DT)
    ) * DT

    names = [e["name"] for e in arms]

    desired_positions = []
    joint_commands = []
    ik_errors = []

    seed = initial_qpos.copy()

    for t in times:

        segment = next(
            i for i in range(len(keyframes)-1)
            if keyframes[i][0] <= t < keyframes[i+1][0]
        )

        t0, p0 = keyframes[segment]
        t1, p1 = keyframes[segment+1]

        alpha = smoothstep(
            (t - t0) / (t1 - t0)
        )

        target = p0 + alpha*(p1-p0)

        solved, error = solve_ik(
            model, seed, arms, hand_id, target
        )

        for e in arms:
            seed[e["q"]] = solved[e["name"]]

        desired_positions.append(target)
        joint_commands.append([
            solved[name] for name in names
        ])
        ik_errors.append(error)

    return (
        times,
        np.asarray(desired_positions),
        np.asarray(joint_commands),
        float(max(ik_errors)),
    )


# ============================================================
# REVO2 GRIP SCHEDULE
# ============================================================

def grip_fraction(t):

    if t < 3.5:
        return 0.0

    if t < 4.5:
        return float(smoothstep((t-3.5)/1.0))

    if t < 6.5:
        return 1.0

    if t < 7.5:
        return float(1.0-smoothstep((t-6.5)/1.0))

    return 0.0


# ============================================================
# INDEPENDENT CONTACT MEASUREMENT
# ============================================================

def child_bodies(model, root_body):

    ids = {root_body}

    for bid in range(root_body+1, model.nbody):
        parent = int(model.body_parentid[bid])

        if parent in ids:
            ids.add(bid)

    return ids


def object_contacts(model, data, object_id, hand_bodies):

    count = 0
    total_force = 0.0

    for i in range(data.ncon):

        contact = data.contact[i]

        body1 = int(
            model.geom_bodyid[contact.geom1]
        )

        body2 = int(
            model.geom_bodyid[contact.geom2]
        )

        touching = (
            (body1 == object_id and body2 in hand_bodies)
            or
            (body2 == object_id and body1 in hand_bodies)
        )

        if not touching:
            continue

        count += 1

        force = np.zeros(6)

        mujoco.mj_contactForce(
            model, data, i, force
        )

        total_force += max(0.0, float(force[0]))

    return count, total_force


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================

def run(args):

    cube_initial = build_experiment_model()

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_B)
    )

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    apply_open_hand_pose(model, data)
    mujoco.mj_forward(model, data)

    entries = controlled_joints(model)

    arms = [e for e in entries if not e["hand"]]

    right_fingers = [
        e for e in entries
        if e["hand"] and e["name"].startswith("right_")
    ]

    left_fingers = [
        e for e in entries
        if e["hand"] and e["name"].startswith("left_")
    ]

    if (
        len(arms) != 14
        or len(right_fingers) != 11
        or len(left_fingers) != 11
    ):
        raise RuntimeError(
            "Incorrect arm/Revo2 actuator configuration"
        )

    right_id = model.body("right_hand_base_link").id
    left_id = model.body("left_hand_base_link").id
    cube_id = model.body("teleop_cube").id

    right_hand_bodies = child_bodies(
        model, right_id
    )

    times, intended, targets, ik_error = plan_trajectory(
        model, data, arms, right_id, cube_initial
    )

    print("\n========== EXPERIMENT B ==========")
    print("G1 fixed stance: enabled")
    print("G1 arms: dynamic")
    print("Right Revo2 hand: dynamic")
    print("Left Revo2 hand: constrained")
    print("Object physics: enabled")
    print("Nominal maximum IK error:", round(ik_error, 4), "m")

    if ik_error > 0.08:
        print(
            "WARNING: Nominal target may be unreachable."
        )

    viewer_cm = None
    viewer = None

    if args.viewer:

        from mujoco import viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(
            model, data
        )

        viewer = viewer_cm.__enter__()

    physics_dt = float(model.opt.timestep)
    stride = round(DT / physics_dt)

    if abs(stride*physics_dt-DT) > 1e-8:
        raise RuntimeError(
            "Unexpected physics timestep"
        )

    cube_start = data.xpos[cube_id].copy()

    command = targets[0].copy()
    last_source_time = 0.0

    queue = []
    sequence = 0

    rows = []
    max_lift = 0.0
    hold_contact_samples = 0

    try:

        for step in range(len(times)*stride):

            t = float(data.time)

            if step % stride == 0:

                index = min(
                    step // stride,
                    len(times)-1,
                )

                intended_target = intended[index].copy()

                if args.condition == "spatial":
                    intended_target += np.array([
                        0.0,
                        0.035*args.severity,
                        0.0,
                    ])

                if args.condition == "oscillation":
                    intended_target += np.array([
                        0.0,
                        0.0,
                        0.015*args.severity*
                        math.sin(2*math.pi*3*t),
                    ])

                if args.condition in (
                    "spatial", "oscillation"
                ):

                    solved, _ = solve_ik(
                        model,
                        data.qpos.copy(),
                        arms,
                        right_id,
                        intended_target,
                    )

                    next_command = np.array([
                        solved[e["name"]]
                        for e in arms
                    ])

                else:
                    next_command = targets[index].copy()

                delay = (
                    0.12*args.severity
                    if args.condition == "latency"
                    else 0.0
                )

                heapq.heappush(
                    queue,
                    (
                        t+delay,
                        sequence,
                        t,
                        next_command,
                    ),
                )

                sequence += 1

            while queue and queue[0][0] <= t+1e-9:

                _, _, last_source_time, command = (
                    heapq.heappop(queue)
                )

            # G1 gravity-compensated arm controller.
            for entry, target in zip(arms, command):

                torque = (
                    data.qfrc_bias[entry["d"]]
                    + entry["kp"]*(
                        target-data.qpos[entry["q"]]
                    )
                    - entry["kd"]*data.qvel[entry["d"]]
                )

                data.ctrl[entry["aid"]] = np.clip(
                    torque/entry["gear"],
                    entry["lo"],
                    entry["hi"],
                )

            fraction = grip_fraction(t)

            # Right Revo2: genuine simulated joint actuation.
            for entry in right_fingers:

                reference = entry["ref"]
                maximum = entry["joint_range"][1]

                target = (
                    reference
                    + 0.75*fraction*(maximum-reference)
                )

                torque = (
                    data.qfrc_bias[entry["d"]]
                    + 5.0*(target-data.qpos[entry["q"]])
                    - 0.30*data.qvel[entry["d"]]
                )

                if args.condition == "grip":
                    torque *= max(
                        0.1,
                        1.0-0.8*args.severity,
                    )

                data.ctrl[entry["aid"]] = np.clip(
                    torque/entry["gear"],
                    entry["lo"],
                    entry["hi"],
                )

            # Left Revo2 remains physically constrained.
            for entry in left_fingers:
                data.ctrl[entry["aid"]] = 0.0

            mujoco.mj_step(model, data)

            if (
                not np.isfinite(data.qpos).all()
                or not np.isfinite(data.qvel).all()
            ):
                raise RuntimeError(
                    f"Unstable physics at {data.time:.3f}s"
                )

            if step % stride != 0:
                continue

            index = min(
                step // stride,
                len(times)-1,
            )

            right_pos = data.xpos[right_id].copy()
            left_pos = data.xpos[left_id].copy()
            cube_pos = data.xpos[cube_id].copy()

            contacts, force = object_contacts(
                model,
                data,
                cube_id,
                right_hand_bodies,
            )

            lift = float(
                cube_pos[2]-cube_start[2]
            )

            max_lift = max(max_lift, lift)

            if 4.5 <= t < 6.5 and contacts > 0:
                hold_contact_samples += 1

            row = {
                "time_s": t,
                "phase": phase_at(t),

                "desired_x_m": float(intended[index, 0]),
                "desired_y_m": float(intended[index, 1]),
                "desired_z_m": float(intended[index, 2]),

                "right_hand_x_m": float(right_pos[0]),
                "right_hand_y_m": float(right_pos[1]),
                "right_hand_z_m": float(right_pos[2]),

                "left_hand_x_m": float(left_pos[0]),
                "left_hand_y_m": float(left_pos[1]),
                "left_hand_z_m": float(left_pos[2]),

                "wrist_error_m": float(
                    np.linalg.norm(
                        intended[index]-right_pos
                    )
                ),

                "cube_x_m": float(cube_pos[0]),
                "cube_y_m": float(cube_pos[1]),
                "cube_z_m": float(cube_pos[2]),
                "cube_lift_m": lift,

                "hand_cube_contact_count": contacts,
                "hand_cube_normal_force_N": force,
                "grasp_fraction_cmd": fraction,

                "command_source_time_s": last_source_time,
                "command_age_ms": 1000.0*(
                    data.time-last_source_time
                ),
            }

            # Log all 14 dynamically controlled arm joints.
            for j, entry in enumerate(arms):

                name = entry["name"]

                row[name+"_q_rad"] = float(
                    data.qpos[entry["q"]]
                )

                row[name+"_dq_rad_s"] = float(
                    data.qvel[entry["d"]]
                )

                row[name+"_target_rad"] = float(
                    command[j]
                )

                row[name+"_motor_Nm"] = float(
                    data.ctrl[entry["aid"]]*entry["gear"]
                )

            # Record both Revo2 hand joint states.
            for entry in right_fingers+left_fingers:

                name = entry["name"]

                row[name+"_q_rad"] = float(
                    data.qpos[entry["q"]]
                )

                row[name+"_dq_rad_s"] = float(
                    data.qvel[entry["d"]]
                )

                row[name+"_motor_Nm"] = float(
                    data.ctrl[entry["aid"]]*entry["gear"]
                )

            rows.append(row)

            if viewer is not None:

                if not viewer.is_running():
                    break

                viewer.sync()
                time.sleep(DT)

    finally:

        if viewer_cm is not None:
            viewer_cm.__exit__(
                None, None, None
            )

    if not rows:
        raise RuntimeError("No telemetry produced")

    # ========================================================
    # INDEPENDENT OUTCOME EVALUATION
    # ========================================================

    OUT.mkdir(parents=True, exist_ok=True)

    filename = (
        f"b_{args.condition}_"
        f"s{args.severity:.2f}_seed{args.seed}"
    )

    telemetry_path = OUT / (
        filename+"_telemetry.csv"
    )

    with telemetry_path.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    final_cube = np.array([
        rows[-1]["cube_x_m"],
        rows[-1]["cube_y_m"],
        rows[-1]["cube_z_m"],
    ])

    placement_target = (
        cube_initial+np.array([0.085, 0.0, 0.0])
    )

    final_error = float(
        np.linalg.norm(
            final_cube-placement_target
        )
    )

    lifted = max_lift >= 0.05
    retained_contact = hold_contact_samples >= 10

    placed = (
        final_error < 0.06
        and
        abs(final_cube[2]-cube_initial[2]) < 0.035
    )

    success = bool(
        lifted and retained_contact and placed
    )

    summary = {
        "condition": args.condition,
        "severity": args.severity,
        "seed": args.seed,

        "samples": len(rows),
        "max_ik_error_m": ik_error,

        "actual_max_cube_lift_m": max_lift,
        "actual_hold_contact_samples": hold_contact_samples,

        "final_cube_position_m": final_cube.tolist(),
        "target_cube_position_m": placement_target.tolist(),
        "final_cube_error_m": final_error,

        "lifted": bool(lifted),
        "contact_retained": bool(retained_contact),
        "placed": bool(placed),
        "task_success": success,

        "outcome_source": (
            "Independent MuJoCo object poses and contacts"
        ),

        "limitations": (
            "Provisional Revo2 mounting and independent "
            "joint actuation; no calibrated six-channel "
            "hardware command mapping."
        ),

        "status": (
            "Experiment B feasibility baseline; "
            "not yet CTQI predictive validation."
        ),
    }

    summary_path = OUT / (
        filename+"_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("\n========== EXPERIMENT B SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print("Results:", OUT)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    parser.add_argument(
        "--condition",
        choices=[
            "baseline",
            "latency",
            "spatial",
            "oscillation",
            "grip",
        ],
        default="baseline",
    )

    parser.add_argument(
        "--severity",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261030,
    )

    run(parser.parse_args())
