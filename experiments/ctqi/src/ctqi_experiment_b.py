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
    STAGE0, MASTER_SUFFIXES, LOCK_JOINTS,
    apply_neutral_hand_pose,
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

def build_experiment_model(arm_only=False):

    build_stage0()

    tree = ET.parse(STAGE0)
    root = tree.getroot()

    equality = root.find("equality")

    if equality is None:
        raise RuntimeError("Stage 0 constraints missing")

    # Six Revo2 master joints on the right are released for Experiment B.
    # The five dependent joints stay coupled through the original mimic EQs.
    right_locks = {
        f"stage0_lock_right_{suffix}_joint" for suffix in MASTER_SUFFIXES
    }
    existing = {element.get("name") for element in equality.findall("joint")}
    if not right_locks.issubset(existing):
        raise RuntimeError("Stage 0 right Revo2 master locks are missing")
    if not {f"stage0_lock_{joint}" for joint in LOCK_JOINTS}.issubset(existing):
        raise RuntimeError("Stage 0 stance locks are missing")
    if not arm_only:
        for element in list(equality):
            if element.get("name") in right_locks:
                equality.remove(element)
    mimic_count = sum(element.get("name", "").startswith("revo2_mimic_")
                      for element in equality.findall("joint"))
    if mimic_count != 10:
        raise RuntimeError(f"Expected 10 Revo2 mimic constraints, got {mimic_count}")

    # Controlled benchmark object, positioned on the table.
    cube = root.find(".//body[@name='teleop_cube']")

    if cube is None:
        raise RuntimeError("Cube body missing")

    # Experiment B uses a 40 mm cube for the Revo2 grasp feasibility test.
    cube_geom = cube.find("./geom[@name='teleop_cube_geom']")
    if cube_geom is None:
        raise RuntimeError("teleop_cube_geom missing")
    cube_geom.set("size", "0.02 0.02 0.02")
    cube_geom.set("mass", "0.03")

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


    # CTQI_B_CUBE_ALIGNED_TARGET_V2
    # The placement search can shift the cube from the original grasp midpoint.
    # The wrist must follow that same shift or the open fingers miss the cube.
    if CALIBRATED_GRASP_TARGET is None:
        raise RuntimeError("Calibrated grasp target unavailable")

    grasp = np.asarray(CALIBRATED_GRASP_TARGET, dtype=float).copy()
    pregrasp = initial_hand + 0.35 * (grasp - initial_hand)

    lifted = grasp + np.array(
        [0.0, 0.0, 0.065]
    )

    # Shorter placement movement to remain within
    # the reachable workspace.
    placed = lifted + np.array(
        [0.040, 0.0, -0.065]
    )

    withdraw = placed + np.array(
        [-0.075, 0.0, 0.085]
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
    worst_ik = (0.0, 0.0, "unknown", None)
    # CTQI_B_IK_PHASE_DIAGNOSTIC_V1

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

        if error > worst_ik[0]:
            worst_ik = (
                float(error),
                float(t),
                f"{t0:.2f}-{t1:.2f}s",
                target.copy(),
            )

    print("\n========== TRAJECTORY IK PREFLIGHT ==========")
    print("Worst IK error (mm):", round(worst_ik[0]*1000, 2))
    print("Time (s):", round(worst_ik[1], 3))
    print("Keyframe interval:", worst_ik[2])
    print("Target XYZ (m):", np.round(worst_ik[3], 4))
    print("Planned lift target:", np.round(lifted, 4))
    print("Planned placement target:", np.round(placed, 4))
    print("Planned withdrawal target:", np.round(withdraw, 4))

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
# REACHABLE WORKSPACE / OBJECT PLACEMENT
# ============================================================

APPROACH_DX = 0.065
CALIBRATED_GRASP_TARGET = None


def finger_link(model, words):

    matches = []

    for bid in range(model.nbody):

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        ) or ""

        if all(word in name for word in words):
            matches.append((name, bid))

    if not matches:
        raise RuntimeError(
            "Missing finger body: " + str(words)
        )

    exact = [
        item for item in matches
        if item[0].endswith("_link")
    ]

    return (exact or matches)[0][1]



def finger_cube_contact_groups(model, data, cube_id):
    """Count actual MuJoCo cube contacts by right-hand finger."""
    groups = {
        name: 0 for name in
        ("thumb", "index", "middle", "ring", "pinky", "other")
    }
    minimum = 0.0

    for c in data.contact:
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])

        other = (
            b2 if b1 == cube_id
            else b1 if b2 == cube_id
            else -1
        )

        if other < 0:
            continue

        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, other
        ) or ""

        if not name.startswith("right_"):
            continue

        group = next(
            (
                f for f in (
                    "thumb", "index", "middle", "ring", "pinky"
                )
                if f in name
            ),
            "other",
        )

        if float(c.dist) <= 0.0005:
            groups[group] += 1
            minimum = min(minimum, float(c.dist))

    return groups, minimum


def virtual_closed_grip_qpos(model, qpos):
    """Build a diagnostic 25-percent closed-hand pose.

    This is a kinematic feasibility test, not a simulation
    of the forces actually produced by the finger controller.
    """
    pose = qpos.copy()

    for suffix in MASTER_SUFFIXES:
        joint = model.joint(f"right_{suffix}_joint")
        adr = int(model.jnt_qposadr[joint.id])
        upper = float(model.jnt_range[joint.id, 1])
        ref = float(qpos[adr])

        pose[adr] = min(
            upper,
            ref + 0.25 * (upper - ref),
        )

    mimics = (
        ("thumb_distal", "thumb_proximal", 1.0),
        ("index_distal", "index_proximal", 1.155),
        ("middle_distal", "middle_proximal", 1.155),
        ("ring_distal", "ring_proximal", 1.155),
        ("pinky_distal", "pinky_proximal", 1.155),
    )

    for distal, proximal, ratio in mimics:
        src = model.joint(f"right_{proximal}_joint")
        dst = model.joint(f"right_{distal}_joint")

        src_adr = int(model.jnt_qposadr[src.id])
        dst_adr = int(model.jnt_qposadr[dst.id])

        pose[dst_adr] = np.clip(
            ratio * pose[src_adr],
            *model.jnt_range[dst.id],
        )

    return pose


def calibrate_object_to_hand(
    model, data, arms, right_hand_id, cube_id
):
    # CTQI_B_OPPOSING_GRASP_V1
    global APPROACH_DX, CALIBRATED_GRASP_TARGET

    start_qpos = data.qpos.copy()
    initial_wrist = data.xpos[right_hand_id].copy()

    table_id = model.body("teleop_table").id
    table_gid = model.geom("teleop_table_top").id
    cube_gid = model.geom("teleop_cube_geom").id

    initial_table_pos = model.body_pos[table_id].copy()
    half = model.geom_size[cube_gid].copy()
    thalf = model.geom_size[table_gid].copy()

    if not np.allclose(
        half, [0.02, 0.02, 0.02], atol=0.001
    ):
        raise RuntimeError(
            f"Expected 40 mm cube, got {half*2000} mm"
        )

    jid = int(model.body_jntadr[cube_id])

    if (
        int(model.body_jntnum[cube_id]) != 1
        or model.jnt_type[jid]
        != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise RuntimeError("Cube freejoint missing")

    qadr = int(model.jnt_qposadr[jid])

    def put_cube(probe, centre):
        probe.qpos[qadr:qadr+3] = centre
        probe.qpos[qadr+3:qadr+7] = [
            1.0, 0.0, 0.0, 0.0
        ]
        probe.qvel[:] = 0.0

    # Find a reachable arm grasp pose.
    chosen = None

    for dx in (0.075, 0.065, 0.050, 0.035):
        target = initial_wrist + np.array(
            [dx, 0.0, 0.0]
        )

        solution, err = solve_ik(
            model,
            start_qpos.copy(),
            arms,
            right_hand_id,
            target,
        )

        print(
            f"Preflight: forward {dx*1000:.0f} mm, "
            f"IK error {err*1000:.1f} mm"
        )

        if err < 0.012:
            chosen = (dx, solution)
            break

    if chosen is None:
        raise RuntimeError(
            "No reachable right-arm grasp pose"
        )

    APPROACH_DX, solution = chosen

    grasp_qpos = start_qpos.copy()

    for e in arms:
        grasp_qpos[e["q"]] = solution[e["name"]]

    closed_qpos = virtual_closed_grip_qpos(
        model, grasp_qpos
    )

    probe = mujoco.MjData(model)
    probe.qpos[:] = grasp_qpos
    mujoco.mj_forward(model, probe)

    wrist_target = probe.xpos[right_hand_id].copy()

    # Actual collision-mesh origins, not visual meshes.
    def pad_origin(finger):
        gid = model.geom(
            f"right_{finger}_touch_link_collision_0"
        ).id
        return probe.geom_xpos[gid].copy()

    thumb = pad_origin("thumb")
    index = pad_origin("index")
    middle = pad_origin("middle")

    seeds = [
        0.5 * (thumb + index),
        0.5 * (thumb + middle),
    ]

    print("\n========== OPPOSING-GRASP PREFLIGHT ==========")
    print("Thumb pad:", np.round(thumb, 4))
    print("Index pad:", np.round(index, 4))
    print("Middle pad:", np.round(middle, 4))
    print("Cube dimensions (mm):", np.round(2*half*1000, 1))

    robot_bodies = set()

    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            bid,
        ) or ""

        if (
            name.startswith(("left_", "right_"))
            or name == "pelvis"
        ):
            robot_bodies.add(bid)

    # Check the cube against the open hand along
    # five interpolated arm configurations.
    def approach_has_penetration(centre):
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            probe.qpos[:] = start_qpos

            for e in arms:
                probe.qpos[e["q"]] = (
                    start_qpos[e["q"]]
                    + fraction * (
                        solution[e["name"]]
                        - start_qpos[e["q"]]
                    )
                )

            put_cube(probe, centre)
            mujoco.mj_forward(model, probe)

            for c in probe.contact:
                b1 = int(model.geom_bodyid[c.geom1])
                b2 = int(model.geom_bodyid[c.geom2])

                other = (
                    b2 if b1 == cube_id
                    else b1 if b2 == cube_id
                    else -1
                )

                if (
                    other in robot_bodies
                    and float(c.dist) < -0.0005
                ):
                    return True

        return False

    # Search relative to the fixed grasp pose.
    # Do not move the wrist and cube together.
    offsets = [
        (dx, dy, dz)
        for dx in (
            0.0, -0.01, 0.01, -0.02, 0.02,
            -0.03, 0.03, -0.04, 0.04
        )
        for dy in (
            0.0, -0.01, 0.01, -0.02, 0.02,
            -0.03, 0.03, -0.04, 0.04
        )
        for dz in (
            0.0, -0.01, 0.01, -0.02, 0.02
        )
    ]

    offsets.sort(
        key=lambda v: sum(x*x for x in v)
    )

    tried = 0
    opposed_candidates = 0
    best = None
    # CTQI_B_REJECTION_DIAGNOSTIC_V1
    rejected_outer = 0
    rejected_deep = 0
    rejected_approach = 0
    rejected_multiple = 0
    candidate_examples = []

    for seed in seeds:
        for offset in offsets:
            centre = seed + np.asarray(offset)

            table_centre = np.array([
                centre[0] + 0.23,
                initial_table_pos[1],
                centre[2] - half[2] - thalf[2],
            ])

            # Keep the object within the tabletop.
            if (
                abs(
                    centre[1] - table_centre[1]
                ) + half[1]
                >= thalf[1] - 0.01
            ):
                continue

            tried += 1
            model.body_pos[table_id] = table_centre

            # Simulate a closed hand kinematically.
            probe.qpos[:] = closed_qpos
            put_cube(probe, centre)
            mujoco.mj_forward(model, probe)

            groups, penetration = (
                finger_cube_contact_groups(
                    model, probe, cube_id
                )
            )

            opposing_contact = (
                groups["thumb"] > 0
                and (
                    groups["index"] > 0
                    or groups["middle"] > 0
                )
            )

            if not opposing_contact:
                continue

            opposed_candidates += 1

            # Diagnose each constraint independently.
            outer = bool(
                groups["ring"] or groups["pinky"]
            )
            deep = penetration < -0.004

            if outer:
                rejected_outer += 1
            if deep:
                rejected_deep += 1

            if outer or deep:
                if outer and deep:
                    rejected_multiple += 1

                if len(candidate_examples) < 8:
                    candidate_examples.append((
                        np.round(centre, 4).tolist(),
                        dict(groups),
                        round(penetration * 1000, 2),
                        "outer/deep"
                    ))
                continue

            if approach_has_penetration(centre):
                rejected_approach += 1

                if len(candidate_examples) < 8:
                    candidate_examples.append((
                        np.round(centre, 4).tolist(),
                        dict(groups),
                        round(penetration * 1000, 2),
                        "approach"
                    ))
                continue

            best = (
                centre.copy(),
                table_centre.copy(),
                groups.copy(),
                np.asarray(offset).copy(),
                penetration,
            )
            break

        if best is not None:
            break

    if best is None:
        model.body_pos[table_id] = initial_table_pos

        print("\n========== GRASP REJECTION ANALYSIS ==========")
        print("Candidates tested:", tried)
        print("Opposing-contact candidates:", opposed_candidates)
        print("Rejected: ring/pinky:", rejected_outer)
        print("Rejected: penetration >4mm:", rejected_deep)
        print("Rejected: open-hand approach:", rejected_approach)
        print("Overlapping outer/deep rejections:", rejected_multiple)

        print("\nExamples:")
        for centre, groups, depth, reason in candidate_examples:
            print(
                f"  centre={centre} "
                f"contacts={groups} "
                f"depth={depth}mm "
                f"reason={reason}"
            )

        raise RuntimeError(
            "No valid opposing-finger grasp found. "
            f"Checked {tried} candidates; "
            f"{opposed_candidates} produced virtual "
            "thumb/index-or-middle contact. "
            "Current grasp geometry or closing "
            "direction may be incompatible."
        )

    (
        cube_position,
        table_centre,
        groups,
        offset,
        penetration,
    ) = best

    model.body_pos[table_id] = table_centre

    data.qpos[:] = start_qpos
    put_cube(data, cube_position)

    model.qpos0[qadr:qadr+3] = cube_position
    model.qpos0[qadr+3:qadr+7] = [
        1.0, 0.0, 0.0, 0.0
    ]

    mujoco.mj_forward(model, data)

    hand_ids = child_bodies(
        model, right_hand_id
    )

    count, _ = object_contacts(
        model, data, cube_id, hand_ids
    )

    if count:
        raise RuntimeError(
            "Initial hand/cube collision detected"
        )

    CALIBRATED_GRASP_TARGET = wrist_target.copy()

    print("Selected cube centre:",
          np.round(cube_position, 4))
    print("Wrist grasp target:",
          np.round(wrist_target, 4))
    print("Virtual closed-hand contacts:", groups)
    print("Maximum virtual penetration (mm):",
          round(penetration*1000, 2))
    print("Offset from pad midpoint (mm):",
          np.round(offset*1000, 1))
    print("Candidates tested:", tried)
    print("Initial real contacts:", count)
    print("Five open-hand approach poses checked")
    print(
        "Table top height (m):",
        round(table_centre[2]+thalf[2], 4)
    )
    print(
        "Kinematic feasibility only; "
        "force closure is not guaranteed."
    )

    return cube_position.copy()


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================

def run(args):

    cube_initial = build_experiment_model(args.arm_only)

    model = mujoco.MjModel.from_xml_path(
        str(MODEL_B)
    )

    # CTQI_B_BOUNDED_HAND_V1
    # Provisional numerical stabilisation, not calibrated
    # Revo2 hardware inertia or joint damping.
    for name in (f"right_{suffix}_joint" for suffix in MASTER_SUFFIXES):
        if not name.startswith("right_"):
            continue

        jid = model.joint(name).id
        dof = int(model.jnt_dofadr[jid])

        model.dof_damping[dof] = max(
            float(model.dof_damping[dof]), 0.08
        )

        model.dof_armature[dof] = max(
            float(model.dof_armature[dof]), 0.001
        )

    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    apply_neutral_hand_pose(model, data)
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
        or len(right_fingers) != 6
        or len(left_fingers) != 6
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

    cube_initial = calibrate_object_to_hand(
        model, data, arms, right_id, cube_id
    )

    times, intended, targets, ik_error = plan_trajectory(
        model, data, arms, right_id, cube_initial
    )

    if ik_error > 0.065:
        raise RuntimeError(
            f"Nominal IK error {ik_error*1000:.1f} mm. "
            "Aborting unreachable trial."
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

    # CTQI_B_STABILITY_FIX_V1
    command = targets[0].copy()
    filtered_command = command.copy()
    command_velocity = np.zeros_like(command)

    # Conservative arm reference limits.
    MAX_REFERENCE_VELOCITY = 0.40   # rad/s
    MAX_REFERENCE_ACCEL = 1.20      # rad/s^2
    last_source_time = 0.0

    queue = []
    sequence = 0

    rows = []
    max_lift = 0.0
    max_supported_hold_lift = 0.0
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

            # Acceleration- and velocity-limited IK targets.
            desired_velocity = np.clip(
                3.0*(command-filtered_command),
                -MAX_REFERENCE_VELOCITY,
                MAX_REFERENCE_VELOCITY,
            )

            command_velocity += np.clip(
                desired_velocity-command_velocity,
                -MAX_REFERENCE_ACCEL*physics_dt,
                MAX_REFERENCE_ACCEL*physics_dt,
            )

            filtered_command += command_velocity*physics_dt

            # G1 gravity-compensated arm controller.
            for entry, target in zip(arms, filtered_command):

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

            # Right Revo2 bounded torque/position control.
            # Arm-only tests leave right fingers locked.
            # No finger qfrc_bias feedforward.

            fraction = grip_fraction(t)

            for entry in right_fingers:

                if args.arm_only:
                    data.ctrl[entry["aid"]] = 0.0
                    continue

                reference = entry["ref"]
                lower, upper = entry["joint_range"]

                # Conservative provisional closing movement.
                # Motor direction is not yet calibrated.
                closing_distance = 0.25 * (
                    upper - reference
                )

                target = np.clip(
                    reference + fraction*closing_distance,
                    lower, upper
                )

                error = (
                    target - data.qpos[entry["q"]]
                )

                velocity = data.qvel[entry["d"]]

                # Low-gain control with strictly bounded torque.
                torque = (
                    0.10*error
                    - 0.025*velocity
                )

                torque = float(np.clip(
                    torque, -0.015, 0.015
                ))

                if args.condition == "grip":
                    torque *= max(
                        0.1, 1.0 - 0.8*args.severity
                    )

                data.ctrl[entry["aid"]] = np.clip(
                    torque/entry["gear"],
                    entry["lo"], entry["hi"]
                )

            # Left Revo2 remains physically constrained.
            for entry in left_fingers:
                data.ctrl[entry["aid"]] = 0.0

            previous_time = float(data.time)

            mujoco.mj_step(model, data)

            # A MuJoCo numerical reset invalidates the trial.
            if data.time <= previous_time:
                raise RuntimeError(
                    "MuJoCo simulation reset at "
                    f"{previous_time:.4f}s. "
                    "Experiment B results are INVALID."
                )

            right_speeds = np.array([
                abs(data.qvel[e["d"]])
                for e in right_fingers
            ])

            if (
                not args.arm_only
                and np.max(right_speeds) > 8.0
            ):
                raise RuntimeError(
                    "Right Revo2 finger velocity exceeded "
                    f"8 rad/s at {data.time:.4f}s. "
                    "Stopping unstable trial."
                )

            if np.max(np.abs(data.qacc)) > 10000.0:
                raise RuntimeError(
                    "Excessive joint acceleration at "
                    f"{data.time:.4f}s. "
                    "Stopping unstable trial."
                )

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

            # CTQI_B_FINGER_CONTACT_DIAGNOSTIC_V1
            if contacts > 0 and (
                phase_at(t) in ("align", "grip", "hold")
            ):
                if step % (stride * 10) == 0:
                    print(
                        f"\\nCONTACT t={t:.2f}s "
                        f"phase={phase_at(t)}"
                    )

                    for ci in range(data.ncon):
                        c = data.contact[ci]

                        b1 = int(model.geom_bodyid[c.geom1])
                        b2 = int(model.geom_bodyid[c.geom2])

                        if cube_id not in (b1, b2):
                            continue

                        other = b2 if b1 == cube_id else b1

                        name = mujoco.mj_id2name(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            other
                        ) or ""

                        if other not in right_hand_bodies:
                            continue

                        print(
                            f"  {name:40s} "
                            f"penetration={min(0.0, c.dist)*1000:.2f}mm"
                        )


            actual_fingers, _ = (
                finger_cube_contact_groups(
                    model, data, cube_id
                )
            )

            opposed_now = bool(
                actual_fingers["thumb"]
                and (
                    actual_fingers["index"]
                    or actual_fingers["middle"]
                )
            )

            lift = float(
                cube_pos[2]-cube_start[2]
            )

            max_lift = max(max_lift, lift)

            if 4.5 <= t < 6.5 and contacts > 0:
                hold_contact_samples += 1
                max_supported_hold_lift = max(max_supported_hold_lift, lift)

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
                "thumb_contact_count": actual_fingers["thumb"],
                "index_contact_count": actual_fingers["index"],
                "middle_contact_count": actual_fingers["middle"],
                "ring_contact_count": actual_fingers["ring"],
                "pinky_contact_count": actual_fingers["pinky"],
                "opposing_finger_contact": int(opposed_now),
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
                    filtered_command[j]
                )

                row[name+"_raw_ik_target_rad"] = float(
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
        f"b_{'arm_only_' if args.arm_only else ''}{args.condition}_"
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
        cube_initial+np.array([0.040, 0.0, 0.0])
    )

    final_error = float(
        np.linalg.norm(
            final_cube-placement_target
        )
    )

    max_opposed_hold_lift = max((
        r["cube_lift_m"] for r in rows
        if r["phase"] == "hold"
        and r["opposing_finger_contact"]
    ), default=0.0)

    opposed_hold_samples = sum(
        int(r["opposing_finger_contact"])
        for r in rows if r["phase"] == "hold"
    )

    lifted = max_opposed_hold_lift >= 0.05
    retained_contact = opposed_hold_samples >= 10

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
        "arm_only": args.arm_only,
        "severity": args.severity,
        "seed": args.seed,

        "samples": len(rows),
        "max_ik_error_m": ik_error,

        "actual_max_cube_lift_m": max_lift,
        "max_contact_supported_hold_lift_m": max_supported_hold_lift,
        "initial_hand_cube_contact_samples": int(
            rows[0]["hand_cube_contact_count"]
        ),
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
            "Provisional Revo2 mounting and six-channel "
            "master actuation with distal mimic joints; no calibrated "
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
        "--arm-only",
        action="store_true",
        help="Keep both Revo2 hands locked to test arm stability",
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
