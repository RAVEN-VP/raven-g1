#!/usr/bin/env python3
"""RAVEN CTQI Stage 0: fixed stance, dynamic G1 arms, neutral Revo2 hands.

Builds a separate stage-0 XML from g1_revo2_ctqi.xml. The unified source model
is never modified. The Revo2 hands use six master joints per hand and five
distal mimic constraints per hand.

Run from any working directory:
    python experiments/ctqi/src/ctqi_stage0.py --mode motion --viewer
"""

import argparse
import csv
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / "experiments/ctqi/models/g1_revo2_ctqi.xml"
STAGE0 = ROOT / "experiments/ctqi/models/g1_revo2_ctqi_stage0.xml"
RESULTS = ROOT / "experiments/ctqi/results/stage0"

LEG_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch", "hip_roll", "hip_yaw",
        "knee", "ankle_pitch", "ankle_roll",
    )
]
LOCK_JOINTS = LEG_JOINTS + [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

MASTER_SUFFIXES = (
    "thumb_metacarpal",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
)

MIMIC_SPECS = (
    ("thumb_distal", "thumb_proximal", 1.0),
    ("index_distal", "index_proximal", 1.155),
    ("middle_distal", "middle_proximal", 1.155),
    ("ring_distal", "ring_proximal", 1.155),
    ("pinky_distal", "pinky_proximal", 1.155),
)

HAND_MASTER_NAMES = {
    f"{side}_{suffix}_joint"
    for side in ("left", "right")
    for suffix in MASTER_SUFFIXES
}
HAND_MIMIC_NAMES = {
    f"{side}_{distal}_joint"
    for side in ("left", "right")
    for distal, _proximal, _multiplier in MIMIC_SPECS
}


def ensure_section(root, tag):
    existing = root.find(tag)
    if existing is not None:
        return existing

    order = (
        "compiler", "option", "size", "visual", "statistic",
        "default", "asset", "worldbody", "contact", "equality",
        "tendon", "actuator", "sensor", "keyframe", "custom",
    )
    rank = order.index(tag)
    section = ET.Element(tag)
    index = next(
        (
            i for i, child in enumerate(root)
            if child.tag in order and order.index(child.tag) > rank
        ),
        len(root),
    )
    root.insert(index, section)
    return section


def build():
    if not ORIGINAL.exists():
        raise FileNotFoundError(
            f"Unified source model not found: {ORIGINAL}\n"
            "Run build_unified_model.py first."
        )

    tree = ET.parse(ORIGINAL)
    root = tree.getroot()
    equality = ensure_section(root, "equality")

    # Do not silently continue with an independently actuated 22-joint hand.
    mimic_names = {
        item.get("name")
        for item in equality.findall("joint")
        if item.get("name", "").startswith("revo2_mimic_")
    }
    if len(mimic_names) != 10:
        raise RuntimeError(
            f"Expected 10 Revo2 mimic constraints in the unified model; "
            f"found {len(mimic_names)}. Rebuild the unified model first."
        )

    # Lock the G1 lower body/waist and only the 12 independent hand channels.
    # The 10 distal joints remain governed by their mimic constraints.
    joints_to_lock = LOCK_JOINTS + sorted(HAND_MASTER_NAMES)

    for joint_name in joints_to_lock:
        if root.find(f".//joint[@name='{joint_name}']") is None:
            raise RuntimeError(f"Missing joint required by Stage 0: {joint_name}")

        constraint_name = "stage0_lock_" + joint_name
        if any(
            item.get("name") == constraint_name
            for item in equality.findall("joint")
        ):
            raise RuntimeError(
                f"Stage-0 lock already exists in source model: {constraint_name}"
            )

        ET.SubElement(
            equality,
            "joint",
            {
                "name": constraint_name,
                "joint1": joint_name,
                "polycoef": "0 0 0 0 0",
                "solref": "0.005 1",
            },
        )

    STAGE0.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(STAGE0, encoding="utf-8", xml_declaration=True)
    return STAGE0


def joint_id(model, name):
    jid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if jid < 0:
        raise RuntimeError(f"Missing joint in compiled model: {name}")
    return jid


def apply_neutral_hand_pose(model, data):
    """Set the six master channels to model qpos0 and satisfy mimic qpos."""
    qpos0 = model.qpos0

    for side in ("left", "right"):
        for suffix in MASTER_SUFFIXES:
            name = f"{side}_{suffix}_joint"
            jid = joint_id(model, name)
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            data.qpos[qadr] = qpos0[qadr]
            data.qvel[dadr] = 0.0

        for distal_suffix, proximal_suffix, multiplier in MIMIC_SPECS:
            distal_name = f"{side}_{distal_suffix}_joint"
            proximal_name = f"{side}_{proximal_suffix}_joint"

            distal_id = joint_id(model, distal_name)
            proximal_id = joint_id(model, proximal_name)

            distal_q = int(model.jnt_qposadr[distal_id])
            proximal_q = int(model.jnt_qposadr[proximal_id])
            distal_d = int(model.jnt_dofadr[distal_id])
            proximal_d = int(model.jnt_dofadr[proximal_id])

            # MuJoCo equality coefficients are relative to qpos0.
            data.qpos[distal_q] = (
                qpos0[distal_q]
                + multiplier * (data.qpos[proximal_q] - qpos0[proximal_q])
            )
            data.qvel[distal_d] = multiplier * data.qvel[proximal_d]


def controlled_joints(model):
    entries = []

    for aid in range(model.nu):
        if model.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT:
            continue

        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0 or model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            continue

        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, jid
        )

        if name in LOCK_JOINTS:
            continue

        if name in HAND_MIMIC_NAMES:
            raise RuntimeError(
                f"Mimic joint must not have its own actuator: {name}"
            )

        hand = name in HAND_MASTER_NAMES
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])

        if model.actuator_ctrllimited[aid]:
            lower, upper = model.actuator_ctrlrange[aid]
        else:
            lower, upper = -np.inf, np.inf

        kp, kd = (8.0, 0.25) if hand else (55.0, 6.0)

        entries.append({
            "aid": aid,
            "name": name,
            "q": qadr,
            "d": dadr,
            "hand": hand,
            "kp": kp,
            "kd": kd,
            "ref": float(model.qpos0[qadr]),
            "lo": float(lower),
            "hi": float(upper),
            "gear": float(model.actuator_gear[aid, 0]),
            "joint_range": np.array(model.jnt_range[jid], dtype=float),
        })

    return entries


def target_for(entry, t, mode):
    base = entry["ref"]
    if mode == "hold":
        return base

    phase = max(0.0, t - 1.5)

    # Stage 0 is only a small deterministic arm tracking shakeout.
    # The Revo2 master joints remain in their neutral open configuration.
    if "shoulder_pitch" in entry["name"] or "elbow" in entry["name"]:
        if t >= 1.5:
            return base + 0.08 * math.sin(2.0 * math.pi * 0.3 * phase)

    return base


def body_name_for_geom(model, geom_id):
    body_id = int(model.geom_bodyid[geom_id])
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, body_id
    )
    return name or f"body_{body_id}"


def geom_name(model, geom_id):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
    )
    return name or f"geom_{geom_id}"


def is_hand_body(name):
    return any(
        token in name
        for token in (
            "hand_base_link", "thumb", "index",
            "middle", "ring", "pinky",
        )
    )


def contact_stats(model, data):
    cube_hand = 0
    cube_robot = 0
    rows = []

    for contact in data.contact:
        g1 = geom_name(model, int(contact.geom1))
        g2 = geom_name(model, int(contact.geom2))
        b1 = body_name_for_geom(model, int(contact.geom1))
        b2 = body_name_for_geom(model, int(contact.geom2))

        cube_pair = "teleop_cube_geom" in (g1, g2)
        if cube_pair:
            other_body = b2 if g1 == "teleop_cube_geom" else b1
            if is_hand_body(other_body):
                cube_hand += 1
            elif other_body.startswith(("left_", "right_")):
                cube_robot += 1

        rows.append({
            "geom1": g1,
            "body1": b1,
            "geom2": g2,
            "body2": b2,
            "distance_m": float(contact.dist),
        })

    return cube_hand, cube_robot, rows


def print_initial_contacts(model, data):
    cube_hand, cube_robot, contacts = contact_stats(model, data)

    print(f"\\nInitial MuJoCo contacts: {len(contacts)}")
    print(f"Initial cube-to-hand contacts: {cube_hand}")
    print(f"Initial cube-to-robot contacts: {cube_robot}")

    for item in contacts:
        print(
            f"  {item['geom1']} [{item['body1']}] <-> "
            f"{item['geom2']} [{item['body2']}] "
            f"dist={item['distance_m']:+.6f} m"
        )

    if cube_hand:
        print(
            "WARNING: the cube is already contacting a Revo2 hand at t=0. "
            "Do not start a grasp trial until this initial overlap is resolved."
        )

    return cube_hand, cube_robot, contacts


def run(args):
    if args.seconds <= 0:
        raise ValueError("--seconds must be greater than zero.")

    build()

    model = mujoco.MjModel.from_xml_path(str(STAGE0))
    data = mujoco.MjData(model)

    expected = {
        "njnt": (model.njnt, 54),
        "nq": (model.nq, 72),
        "nv": (model.nv, 69),
        "nu": (model.nu, 41),
    }
    for label, (actual, target) in expected.items():
        if actual != target:
            raise RuntimeError(
                f"Stage-0 model {label} mismatch: expected {target}, got {actual}."
            )

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    apply_neutral_hand_pose(model, data)
    mujoco.mj_forward(model, data)

    entries = controlled_joints(model)
    if len(entries) != 26:
        raise RuntimeError(
            f"Expected 14 arm + 12 independent hand channels; found {len(entries)}."
        )

    print("\n========== CTQI STAGE 0 ==========")
    print(f"Joints: {model.njnt}; qpos: {model.nq}; qvel: {model.nv}")
    print(f"Actuators: {model.nu} (29 G1 + 12 independent Revo2)")
    print(f"Equality constraints: {model.neq}")
    print("Fixed stance: 12 leg + 3 waist joints")
    print("Hands: 6 master joints per hand held at neutral qpos0")
    print("Mimic: 5 distal joints per hand coupled to their proximal joints")
    print("Table/object positions: unchanged from the unified source model")

    initial_cube_hand_contacts, initial_cube_robot_contacts, initial_contacts = (
        print_initial_contacts(model, data)
    )

    viewer_cm = None
    viewer = None
    if args.viewer:
        from mujoco import viewer as mjviewer
        viewer_cm = mjviewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    rows = []
    step_total = round(args.seconds / model.opt.timestep)
    log_interval = max(1, round(0.02 / model.opt.timestep))

    try:
        for step in range(step_total):
            t = float(data.time)
            targets = {}

            for entry in entries:
                if entry["hand"]:
                    # Stage 0 deliberately holds the hand in its neutral pose.
                    targets[entry["name"]] = entry["ref"]
                    data.ctrl[entry["aid"]] = 0.0
                    continue

                target = target_for(entry, t, args.mode)
                targets[entry["name"]] = target

                error = target - float(data.qpos[entry["q"]])
                velocity = float(data.qvel[entry["d"]])

                torque = (
                    float(data.qfrc_bias[entry["d"]])
                    + entry["kp"] * error
                    - entry["kd"] * velocity
                )

                if abs(entry["gear"]) < 1e-9:
                    raise RuntimeError(
                        f"Zero actuator gear: {entry['name']}"
                    )

                data.ctrl[entry["aid"]] = np.clip(
                    torque / entry["gear"],
                    entry["lo"],
                    entry["hi"],
                )

            mujoco.mj_step(model, data)

            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
            ):
                raise RuntimeError(
                    f"Non-finite simulation state at {data.time:.4f}s"
                )

            if step % log_interval == 0:
                arm_errors = [
                    targets[e["name"]] - float(data.qpos[e["q"]])
                    for e in entries if not e["hand"]
                ]
                hand_errors = [
                    targets[e["name"]] - float(data.qpos[e["q"]])
                    for e in entries if e["hand"]
                ]

                cube = data.xpos[
                    model.body("teleop_cube").id
                ].copy()

                cube_hand, cube_robot, _ = contact_stats(model, data)

                row = {
                    "t_s": round(float(data.time), 4),
                    "arm_tracking_rms_rad": float(
                        np.sqrt(np.mean(np.square(arm_errors)))
                    ),
                    "hand_tracking_rms_rad": float(
                        np.sqrt(np.mean(np.square(hand_errors)))
                    ),
                    "cube_x_m": float(cube[0]),
                    "cube_y_m": float(cube[1]),
                    "cube_z_m": float(cube[2]),
                    "contacts": int(data.ncon),
                    "cube_hand_contacts": int(cube_hand),
                    "cube_robot_contacts": int(cube_robot),
                    "worst_penetration_m": min(
                        (float(contact.dist) for contact in data.contact),
                        default=0.0,
                    ),
                }

                for side in ("left", "right"):
                    wrist = data.xpos[
                        model.body(f"{side}_wrist_yaw_link").id
                    ].copy()
                    for i, dimension in enumerate("xyz"):
                        row[f"{side}_wrist_{dimension}_m"] = float(wrist[i])

                for entry in entries:
                    name = entry["name"]
                    row[f"{name}_q_rad"] = float(data.qpos[entry["q"]])
                    row[f"{name}_target_rad"] = targets[name]
                    row[f"{name}_torque_cmd_Nm"] = float(
                        data.ctrl[entry["aid"]] * entry["gear"]
                    )

                # Record the dependent joint states as well, so the mimic
                # relationship can be checked directly in the telemetry.
                for side in ("left", "right"):
                    for distal_suffix, _proximal_suffix, _multiplier in MIMIC_SPECS:
                        name = f"{side}_{distal_suffix}_joint"
                        jid = joint_id(model, name)
                        qadr = int(model.jnt_qposadr[jid])
                        row[f"{name}_q_rad"] = float(data.qpos[qadr])

                rows.append(row)

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep))

    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)

    if not rows:
        raise RuntimeError("Simulation produced no telemetry.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    telemetry_path = RESULTS / f"{args.mode}_telemetry.csv"

    with telemetry_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    first_cube = np.array([rows[0][f"cube_{axis}_m"] for axis in "xyz"])
    last_cube = np.array([rows[-1][f"cube_{axis}_m"] for axis in "xyz"])

    summary = {
        "mode": args.mode,
        "seconds_simulated": float(data.time),
        "samples": len(rows),
        "fixed_stance": True,
        "neutral_hands_held": True,
        "hand_actuators": 12,
        "mimic_constraints": 10,
        "arm_tracking_rms_rad": float(np.mean([
            row["arm_tracking_rms_rad"] for row in rows
        ])),
        "hand_tracking_rms_rad": float(np.mean([
            row["hand_tracking_rms_rad"] for row in rows
        ])),
        "worst_penetration_m": float(min(
            row["worst_penetration_m"] for row in rows
        )),
        "cube_displacement_m": float(np.linalg.norm(last_cube - first_cube)),
        "initial_cube_hand_contacts": int(initial_cube_hand_contacts),
        "initial_cube_robot_contacts": int(initial_cube_robot_contacts),
        "initial_contact_pairs": initial_contacts,
        "scientific_status": (
            "ENGINEERING SHAKEOUT ONLY. NOT CTQI VALIDATION."
        ),
    }

    summary_path = RESULTS / f"{args.mode}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n========== SIMULATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nTelemetry: {telemetry_path}")
    print(f"Summary:   {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["hold", "motion"], default="motion"
    )
    parser.add_argument("--seconds", type=float, default=7.0)
    parser.add_argument("--viewer", action="store_true")
    run(parser.parse_args())
