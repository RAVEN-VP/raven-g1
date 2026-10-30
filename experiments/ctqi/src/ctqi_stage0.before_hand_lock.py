
"""RAVEN CTQI Stage 0: G1 + Revo2 simulation stabilisation.

Fixed lower-body stance, dynamic arms and Revo2 hands.
Includes provisional open-hand configuration and physics telemetry.

Engineering validation only; not CTQI experimental evidence.
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


# ============================================================
# FIXED LOWER-BODY CONFIGURATION
# ============================================================

LEG_JOINTS = [
    f"{side}_{joint}_joint"
    for side in ("left", "right")
    for joint in (
        "hip_pitch",
        "hip_roll",
        "hip_yaw",
        "knee",
        "ankle_pitch",
        "ankle_roll",
    )
]

LOCK_JOINTS = LEG_JOINTS + [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]


# ============================================================
# REVO2 OPEN-HAND CONFIGURATION
# ============================================================

# Provisional joint-space reference for the Revo2 MJCF.
# These are NOT calibrated physical Revo2 motor commands.
#
# The hand has 11 simulated joints per side, whereas the
# hardware interface has six command channels per hand.

HAND_OPEN_POSE = {
    f"{side}_{joint}_joint": angle
    for side in ("left", "right")
    for joint, angle in {
        "thumb_metacarpal": 0.25,
        "thumb_proximal": 0.20,
        "thumb_distal": 0.15,
        "index_proximal": 0.15,
        "index_distal": 0.10,
        "middle_proximal": 0.15,
        "middle_distal": 0.10,
        "ring_proximal": 0.15,
        "ring_distal": 0.10,
        "pinky_proximal": 0.15,
        "pinky_distal": 0.10,
    }.items()
}


def apply_open_hand_pose(model, data):
    """Initialise left/right Revo2 joints to a neutral pose."""

    for joint_name, target in HAND_OPEN_POSE.items():

        jid = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )

        if jid < 0:
            raise RuntimeError(
                f"Missing Revo2 joint: {joint_name}"
            )

        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(
                f"Expected hinge joint: {joint_name}"
            )

        lower, upper = model.jnt_range[jid]

        if not lower <= target <= upper:
            raise RuntimeError(
                f"{joint_name}: target outside joint range"
            )

        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])

        data.qpos[qadr] = target
        data.qvel[dadr] = 0.0


# ============================================================
# XML HELPERS
# ============================================================

def ensure_section(root, tag):

    existing = root.find(tag)

    if existing is not None:
        return existing

    section = ET.Element(tag)

    order = (
        "compiler", "option", "size", "visual",
        "statistic", "default", "asset", "worldbody",
        "contact", "equality", "tendon", "actuator",
        "sensor", "keyframe", "custom"
    )

    rank = order.index(tag)

    index = next(
        (
            i for i, item in enumerate(root)
            if item.tag in order
            and order.index(item.tag) > rank
        ),
        len(root)
    )

    root.insert(index, section)

    return section


# ============================================================
# BUILD STAGE 0 MODEL
# ============================================================

def build():

    tree = ET.parse(ORIGINAL)
    root = tree.getroot()

    # Move the entire tabletop environment forward.
    # Preserve relative positions between objects.

    dx = 0.40

    for name in (
        "teleop_table",
        "teleop_cube",
        "teleop_cylinder",
        "teleop_handle",
    ):

        body = root.find(
            f".//body[@name='{name}']"
        )

        if body is None:
            raise RuntimeError(
                f"Missing model body: {name}"
            )

        xyz = np.fromstring(
            body.get("pos"),
            sep=" "
        )

        xyz[0] += dx

        body.set(
            "pos",
            " ".join(f"{v:.8g}" for v in xyz)
        )

    # Fix the lower-body joints for this manipulation study.

    equality = ensure_section(root, "equality")

    for joint_name in LOCK_JOINTS:

        if root.find(
            f".//joint[@name='{joint_name}']"
        ) is None:
            raise RuntimeError(
                f"Missing stance joint: {joint_name}"
            )

        ET.SubElement(
            equality,
            "joint",
            name="stage0_lock_" + joint_name,
            joint1=joint_name,
            polycoef="0 0 0 0 0",
        )

    # Exclude only known mounting-adjacent collision pairs.
    # Hand/object contact remains enabled.

    contact = ensure_section(root, "contact")

    for side in ("left", "right"):

        ET.SubElement(
            contact,
            "exclude",
            name=f"stage0_{side}_mounting_interface",
            body1=f"{side}_wrist_pitch_link",
            body2=f"{side}_thumb_metacarpal_link",
        )

    ET.SubElement(
        contact,
        "exclude",
        name="stage0_left_palm_thumb_adjacent",
        body1="left_hand_base_link",
        body2="left_thumb_proximal_link",
    )

    STAGE0.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ET.indent(tree, space="  ")

    tree.write(
        STAGE0,
        encoding="utf-8",
        xml_declaration=True,
    )

    return STAGE0


# ============================================================
# BUILD ACTUATOR CONTROL TABLE
# ============================================================

def controlled_joints(model):

    entries = []

    for aid in range(model.nu):

        if (
            model.actuator_trntype[aid]
            != mujoco.mjtTrn.mjTRN_JOINT
        ):
            continue

        jid = int(model.actuator_trnid[aid, 0])

        if jid < 0:
            continue

        if (
            model.jnt_type[jid]
            != mujoco.mjtJoint.mjJNT_HINGE
        ):
            continue

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            jid,
        )

        if name in LOCK_JOINTS:
            continue

        is_hand = name in HAND_OPEN_POSE

        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])

        if model.actuator_ctrllimited[aid]:
            lower, upper = model.actuator_ctrlrange[aid]
        else:
            lower, upper = -np.inf, np.inf

        # Stage 0 gains; provisional rather than calibrated.

        kp, kd = (
            (8.0, 0.25)
            if is_hand
            else (55.0, 6.0)
        )

        # Use the explicit open-hand target when applicable.

        reference = float(
            HAND_OPEN_POSE.get(
                name,
                model.qpos0[qadr],
            )
        )

        entries.append({
            "aid": aid,
            "name": name,
            "q": qadr,
            "d": dadr,
            "hand": is_hand,
            "kp": kp,
            "kd": kd,
            "ref": reference,
            "lo": float(lower),
            "hi": float(upper),
            "gear": float(model.actuator_gear[aid, 0]),
            "joint_range": np.array(
                model.jnt_range[jid],
                dtype=float,
            ),
        })

    return entries


# ============================================================
# TARGET GENERATION
# ============================================================

def target_for(entry, t, mode):

    base = entry["ref"]

    # Hold all joints at their reference configuration.

    if mode == "hold":
        return base

    # Motion mode:
    # Small controlled arm movement and finger flexion.

    phase = max(0.0, t - 1.5)

    if entry["hand"]:

        activation = np.clip(
            phase / 1.5,
            0.0,
            1.0,
        )

        if t > 4.5:
            activation = np.clip(
                1.0 - (t - 4.5) / 1.5,
                0.0,
                1.0,
            )

        joint_range = entry["joint_range"]

        flexion = (
            0.18
            * (joint_range[1] - joint_range[0])
            * activation
        )

        return float(
            np.clip(
                base + flexion,
                joint_range[0],
                joint_range[1],
            )
        )

    if (
        "shoulder_pitch" in entry["name"]
        or "elbow" in entry["name"]
    ):

        if t >= 1.5:
            return (
                base
                + 0.08
                * math.sin(
                    2.0 * math.pi * 0.3 * phase
                )
            )

    return base


# ============================================================
# RUN PHYSICS SIMULATION
# ============================================================

def run(args):

    build()

    model = mujoco.MjModel.from_xml_path(
        str(STAGE0)
    )

    data = mujoco.MjData(model)

    # Initialise G1 to its reference configuration.

    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0

    # Initialise both Revo2 hands.

    apply_open_hand_pose(model, data)

    mujoco.mj_forward(model, data)

    entries = controlled_joints(model)

    if len(entries) != 36:
        raise RuntimeError(
            f"Expected 14 arm + 22 hand motors, "
            f"found {len(entries)}"
        )

    print("\n========== CTQI STAGE 0 ==========")
    print(f"Joints: {model.njnt}")
    print(f"Motors: {model.nu}")
    print("Controlling: 14 arm + 22 Revo2 joints")
    print("Stance: 12 leg + 3 waist joints constrained")
    print("Table moved forward: 0.40 m")
    print("Open-hand pose: enabled")
    print("Hand control: independent simulated joint torque")
    print("Physical six-channel Revo2 mapping: not calibrated")

    viewer_cm = None
    viewer = None

    if args.viewer:

        # Do not import mujoco.viewer locally:
        # that would shadow the mujoco module inside run().

        from mujoco import viewer as mjviewer

        viewer_cm = mjviewer.launch_passive(
            model,
            data,
        )

        viewer = viewer_cm.__enter__()

    rows = []

    step_total = round(
        args.seconds / model.opt.timestep
    )

    try:

        for step in range(step_total):

            t = float(data.time)

            targets = {}

            for entry in entries:

                target = target_for(
                    entry,
                    t,
                    args.mode,
                )

                targets[entry["name"]] = target

                error = (
                    target
                    - float(data.qpos[entry["q"]])
                )

                velocity = float(
                    data.qvel[entry["d"]]
                )

                # Gravity and velocity-bias compensation,
                # plus proportional/derivative feedback.

                torque = (
                    float(data.qfrc_bias[entry["d"]])
                    + entry["kp"] * error
                    - entry["kd"] * velocity
                )

                gear = entry["gear"]

                if abs(gear) < 1e-9:
                    raise RuntimeError(
                        f"Zero actuator gear: {entry['name']}"
                    )

                data.ctrl[entry["aid"]] = np.clip(
                    torque / gear,
                    entry["lo"],
                    entry["hi"],
                )

            mujoco.mj_step(model, data)

            if (
                not np.all(np.isfinite(data.qpos))
                or not np.all(np.isfinite(data.qvel))
            ):
                raise RuntimeError(
                    f"Non-finite simulation at {data.time:.4f}s"
                )

            # Log at approximately 50 Hz.

            log_interval = max(
                1,
                round(0.02 / model.opt.timestep),
            )

            if step % log_interval == 0:

                arm_errors = [
                    targets[e["name"]]
                    - float(data.qpos[e["q"]])
                    for e in entries
                    if not e["hand"]
                ]

                hand_errors = [
                    targets[e["name"]]
                    - float(data.qpos[e["q"]])
                    for e in entries
                    if e["hand"]
                ]

                cube = data.xpos[
                    model.body("teleop_cube").id
                ].copy()

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
                    "worst_penetration_m": min(
                        (
                            float(c.dist)
                            for c in data.contact
                        ),
                        default=0.0,
                    ),
                }

                # Actual G1 wrist positions.

                for side in ("left", "right"):

                    wrist = data.xpos[
                        model.body(
                            f"{side}_wrist_yaw_link"
                        ).id
                    ].copy()

                    for i, dimension in enumerate("xyz"):
                        row[
                            f"{side}_wrist_{dimension}_m"
                        ] = float(wrist[i])

                # All dynamically controlled joint states.

                for entry in entries:

                    name = entry["name"]

                    row[f"{name}_q_rad"] = float(
                        data.qpos[entry["q"]]
                    )

                    row[f"{name}_target_rad"] = (
                        targets[name]
                    )

                    row[f"{name}_torque_cmd_Nm"] = float(
                        data.ctrl[entry["aid"]]
                        * entry["gear"]
                    )

                rows.append(row)

            if viewer is not None:

                if not viewer.is_running():
                    break

                viewer.sync()

                time.sleep(
                    max(0.0, model.opt.timestep)
                )

    finally:

        if viewer_cm is not None:
            viewer_cm.__exit__(
                None,
                None,
                None,
            )

    if not rows:
        raise RuntimeError(
            "Simulation produced no telemetry"
        )

    # ========================================================
    # SAVE TELEMETRY
    # ========================================================

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULTS / (
        f"{args.mode}_telemetry.csv"
    )

    with path.open("w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    arm_rms = float(
        np.mean([
            row["arm_tracking_rms_rad"]
            for row in rows
        ])
    )

    hand_rms = float(
        np.mean([
            row["hand_tracking_rms_rad"]
            for row in rows
        ])
    )

    first_cube = np.array([
        rows[0][f"cube_{i}_m"]
        for i in "xyz"
    ])

    last_cube = np.array([
        rows[-1][f"cube_{i}_m"]
        for i in "xyz"
    ])

    summary = {
        "mode": args.mode,
        "seconds_simulated": float(data.time),
        "samples": len(rows),
        "fixed_stance": True,
        "open_hand_pose": True,
        "virtual_hand_actuation": (
            "22 independent simulated joints, provisional"
        ),
        "arm_tracking_rms_rad": arm_rms,
        "hand_tracking_rms_rad": hand_rms,
        "worst_penetration_m": float(
            min(
                row["worst_penetration_m"]
                for row in rows
            )
        ),
        "cube_displacement_m": float(
            np.linalg.norm(
                last_cube - first_cube
            )
        ),
        "scientific_status": (
            "ENGINEERING SHAKEOUT ONLY. "
            "NOT CTQI VALIDATION."
        ),
    }

    summary_path = RESULTS / (
        f"{args.mode}_summary.json"
    )

    summary_path.write_text(
        json.dumps(summary, indent=2)
    )

    print("\n========== SIMULATION SUMMARY ==========")
    print(json.dumps(summary, indent=2))
    print(f"\nTelemetry: {path}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["hold", "motion"],
        default="motion",
    )

    parser.add_argument(
        "--seconds",
        type=float,
        default=7.0,
    )

    parser.add_argument(
        "--viewer",
        action="store_true",
    )

    run(parser.parse_args())
