"""Prepare the official Unitree G1 Mode 15 + BrainCo URDF for Isaac Lab.

The official URDF contains several fingertip joints represented as
revolute joints with zero-width limits.

Isaac Lab/PhysX treats them as articulation DOFs and later rejects some
of their default positions during articulation validation.

For Isaac simulation these joints are more correctly represented as
fixed joints.

The official source URDF is NEVER modified.
A derived "_isaac.urdf" file is written beside it so all relative mesh
paths continue to resolve correctly.
"""

from pathlib import Path
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_URDF = (
    PROJECT_ROOT
    / "external"
    / "unitree_ros"
    / "robots"
    / "g1_with_brainco_hand"
    / "g1_29dof_mode_15_brainco_hand.urdf"
)

OUTPUT_URDF = (
    SOURCE_URDF.parent
    / "g1_29dof_mode_15_brainco_hand_isaac.urdf"
)


# ---------------------------------------------------------------------------
# Joints which are geometrically locked in the official BrainCo model.
#
# Value = fixed revolute position encoded by the original limits.
#
# The four left non-thumb fingertips are locked at +1 rad.
# Everything else below is locked at 0 rad.
# ---------------------------------------------------------------------------

LOCKED_JOINTS = {
    "left_thumb_tip_joint": 0.0,
    "left_index_tip_joint": 1.0,
    "left_middle_tip_joint": 1.0,
    "left_ring_tip_joint": 1.0,
    "left_pinky_tip_joint": 1.0,

    # Note: the official URDF names this one without "_joint".
    "right_thumb_tip": 0.0,
    "right_index_tip_joint": 0.0,
    "right_middle_tip_joint": 0.0,
    "right_ring_tip_joint": 0.0,
    "right_pinky_tip_joint": 0.0,
}


def parse_vec3(text: str) -> list[float]:
    values = [float(value) for value in text.split()]

    if len(values) != 3:
        raise ValueError(
            f"Expected 3 values, received: {text}"
        )

    return values


def format_vec3(values: list[float]) -> str:
    return " ".join(
        f"{value:.12g}"
        for value in values
    )


def main():

    if not SOURCE_URDF.exists():
        raise FileNotFoundError(
            f"Source URDF not found:\n{SOURCE_URDF}"
        )

    print()
    print("============================================================")
    print(" RAVEN - PREPARE G1 MODE 15 + BRAINCO FOR ISAAC")
    print("============================================================")
    print()
    print(f"Source : {SOURCE_URDF}")
    print(f"Output : {OUTPUT_URDF}")
    print()

    tree = ET.parse(SOURCE_URDF)
    root = tree.getroot()

    found = set()

    for joint in root.findall("joint"):

        joint_name = joint.attrib.get("name")

        if joint_name not in LOCKED_JOINTS:
            continue

        found.add(joint_name)

        joint_type = joint.attrib.get("type")

        if joint_type != "revolute":
            raise RuntimeError(
                f"{joint_name}: expected revolute joint, "
                f"found '{joint_type}'."
            )

        limit = joint.find("limit")

        if limit is None:
            raise RuntimeError(
                f"{joint_name}: missing <limit>."
            )

        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])

        expected_position = LOCKED_JOINTS[joint_name]

        if abs(lower - upper) > 1.0e-12:
            raise RuntimeError(
                f"{joint_name}: expected zero-width limits, "
                f"found [{lower}, {upper}]."
            )

        if abs(lower - expected_position) > 1.0e-9:
            raise RuntimeError(
                f"{joint_name}: expected locked position "
                f"{expected_position}, found {lower}."
            )

        origin = joint.find("origin")

        if origin is None:
            raise RuntimeError(
                f"{joint_name}: missing <origin>."
            )

        # ---------------------------------------------------------------
        # A revolute joint transform is:
        #
        #     T_origin * R_axis(q)
        #
        # When converting a non-zero locked joint to a fixed joint we
        # must bake q into the fixed origin rotation.
        #
        # The only non-zero locked joints in this URDF rotate around Y,
        # and their existing origin RPY contains only a Y rotation.
        # Therefore the exact equivalent fixed transform is:
        #
        #     pitch_fixed = pitch_origin + q
        # ---------------------------------------------------------------

        if abs(expected_position) > 1.0e-12:

            axis = joint.find("axis")

            if axis is None:
                raise RuntimeError(
                    f"{joint_name}: missing <axis>."
                )

            axis_xyz = parse_vec3(
                axis.attrib["xyz"]
            )

            if (
                abs(axis_xyz[0]) > 1.0e-12
                or abs(axis_xyz[1] - 1.0) > 1.0e-12
                or abs(axis_xyz[2]) > 1.0e-12
            ):
                raise RuntimeError(
                    f"{joint_name}: expected Y axis, "
                    f"found {axis_xyz}."
                )

            rpy = parse_vec3(
                origin.attrib.get(
                    "rpy",
                    "0 0 0",
                )
            )

            if (
                abs(rpy[0]) > 1.0e-12
                or abs(rpy[2]) > 1.0e-12
            ):
                raise RuntimeError(
                    f"{joint_name}: non-zero roll/yaw "
                    f"requires general rotation composition. "
                    f"RPY={rpy}"
                )

            rpy[1] += expected_position

            origin.attrib["rpy"] = format_vec3(
                rpy
            )

        # ---------------------------------------------------------------
        # Convert the zero-range revolute joint into an actual fixed joint.
        # ---------------------------------------------------------------

        joint.attrib["type"] = "fixed"

        for child_name in (
            "axis",
            "limit",
            "dynamics",
            "mimic",
            "safety_controller",
            "calibration",
        ):
            child = joint.find(child_name)

            if child is not None:
                joint.remove(child)

        print(
            f"[FIXED] {joint_name:30s} "
            f"locked_position={expected_position:+.3f} rad "
            f"origin_rpy={origin.attrib.get('rpy', '0 0 0')}"
        )

    missing = set(LOCKED_JOINTS) - found

    if missing:
        raise RuntimeError(
            "Did not find expected locked joints:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    # -------------------------------------------------------------------
    # Safety check: there should be no remaining revolute joints with
    # zero-width limits.
    # -------------------------------------------------------------------

    unexpected_zero_range = []

    for joint in root.findall("joint"):

        if joint.attrib.get("type") != "revolute":
            continue

        limit = joint.find("limit")

        if limit is None:
            continue

        lower = float(limit.attrib["lower"])
        upper = float(limit.attrib["upper"])

        if abs(lower - upper) <= 1.0e-12:
            unexpected_zero_range.append(
                joint.attrib.get("name", "<unnamed>")
            )

    if unexpected_zero_range:
        raise RuntimeError(
            "Unexpected zero-range revolute joints remain:\n"
            + "\n".join(
                unexpected_zero_range
            )
        )

    ET.indent(
        tree,
        space="  ",
    )

    tree.write(
        OUTPUT_URDF,
        encoding="utf-8",
        xml_declaration=True,
    )

    print()
    print(
        f"[RAVEN] Converted {len(found)} locked "
        "fingertip joints to fixed joints."
    )

    print(
        "[RAVEN] Official source URDF was NOT modified."
    )

    print(
        f"[RAVEN] Isaac URDF written to:\n"
        f"        {OUTPUT_URDF}"
    )

    print()


if __name__ == "__main__":
    main()