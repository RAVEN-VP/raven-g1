import mujoco
import numpy as np

from ctqi_stage0 import apply_open_hand_pose, controlled_joints

from ctqi_experiment_b import (
    MODEL_B,
    build_experiment_model,
    calibrate_object_to_hand,
)

build_experiment_model(arm_only=False)

model = mujoco.MjModel.from_xml_path(str(MODEL_B))
data = mujoco.MjData(model)

data.qpos[:] = model.qpos0
data.qvel[:] = 0.0

apply_open_hand_pose(model, data)
mujoco.mj_forward(model, data)

entries = controlled_joints(model)
arms = [e for e in entries if not e["hand"]]

right_id = model.body("right_hand_base_link").id
cube_id = model.body("teleop_cube").id

cube_pos = calibrate_object_to_hand(
    model, data, arms, right_id, cube_id
)

mujoco.mj_forward(model, data)

hand_bodies = {right_id}

for bid in range(1, model.nbody):
    if int(model.body_parentid[bid]) in hand_bodies:
        hand_bodies.add(bid)

print("\n========== INITIAL OBJECT/HAND GEOMETRY ==========")
print("Cube centre:", np.round(cube_pos, 4))

print("\n========== INITIAL COLLISIONS ==========")

count = 0

for i in range(data.ncon):
    c = data.contact[i]

    b1 = int(model.geom_bodyid[c.geom1])
    b2 = int(model.geom_bodyid[c.geom2])

    if not (
        (b1 == cube_id and b2 in hand_bodies)
        or
        (b2 == cube_id and b1 in hand_bodies)
    ):
        continue

    force = np.zeros(6)
    mujoco.mj_contactForce(model, data, i, force)

    g1 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1
    )

    g2 = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2
    )

    print(
        f"{g1} <-> {g2} | "
        f"penetration={c.dist*1000:.2f} mm | "
        f"normal_force={force[0]:.3f} N | "
        f"position={np.round(c.pos, 4)}"
    )

    count += 1

print("\nInitial hand/object contact pairs:", count)

print("\n========== RIGHT HAND COLLISION GEOMETRIES ==========")

for gid in range(model.ngeom):

    body = int(model.geom_bodyid[gid])

    if body not in hand_bodies:
        continue

    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, gid
    )

    print(
        name,
        "| body:",
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, body
        ),
        "| position:",
        np.round(data.geom_xpos[gid], 4),
        "| type:",
        int(model.geom_type[gid]),
        "| size:",
        np.round(model.geom_size[gid], 4),
    )

