
"""RAVEN - Interactive Isaac Sim viewport-gizmo grasp calibration.

Select /World/RAVEN_Grasp_Target in Stage.
Move with W, rotate with E. The 7-DOF left arm follows with the waist locked.
G closes/opens fingers; P saves the measured pose.
The on-screen buttons provide an alternative to keyboard shortcuts.

No physics-assisted grasp is attached during calibration.
The validated RAVEN tracking controller is not modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACKER = HERE / "raven_g1_actor_tracking.py"

if not TRACKER.is_file():
    raise SystemExit(f"Cannot find RAVEN tracker: {TRACKER}")

spec = importlib.util.spec_from_file_location(
    "raven_gizmo_baseline", TRACKER
)
if spec is None or spec.loader is None:
    raise SystemExit(f"Cannot import {TRACKER}")

base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

import carb.input
import omni.appwindow
import omni.ui as ui
import omni.usd
import torch
from pxr import Gf, Usd, UsdGeom

ROBOT_FORWARD_SHIFT_M = 0.10
INITIAL_ARM_APPROACH_FRACTION = 0.60
INITIAL_ARM_APPROACH_S = 2.25
INITIAL_SETTLE_S = 0.5
JOINT_MAX_STEP_RAD = 0.012
FINGER_CLOSE_S = 0.8

TARGET_PATH = "/World/RAVEN_Grasp_Target"

OUT = (
    Path.cwd()
    / "raven_training_results"
    / "gizmo_grasp_calibration"
)
OUT.mkdir(parents=True, exist_ok=True)

SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")


def vals(tensor):
    return [
        float(v) if math.isfinite(float(v)) else None
        for v in tensor.detach().cpu().reshape(-1).tolist()
    ]


def world_pose(prim):
    transform = UsdGeom.Xformable(
        prim
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

    translation = transform.ExtractTranslation()
    quaternion = transform.ExtractRotation().GetQuat()
    imag = quaternion.GetImaginary()

    pos = [float(translation[i]) for i in range(3)]
    quat = [float(quaternion.GetReal())] + [
        float(imag[i]) for i in range(3)
    ]

    size = math.sqrt(sum(x * x for x in quat))
    if size < 1e-8:
        raise RuntimeError("Invalid gizmo quaternion")

    return pos, [x / size for x in quat]


def cube(parent_path, name, offset, dimensions, rgb):
    stage = omni.usd.get_context().get_stage()
    prim = UsdGeom.Cube.Define(
        stage, f"{parent_path}/{name}"
    )

    prim.GetSizeAttr().Set(1.0)
    prim.GetDisplayColorAttr().Set([
        Gf.Vec3f(*rgb)
    ])

    xform = UsdGeom.Xformable(prim.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
    xform.AddScaleOp().Set(Gf.Vec3f(*dimensions))


def create_target(position, orientation):
    stage = omni.usd.get_context().get_stage()

    if stage.GetPrimAtPath(TARGET_PATH).IsValid():
        raise RuntimeError(
            f"Target already exists: {TARGET_PATH}"
        )

    marker = UsdGeom.Xform.Define(stage, TARGET_PATH)
    marker.AddTranslateOp().Set(Gf.Vec3d(*position))

    marker.AddOrientOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(
        Gf.Quatd(
            float(orientation[0]),
            Gf.Vec3d(*[
                float(v) for v in orientation[1:]
            ]),
        )
    )

    cube(
        TARGET_PATH, "Axis_X",
        (0.033, 0, 0),
        (0.066, 0.005, 0.005),
        (1.0, 0.18, 0.18),
    )

    cube(
        TARGET_PATH, "Axis_Y",
        (0, 0.033, 0),
        (0.005, 0.066, 0.005),
        (0.18, 1.0, 0.18),
    )

    cube(
        TARGET_PATH, "Axis_Z",
        (0, 0, 0.033),
        (0.005, 0.005, 0.066),
        (0.18, 0.42, 1.0),
    )

    cube(
        TARGET_PATH, "Centre",
        (0, 0, 0),
        (0.014, 0.014, 0.014),
        (1.0, 0.94, 0.12),
    )

    selection = omni.usd.get_context().get_selection()
    selection.set_selected_prim_paths(
        [TARGET_PATH], True
    )

    return marker.GetPrim()


class Controls:
    def __init__(self):
        self.pending_save = False
        self.pending_toggle = False
        self.pending_status = False
        self.quit = False

        self._interface = (
            carb.input.acquire_input_interface()
        )
        self._window = (
            omni.appwindow.get_default_app_window()
        )
        self._keyboard = self._window.get_keyboard()

        if self._keyboard is None:
            raise RuntimeError(
                "Isaac Sim GUI keyboard unavailable"
            )

        self._sub_id = (
            self._interface.subscribe_to_keyboard_events(
                self._keyboard, self._on_key
            )
        )

        self.ui_window = None
        self.status_label = None

        try:
            self.ui_window = ui.Window(
                "RAVEN Grasp Calibration",
                width=370,
                height=208,
            )

            with self.ui_window.frame:
                with ui.VStack(spacing=7):
                    ui.Label(
                        "Select /World/RAVEN_Grasp_Target"
                    )
                    ui.Label(
                        "W = Move    E = Rotate"
                    )
                    ui.Label(
                        "G = Fingers    P = Save"
                    )

                    with ui.HStack(spacing=6):
                        ui.Button(
                            "Open / Close",
                            clicked_fn=self.toggle,
                        )
                        ui.Button(
                            "SAVE",
                            clicked_fn=self.save,
                        )
                        ui.Button(
                            "Print pose",
                            clicked_fn=self.status,
                        )

                    self.status_label = ui.Label(
                        "Waist locked. No fixed grasp joint."
                    )

        except Exception as exc:
            print(
                "[GIZMO] Optional UI unavailable:",
                exc,
            )

    def _on_key(self, event):
        if (
            event.type
            != carb.input.KeyboardEventType.KEY_PRESS
        ):
            return False

        if event.input == carb.input.KeyboardInput.P:
            self.save()
        elif event.input == carb.input.KeyboardInput.G:
            self.toggle()
        elif event.input == carb.input.KeyboardInput.H:
            self.status()
        elif event.input == carb.input.KeyboardInput.Q:
            self.quit = True
        else:
            return False

        return True

    def save(self):
        self.pending_save = True

    def toggle(self):
        self.pending_toggle = True

    def status(self):
        self.pending_status = True

    def show_status(self, value):
        if self.status_label is not None:
            self.status_label.text = value

    def close(self):
        if self._sub_id is not None:
            self._interface.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._sub_id,
            )
            self._sub_id = None

        if self.ui_window is not None:
            self.ui_window.visible = False


def run(sim, scene):
    robot = scene["robot"]
    tripod = scene["tripod"]
    dt = float(sim.get_physics_dt())

    rn = {
        name: i
        for i, name in enumerate(
            robot.data.joint_names
        )
    }

    rb = {
        name: i
        for i, name in enumerate(
            robot.data.body_names
        )
    }

    tb = {
        name: i
        for i, name in enumerate(
            tripod.data.body_names
        )
    }

    tn = {
        name: i
        for i, name in enumerate(
            tripod.data.joint_names
        )
    }

    arm_ids = [
        rn[name] for name in base.ARM_NAMES
    ]

    right_ids = [
        rn[name] for name in base.RIGHT_ARM_NAMES
    ]

    finger_ids = [
        rn[name] for name in base.HAND_CLOSE
    ]

    waist_id = rn["waist_yaw_joint"]
    hand_body = rb["left_base_link"]
    handle_body = tb["control_handle_link"]
    elbow_id = rn["left_elbow_joint"]

    jac_body = (
        hand_body - 1
        if robot.is_fixed_base
        else hand_body
    )

    pan_id = tn["pan_joint"]
    tilt_id = tn["tilt_joint"]

    neutral = robot.data.default_joint_pos.clone()

    neutral[:, arm_ids] = torch.as_tensor(
        base.LEFT_ARM_STAND,
        device=sim.device,
        dtype=neutral.dtype,
    )[None, :]

    neutral[:, right_ids] = torch.as_tensor(
        base.RIGHT_ARM_STAND,
        device=sim.device,
        dtype=neutral.dtype,
    )[None, :]

    neutral[:, finger_ids] = 0.0

    robot.write_joint_state_to_sim(
        neutral,
        torch.zeros_like(neutral),
    )
    robot.set_joint_position_target(neutral)

    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(dt)

    calibrated = torch.as_tensor(
        base.ARM_CALIBRATED,
        device=sim.device,
        dtype=neutral.dtype,
    )[None, :]

    arm_start = neutral[:, arm_ids].clone()

    near = (
        arm_start
        + INITIAL_ARM_APPROACH_FRACTION
        * (calibrated - arm_start)
    )

    commanded = neutral.clone()

    for step in range(
        round(INITIAL_ARM_APPROACH_S / dt)
    ):
        if not base.simulation_app.is_running():
            return

        alpha = base.smoothstep(
            (step + 1)
            * dt
            / INITIAL_ARM_APPROACH_S
        )

        commanded[:, arm_ids] = (
            arm_start
            + alpha * (near - arm_start)
        )

        robot.set_joint_position_target(commanded)

        tripod.set_joint_position_target(
            torch.zeros(
                (1, 2),
                device=sim.device,
            ),
            joint_ids=[pan_id, tilt_id],
        )

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt)

    for _ in range(
        round(INITIAL_SETTLE_S / dt)
    ):
        if not base.simulation_app.is_running():
            return

        robot.set_joint_position_target(commanded)

        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(dt)

    hand_pose = robot.data.body_pose_w[
        0, hand_body
    ]

    marker = create_target(
        vals(hand_pose[:3]),
        vals(hand_pose[3:7]),
    )

    controls = Controls()

    ik = base.DifferentialIKController(
        base.DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.05},
        ),
        num_envs=1,
        device=sim.device,
    )

    ik_command = torch.zeros(
        (1, ik.action_dim),
        device=sim.device,
    )

    limits = robot.data.joint_pos_limits[
        :, arm_ids, :
    ]

    closed_pos = torch.as_tensor(
        list(base.HAND_CLOSE.values()),
        device=sim.device,
        dtype=neutral.dtype,
    )[None, :]

    closed = False
    finger_alpha = 0.0
    save_count = 0

    print(
        "\n===== RAVEN GIZMO GRASP CALIBRATION ====="
    )
    print(
        f"[GIZMO] Target: {TARGET_PATH}"
    )
    print(
        "[GIZMO] Select target in Stage"
    )
    print(
        "[GIZMO] W = Move, E = Rotate"
    )
    print(
        "[GIZMO] G = Fingers, P = Save"
    )
    print(
        "[GIZMO] Waist locked"
    )
    print(
        "=======================================\n"
    )

    def print_measurements():
        hand = robot.data.body_pose_w[
            0, hand_body
        ]

        handle = tripod.data.body_pose_w[
            0, handle_body
        ]

        relative_p, relative_q = (
            base.subtract_frame_transforms(
                handle[:3].unsqueeze(0),
                handle[3:7].unsqueeze(0),
                hand[:3].unsqueeze(0),
                hand[3:7].unsqueeze(0),
            )
        )

        desired_p, desired_q = world_pose(
            marker
        )

        error_m = float(
            torch.linalg.vector_norm(
                hand[:3]
                - torch.as_tensor(
                    desired_p,
                    device=sim.device,
                )
            )
        )

        alignment = abs(
            float(
                torch.dot(
                    hand[3:7],
                    torch.as_tensor(
                        desired_q,
                        device=sim.device,
                    ),
                )
            )
        )

        error_deg = math.degrees(
            2 * math.acos(
                min(
                    1.0,
                    max(0.0, alignment),
                )
            )
        )

        print(
            "[GIZMO] Hand relative to handle:",
            vals(relative_p[0]),
        )

        print(
            "[GIZMO] Elbow:",
            math.degrees(
                float(
                    robot.data.joint_pos[
                        0, elbow_id
                    ]
                )
            ),
            "degrees",
        )

        print(
            "[GIZMO] Target error:",
            round(error_m * 1000, 2),
            "mm,",
            round(error_deg, 2),
            "degrees",
        )

        return (
            relative_p,
            relative_q,
            error_m,
            error_deg,
        )

    def save():
        nonlocal save_count

        if not closed or finger_alpha < 0.98:
            print(
                "[GIZMO] Close fingers with G first."
            )
            controls.show_status(
                "Close fingers, then Save"
            )
            return

        (
            relative_p,
            relative_q,
            error_m,
            error_deg,
        ) = print_measurements()

        max_speed = float(
            robot.data.joint_vel[
                :, arm_ids
            ].abs().max()
        )

        if max_speed > 0.30:
            print(
                "[GIZMO] Arm is still moving."
            )
            return

        hand = robot.data.body_pose_w[
            0, hand_body
        ]

        handle = tripod.data.body_pose_w[
            0, handle_body
        ]

        desired_p, desired_q = world_pose(
            marker
        )

        if error_m > 0.025 or error_deg > 15:
            print(
                "[GIZMO] Warning: arm has not "
                "fully reached target."
            )

        save_count += 1

        payload = {
            "experiment":
                "RAVEN viewport grasp calibration",

            "version": 1,

            "status":
                "VISUAL_CALIBRATION_ONLY",

            "session": SESSION,

            "capture_index": save_count,

            "operator_stance_forward_shift_m":
                ROBOT_FORWARD_SHIFT_M,

            "robot_root_pose_w_xyz_quat_wxyz":
                vals(robot.data.root_pose_w[0]),

            "g1_joint_names":
                list(robot.data.joint_names),

            "g1_joint_positions_rad":
                vals(robot.data.joint_pos[0]),

            "left_arm_joint_names":
                list(base.ARM_NAMES),

            "left_arm_joint_positions_rad":
                vals(
                    robot.data.joint_pos[
                        0, arm_ids
                    ]
                ),

            "waist_yaw_rad":
                float(
                    robot.data.joint_pos[
                        0, waist_id
                    ]
                ),

            "finger_joint_names":
                list(base.HAND_CLOSE.keys()),

            "finger_joint_positions_rad":
                vals(
                    robot.data.joint_pos[
                        0, finger_ids
                    ]
                ),

            "fingers_closed": closed,

            "robot_hand_link":
                "left_base_link",

            "tripod_handle_link":
                "control_handle_link",

            "hand_world_pose_xyz_quat_wxyz":
                vals(hand),

            "handle_world_pose_xyz_quat_wxyz":
                vals(handle),

            "hand_pose_relative_to_handle_pos_m":
                vals(relative_p[0]),

            "hand_pose_relative_to_handle_quat_wxyz":
                vals(relative_q[0]),

            "gizmo_target_world_pos_m":
                desired_p,

            "gizmo_target_world_quat_wxyz":
                desired_q,

            "gizmo_to_hand_error_m":
                error_m,

            "gizmo_to_hand_orientation_error_deg":
                error_deg,

            "maximum_arm_joint_velocity_rad_s":
                max_speed,

            "tripod_joint_names":
                list(tripod.data.joint_names),

            "tripod_joint_positions_rad":
                vals(
                    tripod.data.joint_pos[0]
                ),

            "source_tracker_sha256":
                hashlib.sha256(
                    TRACKER.read_bytes()
                ).hexdigest(),

            "notes":
                "Measured hand pose with fingers closed. "
                "No assisted physics grasp attached.",
        }

        fname = OUT / (
            f"raven_gizmo_grasp_"
            f"{SESSION}_{save_count:02d}.json"
        )

        latest = OUT / (
            "raven_gizmo_grasp_latest.json"
        )

        fname.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        latest.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        print(
            f"\n[GIZMO] SAVED: {fname}"
        )

        controls.show_status(
            f"Saved {fname.name}"
        )

    try:
        while (
            base.simulation_app.is_running()
            and not controls.quit
        ):
            if controls.pending_toggle:
                controls.pending_toggle = False
                closed = not closed

                print(
                    "[GIZMO] Fingers",
                    "CLOSING" if closed
                    else "OPENING",
                )

            if controls.pending_status:
                controls.pending_status = False
                print_measurements()

            if controls.pending_save:
                controls.pending_save = False
                save()

            desired_p, desired_q = world_pose(
                marker
            )

            if not (
                all(
                    math.isfinite(v)
                    for v in desired_p
                )
                and all(
                    math.isfinite(v)
                    for v in desired_q
                )
            ):
                raise RuntimeError(
                    "Invalid gizmo transform"
                )

            t_p = torch.tensor(
                [desired_p],
                device=sim.device,
                dtype=neutral.dtype,
            )

            t_q = torch.tensor(
                [desired_q],
                device=sim.device,
                dtype=neutral.dtype,
            )

            root = robot.data.root_pose_w
            hand = robot.data.body_pose_w[
                :, hand_body
            ]

            goal_p, goal_q = (
                base.subtract_frame_transforms(
                    root[:, :3],
                    root[:, 3:7],
                    t_p,
                    t_q,
                )
            )

            cur_p, cur_q = (
                base.subtract_frame_transforms(
                    root[:, :3],
                    root[:, 3:7],
                    hand[:, :3],
                    hand[:, 3:7],
                )
            )

            ik_command[:, :3] = goal_p
            ik_command[:, 3:7] = goal_q

            ik.set_command(ik_command)

            jacobian = (
                robot.root_physx_view.get_jacobians()[
                    :, jac_body, :, arm_ids
                ]
            )

            actual = robot.data.joint_pos[
                :, arm_ids
            ]

            proposed = ik.compute(
                cur_p,
                cur_q,
                jacobian,
                actual,
            )

            commanded[:, arm_ids] = (
                torch.maximum(
                    torch.minimum(
                        actual
                        + (
                            proposed - actual
                        ).clamp(
                            -JOINT_MAX_STEP_RAD,
                            JOINT_MAX_STEP_RAD,
                        ),
                        limits[..., 1],
                    ),
                    limits[..., 0],
                )
            )

            commanded[:, waist_id] = (
                neutral[:, waist_id]
            )

            if closed:
                finger_alpha = min(
                    1.0,
                    finger_alpha
                    + dt / FINGER_CLOSE_S,
                )
            else:
                finger_alpha = max(
                    0.0,
                    finger_alpha
                    - dt / FINGER_CLOSE_S,
                )

            commanded[:, finger_ids] = (
                finger_alpha * closed_pos
            )

            robot.set_joint_position_target(
                commanded
            )

            tripod.set_joint_position_target(
                torch.zeros(
                    (1, 2),
                    device=sim.device,
                ),
                joint_ids=[
                    pan_id,
                    tilt_id,
                ],
            )

            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(dt)

    finally:
        controls.close()


def main():
    sim = base.sim_utils.SimulationContext(
        base.sim_utils.SimulationCfg(
            dt=base.DT,
            device="cpu",
            use_fabric=False,
        )
    )

    sim.set_camera_view(
        eye=(1.6, 1.2, 1.48),
        target=(0.43, 0.12, 0.92),
    )

    cfg = base.SceneCfg(
        num_envs=1,
        env_spacing=2.0,
    )

    p = cfg.robot.init_state.pos

    cfg.robot.init_state.pos = (
        p[0] + ROBOT_FORWARD_SHIFT_M,
        p[1],
        p[2],
    )

    print(
        "[GIZMO] G1 spawned 10 cm closer"
    )

    scene = base.InteractiveScene(cfg)
    sim.reset()

    run(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        base.simulation_app.close()
