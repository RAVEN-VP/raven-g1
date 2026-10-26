"""RAVEN: viewport grasp calibration v2 (Isaac Sim 5.1 / Isaac Lab 2.3).

Select /World/RAVEN_Grasp_Target in Stage. W to move, E to rotate.
The G1 left arm follows through 7-joint WORLD-FRAME IK; waist is locked.
G closes/opens fingers; P saves measured grasp. Buttons are alternatives.
The target is ONE selectable cube with no collision or rigid body.
No fixed grasp joint; validated tracker remains untouched.
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
    raise SystemExit(f"Original tracker not found: {TRACKER}")

spec = importlib.util.spec_from_file_location("raven_gizmo_v2_base", TRACKER)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)  # Creates the Isaac Sim application.

import torch
import carb.input
import omni.appwindow
import omni.ui as ui
import omni.usd
from pxr import Gf, Usd, UsdGeom

TARGET = "/World/RAVEN_Grasp_Target"
OUTPUT = Path.cwd() / "raven_training_results" / "gizmo_grasp_calibration"
OUTPUT.mkdir(parents=True, exist_ok=True)
SESSION = datetime.now().strftime("%Y%m%d_%H%M%S")
FORWARD_SHIFT_M = 0.10
START_REACH_FRACTION = 0.78
MAX_JOINT_STEP = 0.008
FINGER_CLOSE_S = 0.8


def numbers(t):
    return [float(v) for v in t.detach().cpu().reshape(-1).tolist()]


def read_target(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())
    p = m.ExtractTranslation()
    q = m.ExtractRotation().GetQuat()
    qi = q.GetImaginary()
    return ([float(p[i]) for i in range(3)],
            [float(q.GetReal())] + [float(qi[i]) for i in range(3)])


def create_target(p, q):
    stage = omni.usd.get_context().get_stage()
    # A SINGLE selectable prim: no children that can be moved accidentally.
    if stage.GetPrimAtPath(TARGET).IsValid():
        stage.RemovePrim(TARGET)
    cube = UsdGeom.Cube.Define(stage, TARGET)
    cube.GetSizeAttr().Set(0.045)
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.9, 0.1)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*p))
    cube.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(q[0], Gf.Vec3d(*q[1:])))
    omni.usd.get_context().get_selection().set_selected_prim_paths(
        [TARGET], True)
    return cube.GetPrim()


class Controls:
    def __init__(self):
        self.save_requested = False
        self.fingers_requested = False
        self.quit_requested = False
        self.status_requested = False
        self.window = ui.Window("RAVEN Grasp Calibration v2", width=360, height=220)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Select /World/RAVEN_Grasp_Target")
                ui.Label("W: move   E: rotate   G: fingers   P: save")
                with ui.HStack():
                    ui.Button("Open/Close (G)", clicked_fn=self.toggle)
                    ui.Button("SAVE (P)", clicked_fn=self.save)
                    ui.Button("Measure (H)", clicked_fn=self.status)
                ui.Button("TEST: shift cube +2 cm X", clicked_fn=self.test_nudge)
                self.label = ui.Label("Move the yellow cube; waist locked")
        self.interface = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self.sub = self.interface.subscribe_to_keyboard_events(
            self.keyboard, self.key)