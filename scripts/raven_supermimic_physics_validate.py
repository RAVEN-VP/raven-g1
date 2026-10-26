
"""RAVEN Policy V1.4: paired, in-process Isaac Sim receding-horizon test.

Run from the RAVEN repository root:
    python scripts/raven_supermimic_physics_validate.py --case 0

IMPORTANT: This is a single-scene, state-restoration A/B test, not a perfect
fork of PhysX's internal solver/contact cache. A repeat HELD run checks whether
restoration was sufficiently reproducible. If that check fails the result is
INVALID and no policy improvement is claimed. No G1 production scripts edited.

Runs, all from the same saved physics state / same camera reference:
  1. HELD: 8 physics steps of the current joint targets.
  2. HELD_REPEAT: restored snapshot, same 8 commands (replay null test).
  3. PLANNED: restored snapshot, optimise a fresh bounded V2 action every physics step.

Planning uses live measured robot state at each of 8 steps (receding-horizon MPC).
No RGB-D image evaluation during this short frozen-actor 66.7 ms window.
The tracking controller obtains the reference before the branch starts.
A simulated fixed joint supports the assisted grasp; unaided finger-only grasp
is outside the scope of this experiment. There is no trained RL policy yet.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACKER = HERE / "raven_g1_actor_tracking.py"
MODEL_MODULE = HERE / "raven_supermimic_world_model_v2.py"
POLICY_MODULE = HERE / "raven_supermimic_policy_v1.py"
DEFAULT_MODEL = ROOT / "raven_training_results/world_model_v2/world_model_v2.pt"
DEFAULT_METRICS = ROOT / "raven_training_results/policy_v1/policy_v1_frame_corrected_metrics.json"
DEFAULT_OUT = ROOT / "raven_training_results/policy_physics_v2"
EXPECTED_TRACKER_SHA = "2b7da3c47a8614b690bb20f2f461d0bb93a90a90b9f0c95a3d6c6b4648ecd038"
HORIZON = 8


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=int, default=0)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--iterations", type=int, default=60)
    return p


def load_case(args):
    if not args.metrics.is_file():
        raise SystemExit(f"Missing frame-corrected policy metrics: {args.metrics}")
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    if "coordinate-aligned" not in payload.get("experiment", ""):
        raise SystemExit("Please provide Policy V1.1 frame-corrected metrics")
    cases = payload["cases_detail"]
    if not 0 <= args.case < len(cases):
        raise SystemExit(f"Case must be 0..{len(cases)-1}")
    return cases[args.case]


def create_report(held, repeat, planned, tolerances, output):
    """Pure comparison function; can be exercised without Isaac Sim."""
    import numpy as np

    def max_abs(x, y):
        return float(np.max(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))))

    a, b, c = (branch["start"] for branch in (held, repeat, planned))
    start_diffs = {}
    for label, other in (("held_repeat", b), ("planned", c)):
        start_diffs[label] = {
            "q_rad": max_abs(a["robot_q_rad"], other["robot_q_rad"]),
            "dq_rad_s": max_abs(a["robot_dq_rad_s"], other["robot_dq_rad_s"]),
            "tripod_q_rad": max_abs(a["tripod_q_rad"], other["tripod_q_rad"]),
            "tripod_dq_rad_s": max_abs(a["tripod_dq_rad_s"], other["tripod_dq_rad_s"]),
            "robot_root": max_abs(a["robot_root_state_w"], other["robot_root_state_w"]),
            "tripod_root": max_abs(a["tripod_root_state_w"], other["tripod_root_state_w"]),
            "hand_pose": max_abs(a["hand_pose_w"], other["hand_pose_w"]),
            "handle_pose": max_abs(a["handle_pose_w"], other["handle_pose_w"]),
            "reference_deg": abs(a["reference_relative_deg"] - other["reference_relative_deg"]),
        }

    matched = all(
        diffs["q_rad"] <= tolerances["joint_rad"]
        and diffs["dq_rad_s"] <= tolerances["velocity_rad_s"]
        and diffs["tripod_q_rad"] <= tolerances["joint_rad"]
        and diffs["tripod_dq_rad_s"] <= tolerances["velocity_rad_s"]
        and diffs["robot_root"] <= tolerances["root_state"]
        and diffs["tripod_root"] <= tolerances["root_state"]
        and diffs["hand_pose"] <= tolerances["body_pose"]
        and diffs["handle_pose"] <= tolerances["body_pose"]
        and diffs["reference_deg"] <= 1e-6
        for diffs in start_diffs.values()
    )

    replay_pan = abs(held["end_pan_abs_deg"] - repeat["end_pan_abs_deg"])
    replay_tilt = abs(held["end_tilt_abs_deg"] - repeat["end_tilt_abs_deg"])
    replay_q = max_abs(held["end_robot_q_rad"], repeat["end_robot_q_rad"])

    replay_ok = (
        replay_pan <= tolerances["null_pan_deg"]
        and replay_tilt <= tolerances["null_tilt_deg"]
        and replay_q <= tolerances["null_q_rad"]
    )

    safety = all(x["safety_status"] == "OK" for x in (held, repeat, planned))
    unclipped = planned["unfeasible_action_steps"] == 0
    complete = all(x["steps_applied"] == HORIZON for x in (held, repeat, planned))

    valid = matched and replay_ok and safety and unclipped and complete

    delta = (
        held["actual_terminal_pan_error_deg"]
        - planned["actual_terminal_pan_error_deg"]
    )

    report = {
        "experiment": "RAVEN in-process paired counterfactual physics test",
        "valid_comparison": bool(valid),
        "start_states_matched": bool(matched),
        "physics_restore_replay_reproducible": bool(replay_ok),
        "all_conditions_safe": bool(safety),
        "planned_commands_feasible_without_runtime_clipping": bool(unclipped),
        "complete_eight_steps": bool(complete),
        "start_differences": start_diffs,
        "replay_differences": {
            "end_pan_deg": replay_pan,
            "end_tilt_deg": replay_tilt,
            "max_robot_joint_rad": replay_q,
        },
        "tolerances": tolerances,
        "held_measured_error_deg": held["actual_terminal_pan_error_deg"],
        "planned_measured_error_deg": planned["actual_terminal_pan_error_deg"],
        "candidate_error_change_deg_NOT_CLAIMABLE_IF_INVALID": delta,
        "measured_improvement_percent": (
            100 * delta / held["actual_terminal_pan_error_deg"]
            if valid and held["actual_terminal_pan_error_deg"] > 1e-9 else None
        ),
        "held": held,
        "held_repeat": repeat,
        "planned": planned,
        "limitations": [
            "Articulation state is restored, but internal PhysX solver/contact warm starts may not restore exactly.",
            "Held-repeat null experiment is mandatory and must agree before policy comparisons are valid.",
            "Tracking reference and actor position are frozen for the 66.7 ms branches.",
            "Only one controlled initial state; more cases and replicates are required.",
            "Assisted fixed-joint grasp; physical finger-only grasp is not evaluated.",
            "PLANNED is receding-horizon measured-state MPC, not replay of one eight-step open-loop plan.",
            "This is counterfactual action optimisation, not a deployed or trained RL policy.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def worker_main(args, case):
    if hashlib.sha256(TRACKER.read_bytes()).hexdigest() != EXPECTED_TRACKER_SHA:
        raise SystemExit(
            "Validated tracker source changed; refusing instrumentation. Baseline unchanged."
        )

    out = args.out.resolve() / f"case_{args.case:02d}"
    out.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        str(TRACKER),
        "--actor-amplitude-m", "0.25",
        "--actor-period-s", "5.0",
        "--tracking-duration-s", "35.0",
        "--output-dir", str(out / "tracking"),
    ]

    spec = importlib.util.spec_from_file_location(
        "raven_validated_physics", TRACKER
    )

    if spec is None or spec.loader is None:
        raise SystemExit("Unable to import validated controller")

    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)

    import torch

    from raven_supermimic_world_model_v2 import ResidualDynamics, JOINTS
    from raven_supermimic_policy_v1 import objective, MAX_DELTA_RAD

    ckpt = torch.load(
        args.model,
        map_location="cpu",
        weights_only=True,
    )

    if (
        ckpt.get("model_class") != "ResidualDynamics"
        or tuple(ckpt["joint_names"]) != tuple(JOINTS)
    ):
        raise SystemExit("Expected V2 residual world-model checkpoint")

    if int(ckpt["horizon"]) != HORIZON:
        raise SystemExit("Incompatible V2 prediction horizon")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResidualDynamics(*ckpt["normalization"]).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    class PairedTest:
        def __init__(self):
            self.completed = False
            self.pan_zero = None
            self.reference_deg = None
            self.grip_pos = None
            self.grip_quat = None
            self.held_targets = None
            self.planned_targets = None
            self.predicted = None
            self.output = out

        def extract(self, robot, tripod, indices):
            rb = robot.data.body_names
            tb = tripod.data.body_names

            hand = robot.data.body_pose_w[
                0, rb.index("left_base_link")
            ]

            handle = tripod.data.body_pose_w[
                0, tb.index("control_handle_link")
            ]

            return {
                "robot_q_rad": robot.data.joint_pos[0].clone(),
                "robot_dq_rad_s": robot.data.joint_vel[0].clone(),
                "tripod_q_rad": tripod.data.joint_pos[0].clone(),
                "tripod_dq_rad_s": tripod.data.joint_vel[0].clone(),
                "robot_root_state_w": robot.data.root_state_w[0].clone(),
                "tripod_root_state_w": tripod.data.root_state_w[0].clone(),
                "hand_pose_w": hand.clone(),
                "handle_pose_w": handle.clone(),
            }

        def to_json(self, state):
            return {
                k: [
                    float(v)
                    for v in tensor.detach().cpu().reshape(-1).tolist()
                ]
                for k, tensor in state.items()
            }

        def restore(
            self,
            robot,
            tripod,
            scene,
            sim,
            state,
            full_command,
            tilt_id,
        ):
            # Restore all available articulation states.
            # Robot and tripod roots are included in the snapshot.
            for asset, prefix in (
                (robot, "robot"),
                (tripod, "tripod"),
            ):
                root = state[
                    f"{prefix}_root_state_w"
                ].unsqueeze(0)

                asset.write_root_pose_to_sim(
                    root[:, :7].clone()
                )

                asset.write_root_velocity_to_sim(
                    root[:, 7:13].clone()
                )

                asset.write_joint_state_to_sim(
                    state[f"{prefix}_q_rad"].unsqueeze(0).clone(),
                    state[f"{prefix}_dq_rad_s"].unsqueeze(0).clone(),
                )

            robot.set_joint_position_target(full_command)

            tripod.set_joint_position_target(
                base.torch_column(0.0, sim.device),
                joint_ids=[tilt_id],
            )

            scene.write_data_to_sim()

            # Refresh Isaac Lab data caches without advancing time.
            scene.update(0.0)

        def error_from_state(
            self,
            state,
            hand_p,
            hand_q,
            robot,
            tripod,
        ):
            rb = robot.data.body_names.index(
                "left_base_link"
            )

            tb = tripod.data.body_names.index(
                "control_handle_link"
            )

            handle = tripod.data.body_pose_w[:, tb]
            hand = robot.data.body_pose_w[:, rb]

            target, _ = base.combine_frame_transforms(
                handle[:, :3],
                handle[:, 3:7],
                hand_p,
                hand_q,
            )

            return float(
                torch.linalg.vector_norm(
                    hand[:, :3] - target,
                    dim=-1,
                )[0]
            )

        def plan(
            self,
            state,
            held,
            robot,
            tripod,
            dt,
            ref_abs,
            ids,
        ):
            q = robot.data.joint_pos[:, ids]
            dq = robot.data.joint_vel[:, ids]

            tripod_joint_names = list(
                tripod.data.joint_names
            )

            pan_id = tripod_joint_names.index(
                "pan_joint"
            )

            tilt_id = tripod_joint_names.index(
                "tilt_joint"
            )

            x = torch.cat(
                (
                    q,
                    dq,
                    tripod.data.joint_pos[:, [pan_id, tilt_id]],
                    tripod.data.joint_vel[:, [pan_id, tilt_id]],
                ),
                dim=-1,
            ).detach().to(device)

            center = held[:, ids].detach().to(device)

            lo_hi = robot.data.joint_pos_limits[
                :, ids
            ].to(device)

            lower = lo_hi[..., 0]
            upper = lo_hi[..., 1]

            delta = torch.tensor(
                MAX_DELTA_RAD,
                dtype=x.dtype,
                device=device,
            )

            max_step = torch.full_like(
                delta, base.MAX_ARM_STEP
            )

            max_step[0] = base.MAX_WAIST_STEP

            ref = torch.tensor(
                [ref_abs],
                dtype=x.dtype,
                device=device,
            )

            raw = torch.nn.Parameter(
                torch.zeros(
                    (1, HORIZON, len(ids)),
                    device=device,
                )
            )

            optimiser = torch.optim.Adam(
                [raw],
                lr=0.08,
            )

            fallback = (
                center[:, None, :]
                .expand(-1, HORIZON, -1)
                .clone()
            )

            def project_and_rollout(param):
                s = x
                actions = []
                states = []

                for index in range(HORIZON):
                    # V1.4 correction:
                    #
                    # First step: constrain the joint target against
                    # the ACTUAL measured joint positions.
                    #
                    # Future steps: constrain setpoint changes
                    # relative to the previously planned setpoint.
                    #
                    # Using predicted joint positions for future
                    # constraints can create impossible intervals
                    # when the world model leaves its training
                    # distribution.
                    #
                    # Only the first planned action is executed.
                    # It is checked again against measured physics
                    # inside trial() before sim.step().

                    q_for_command_bounds = (
                        x[:, :8]
                        if index == 0
                        else actions[-1]
                    )

                    min_allowed = torch.maximum(
                        lower,
                        torch.maximum(
                            center - delta,
                            q_for_command_bounds - max_step,
                        ),
                    )

                    max_allowed = torch.minimum(
                        upper,
                        torch.minimum(
                            center + delta,
                            q_for_command_bounds + max_step,
                        ),
                    )

                    if bool(
                        torch.any(
                            min_allowed > max_allowed + 1e-7
                        )
                    ):
                        invalid = (
                            min_allowed > max_allowed + 1e-7
                        )[0]

                        problem_joints = [
                            name
                            for name, bad in zip(
                                JOINTS,
                                invalid.tolist(),
                            )
                            if bad
                        ]

                        raise RuntimeError(
                            f"No feasible "
                            f"{'measured' if index == 0 else 'planned'} "
                            f"command interval at horizon step "
                            f"{index + 1}: {problem_joints}"
                        )

                    suggested = (
                        center
                        + delta * torch.tanh(
                            param[:, index]
                        )
                    )

                    selected = torch.minimum(
                        torch.maximum(
                            suggested,
                            min_allowed,
                        ),
                        max_allowed,
                    )

                    s = model.step(s, selected)

                    actions.append(selected)
                    states.append(s)

                return (
                    torch.stack(actions, dim=1),
                    torch.stack(states, dim=1),
                )

            with torch.no_grad():
                unmodified_forecast = model.rollout(
                    x, fallback
                )

                best_cost = float(
                    objective(
                        unmodified_forecast,
                        fallback,
                        center,
                        ref,
                        x,
                        delta,
                    )[0]
                )

            best = fallback.clone()

            for _ in range(args.iterations):
                sequence, forecast = (
                    project_and_rollout(raw)
                )

                cost = objective(
                    forecast,
                    sequence,
                    center,
                    ref,
                    x,
                    delta,
                ).mean()

                if not torch.isfinite(cost):
                    raise RuntimeError(
                        "Nonfinite planning objective"
                    )

                with torch.no_grad():
                    if float(cost) < best_cost:
                        best_cost = float(cost)
                        best = sequence.detach().clone()

                optimiser.zero_grad(
                    set_to_none=True
                )

                cost.backward()

                torch.nn.utils.clip_grad_norm_(
                    [raw], 1.0
                )

                optimiser.step()

            with torch.no_grad():
                final_actions, final_states = (
                    project_and_rollout(raw)
                )

                final_cost = float(
                    objective(
                        final_states,
                        final_actions,
                        center,
                        ref,
                        x,
                        delta,
                    )[0]
                )

                if final_cost < best_cost:
                    best = final_actions.detach().clone()

                model_planned = model.rollout(
                    x, best
                )

                estimates = {
                    "held_predicted_error_deg": (
                        abs(
                            float(
                                unmodified_forecast[
                                    0, -1, 16
                                ]
                            ) - ref_abs
                        )
                        * 180.0 / math.pi
                    ),
                    "planned_predicted_error_deg": (
                        abs(
                            float(
                                model_planned[
                                    0, -1, 16
                                ]
                            ) - ref_abs
                        )
                        * 180.0 / math.pi
                    ),
                }

            return (
                fallback[0].cpu(),
                best[0].detach().cpu(),
                estimates,
            )

        def trial(
            self,
            mode,
            actions,
            start,
            robot,
            tripod,
            sim,
            scene,
            dt,
            ids,
            tilt_id,
        ):
            pan_id = list(
                tripod.data.joint_names
            ).index("pan_joint")

            actual_initial = self.extract(
                robot,
                tripod,
                ids,
            )

            start_json = self.to_json(
                actual_initial
            )

            start_json[
                "reference_relative_deg"
            ] = self.reference_deg

            grip_max = self.error_from_state(
                actual_initial,
                self.grip_pos,
                self.grip_quat,
                robot,
                tripod,
            )

            max_velocity = 0.0
            max_excess = 0.0
            peak_tilt = 0.0
            abort_reason = "OK"
            infeasible = 0
            feedback_replans = 0
            applied = []

            start_tilt = float(
                tripod.data.joint_pos[
                    0, tilt_id
                ]
            )

            for i in range(HORIZON):
                # Receding-horizon MPC:
                # Replan from current measured PhysX state.
                # Apply the first action of the new rollout.

                if mode == "planned":
                    try:
                        live = self.extract(
                            robot,
                            tripod,
                            ids,
                        )

                        (
                            _baseline,
                            replanned,
                            _forecast,
                        ) = self.plan(
                            live,
                            self.full_command,
                            robot,
                            tripod,
                            dt,
                            math.radians(
                                self.reference_deg
                                + self.pan_zero
                            ),
                            ids,
                        )

                        proposed = replanned[0].to(
                            sim.device
                        )

                        feedback_replans += 1

                    except (ValueError, RuntimeError) as exc:
                        infeasible += 1
                        abort_reason = (
                            "MPC_REPLAN_INFEASIBLE"
                        )

                        print(
                            f"[RAVEN MPC] step "
                            f"{i + 1} rejected: {exc}"
                        )
                        break
                else:
                    proposed = actions[i].to(
                        sim.device
                    )

                # Verify the proposed command against the
                # ACTUAL measured robot state, without clipping.

                q = robot.data.joint_pos[
                    0, ids
                ]

                physical_limits = (
                    robot.data.joint_pos_limits[
                        0, ids
                    ]
                )

                max_step = torch.full_like(
                    q, base.MAX_ARM_STEP
                )

                max_step[0] = (
                    base.MAX_WAIST_STEP
                )

                viable = (
                    (
                        proposed
                        >= physical_limits[:, 0] - 1e-6
                    )
                    & (
                        proposed
                        <= physical_limits[:, 1] + 1e-6
                    )
                    & (
                        (proposed - q).abs()
                        <= max_step + 1e-6
                    )
                )

                if not bool(viable.all()):
                    infeasible += 1

                    abort_reason = (
                        "INFEASIBLE_COMMAND_NO_CLIPPING"
                    )

                    bad = [
                        (
                            joint_name,
                            round(
                                float(
                                    (proposed[j] - q[j]).abs()
                                ),
                                6,
                            ),
                            round(
                                float(max_step[j]),
                                6,
                            ),
                        )
                        for j, joint_name in enumerate(JOINTS)
                        if not bool(viable[j])
                    ]

                    print(
                        f"[RAVEN MPC] step {i + 1} "
                        f"infeasible "
                        f"(joint, target gap rad, max rad): "
                        f"{bad}"
                    )
                    break

                command = self.full_command.clone()

                command[:, ids] = (
                    proposed.unsqueeze(0)
                )

                robot.set_joint_position_target(
                    command
                )

                tripod.set_joint_position_target(
                    base.torch_column(
                        0.0, sim.device
                    ),
                    joint_ids=[tilt_id],
                )

                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(dt)

                applied.append(
                    [
                        float(v)
                        for v in proposed.detach().cpu().tolist()
                    ]
                )

                grip_max = max(
                    grip_max,
                    self.error_from_state(
                        actual_initial,
                        self.grip_pos,
                        self.grip_quat,
                        robot,
                        tripod,
                    ),
                )

                max_velocity = max(
                    max_velocity,
                    float(
                        robot.data.joint_vel[
                            0, ids
                        ].abs().max()
                    ),
                )

                tilt_change = abs(
                    math.degrees(
                        float(
                            tripod.data.joint_pos[
                                0, tilt_id
                            ]
                        )
                        - start_tilt
                    )
                )

                peak_tilt = max(
                    peak_tilt,
                    tilt_change,
                )

                limits = (
                    robot.data.joint_pos_limits[
                        0, ids
                    ]
                )

                measured_q = (
                    robot.data.joint_pos[
                        0, ids
                    ]
                )

                excess = torch.maximum(
                    limits[:, 0] - measured_q,
                    measured_q - limits[:, 1],
                ).clamp(min=0)

                max_excess = max(
                    max_excess,
                    float(excess.max()),
                )

                if max_excess > 0.01:
                    abort_reason = "JOINT_LIMIT"

                elif (
                    grip_max
                    > base.MAX_HAND_POSITION_ERROR_M
                ):
                    abort_reason = "GRASP_ERROR"

                elif (
                    abs(
                        math.degrees(
                            float(
                                tripod.data.joint_pos[
                                    0, pan_id
                                ]
                            )
                        )
                        - self.pan_zero
                    )
                    > base.PAN_PHYSICAL_LIMIT_DEG + 1.0
                ):
                    abort_reason = "PAN_LIMIT"

                elif (
                    abs(
                        math.degrees(
                            float(
                                robot.data.joint_pos[
                                    0, ids[0]
                                ]
                            )
                        )
                    )
                    > base.WAIST_YAW_LIMIT_DEG + 2.0
                ):
                    abort_reason = "WAIST_LIMIT"

                if abort_reason != "OK":
                    break

            result_state = self.extract(
                robot,
                tripod,
                ids,
            )

            terminal_deg = math.degrees(
                float(
                    tripod.data.joint_pos[
                        0, pan_id
                    ]
                )
            )

            tilt_deg = math.degrees(
                float(
                    tripod.data.joint_pos[
                        0, tilt_id
                    ]
                )
            )

            return {
                "mode": mode,
                "start": start_json,
                "steps_applied": len(applied),
                "safety_status": abort_reason,
                "unfeasible_action_steps": infeasible,
                "feedback_replans": feedback_replans,
                "controller_mode": (
                    "measured-state MPC"
                    if mode == "planned"
                    else "fixed held targets"
                ),
                "applied_targets_rad": applied,
                "actual_terminal_pan_error_deg": abs(
                    (terminal_deg - self.pan_zero)
                    - self.reference_deg
                ),
                "end_pan_abs_deg": terminal_deg,
                "end_tilt_abs_deg": tilt_deg,
                "end_robot_q_rad": self.to_json(
                    result_state
                )["robot_q_rad"],
                "max_grasp_error_m": grip_max,
                "max_joint_speed_rad_s": max_velocity,
                "max_joint_limit_excess_rad": max_excess,
                "max_tilt_change_deg": peak_tilt,
            }

        def run_once(
            self,
            relative_time,
            phase,
            grip_attached,
            desired_pan,
            pan_zero,
            grip_pos,
            grip_quat,
            robot,
            tripod,
            command,
            sim,
            scene,
            dt,
        ):
            if self.completed:
                return False

            if (
                phase != "ACTOR_TRACKING"
                or not grip_attached
                or pan_zero is None
                or relative_time + 1e-6
                < float(case["tracking_time_s"])
            ):
                return False

            if grip_pos is None or grip_quat is None:
                raise RuntimeError(
                    "No grasp calibration available"
                )

            self.grip_pos = (
                grip_pos.detach().clone()
            )

            self.grip_quat = (
                grip_quat.detach().clone()
            )

            self.pan_zero = math.degrees(
                float(pan_zero[0])
            )

            self.reference_deg = float(
                desired_pan
            )

            self.full_command = (
                command.detach().clone()
            )

            rn = {
                name: i
                for i, name in enumerate(
                    robot.data.joint_names
                )
            }

            ids = [
                rn[name]
                for name in JOINTS
            ]

            tilt_id = list(
                tripod.data.joint_names
            ).index("tilt_joint")

            snapshot = self.extract(
                robot,
                tripod,
                ids,
            )

            (
                self.held_targets,
                self.planned_targets,
                self.predicted,
            ) = self.plan(
                snapshot,
                self.full_command,
                robot,
                tripod,
                dt,
                math.radians(
                    self.reference_deg
                    + self.pan_zero
                ),
                ids,
            )

            # Branch 1: HELD from original state.
            held = self.trial(
                "held",
                self.held_targets,
                snapshot,
                robot,
                tripod,
                sim,
                scene,
                dt,
                ids,
                tilt_id,
            )

            # Branch 2: Restore then repeat HELD.
            self.restore(
                robot,
                tripod,
                scene,
                sim,
                snapshot,
                self.full_command,
                tilt_id,
            )

            repeat = self.trial(
                "held_repeat",
                self.held_targets,
                snapshot,
                robot,
                tripod,
                sim,
                scene,
                dt,
                ids,
                tilt_id,
            )

            # Branch 3: Restore then run MPC.
            self.restore(
                robot,
                tripod,
                scene,
                sim,
                snapshot,
                self.full_command,
                tilt_id,
            )

            planned = self.trial(
                "planned",
                self.planned_targets,
                snapshot,
                robot,
                tripod,
                sim,
                scene,
                dt,
                ids,
                tilt_id,
            )

            tolerances = {
                "joint_rad": 2e-4,
                "velocity_rad_s": 2e-3,
                "root_state": 1e-4,
                "body_pose": 2e-4,
                "null_pan_deg": 0.03,
                "null_tilt_deg": 0.03,
                "null_q_rad": 0.002,
            }

            report_path = (
                out / "paired_physics_comparison.json"
            )

            report = create_report(
                held,
                repeat,
                planned,
                tolerances,
                report_path,
            )

            # This is the initial open-loop forecast.
            # It is NOT a prediction of the subsequently executed
            # receding-horizon feedback-MPC trajectory.
            report[
                "initial_open_loop_model_forecast_NOT_EXECUTED_PLAN"
            ] = self.predicted

            report["planned_controller"] = (
                "8-step receding horizon; "
                "replan from measured state each step"
            )

            report["planned_feedback_replans"] = (
                planned["feedback_replans"]
            )

            report_path.write_text(
                json.dumps(
                    report,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                "\n[RAVEN PAIRED] "
                "8-step actual physics "
                "(same-state restoration)"
            )

            print(
                "[RAVEN PAIRED] HELD        "
                f"{held['actual_terminal_pan_error_deg']:.5f} deg"
            )

            print(
                "[RAVEN PAIRED] HELD repeat "
                f"{repeat['actual_terminal_pan_error_deg']:.5f} deg"
            )

            print(
                "[RAVEN PAIRED] PLANNED     "
                f"{planned['actual_terminal_pan_error_deg']:.5f} deg"
            )

            print(
                "[RAVEN PAIRED] Reproducible held replay: "
                f"{report['physics_restore_replay_reproducible']}"
            )

            print(
                "[RAVEN PAIRED] Matched full state: "
                f"{report['start_states_matched']}"
            )

            print(
                "[RAVEN PAIRED] MPC replans completed: "
                f"{planned['feedback_replans']}/{HORIZON}"
            )

            print(
                "[RAVEN PAIRED] Commands feasible: "
                f"{report['planned_commands_feasible_without_runtime_clipping']}"
            )

            print(
                "[RAVEN PAIRED] VALID comparison: "
                f"{report['valid_comparison']}"
            )

            print(
                f"[RAVEN PAIRED] Saved {report_path}"
            )

            self.completed = True
            return True

    paired = PairedTest()

    source = ast.parse(
        TRACKER.read_text(
            encoding="utf-8"
        ),
        filename=str(TRACKER),
    )

    funcs = [
        n
        for n in source.body
        if isinstance(
            n,
            ast.FunctionDef,
        )
        and n.name == "run"
    ]

    if len(funcs) != 1:
        raise RuntimeError(
            "Validated tracker has unexpected run() definition"
        )

    fn = funcs[0]

    loops = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.While)
        and "simulation_app.is_running()"
        in ast.unparse(n.test)
    ]

    if len(loops) != 1:
        raise RuntimeError(
            "Unable to locate validated tracking loop"
        )

    loop = loops[0]

    positions = [
        i
        for i, node in enumerate(
            loop.body
        )
        if isinstance(node, ast.Expr)
        and ast.unparse(node)
        == "robot.set_joint_position_target(command)"
    ]

    if len(positions) != 1:
        raise RuntimeError(
            "Unable to locate safe hook point before physics step"
        )

    loop.body[
        positions[0]:positions[0]
    ] = ast.parse('''
if raven_paired_trial(relative_time, phase, grip_attached, desired_pan, pan_zero,
                      grip_pos, grip_quat, robot, tripod, command, sim, scene, dt):
    status = "PAIRED_PHYSICS_WINDOW_DONE"
    break
''').body

    program = ast.fix_missing_locations(
        ast.Module(
            body=[fn],
            type_ignores=[],
        )
    )

    base.raven_paired_trial = (
        paired.run_once
    )

    exec(
        compile(
            program,
            str(TRACKER),
            "exec",
        ),
        base.__dict__,
    )

    try:
        base.main()

        if not paired.completed:
            raise RuntimeError(
                "Physics window never ran; "
                "check console and tracking reference"
            )

    finally:
        base.simulation_app.close()


def main():
    args = parser().parse_args()

    if args.iterations < 1:
        raise SystemExit(
            "--iterations must be positive"
        )

    for path in (
        TRACKER,
        MODEL_MODULE,
        POLICY_MODULE,
        args.model,
    ):
        if not path.is_file():
            raise SystemExit(
                f"Missing required input: {path}"
            )

    selected = load_case(args)
    worker_main(args, selected)


if __name__ == "__main__":
    main()
