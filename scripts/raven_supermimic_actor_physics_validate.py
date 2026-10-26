
"""RAVEN V1.1: physical eight-step validation of the selected trained actor.

Uses the unchanged validated G1 RGB-D controller, via an in-memory AST hook.
Runs HELD -> HELD_REPEAT -> ACTOR from the same saved G1/tripod state.
A strict HELD-repeat null test guards against PhysX restoration differences.
Only the actor's eight joint position targets change; all other commands hold.

This is NOT a full-trajectory comparison and NOT proof of a safe real robot
controller. Internal PhysX contact/solver caches are not exactly restorable.

Run with env_isaaclab from raven-g1:
    python scripts/raven_supermimic_actor_physics_validate.py --case 0
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

# IMPORTANT: Do not import NumPy before Isaac Sim Kit starts.
# Kit adds its own bundled C-extension search paths during AppLauncher startup.
# An eager import can mix incompatible NumPy 2.x / Kit prebundle modules.

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACKER = HERE / 'raven_g1_actor_tracking.py'
DEFAULT_ACTOR = (ROOT / 'raven_training_results/policy_v2_actor_critic/'
                 'train_v1_1/actor_critic_v1_1_best.pt')
DEFAULT_METRICS = (ROOT / 'raven_training_results/policy_v1/'
                   'policy_v1_frame_corrected_metrics.json')
DEFAULT_OUT = ROOT / 'raven_training_results/policy_v2_actor_physics'
TRACKER_SHA256 = '2b7da3c47a8614b690bb20f2f461d0bb93a90a90b9f0c95a3d6c6b4648ecd038'
HORIZON = 8


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=int, default=0)
    p.add_argument('--actor', type=Path, default=DEFAULT_ACTOR)
    p.add_argument('--metrics', type=Path, default=DEFAULT_METRICS)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    return p


def load_case(path, index):
    doc = json.loads(path.read_text(encoding='utf-8'))
    if 'coordinate-aligned' not in doc.get('experiment', ''):
        raise ValueError('Expected coordinate-aligned Policy V1.1 case metrics')
    cases = doc['cases_detail']
    if not 0 <= index < len(cases):
        raise ValueError(f'Case index must be 0..{len(cases)-1}')
    return cases[index]


def compare(held, repeat, actor, out):
    """All acceptance thresholds match the existing tested V1.4 comparison."""
    import numpy as np  # Delayed: compare() runs only after Kit startup.

    def maxdiff(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    initial = held['start']
    start_diffs = {}
    for label, test in [('held_repeat', repeat), ('actor', actor)]:
        other = test['start']
        start_diffs[label] = {
            'q_rad': maxdiff(initial['robot_q_rad'], other['robot_q_rad']),
            'dq_rad_s': maxdiff(initial['robot_dq_rad_s'], other['robot_dq_rad_s']),
            'tripod_q_rad': maxdiff(initial['tripod_q_rad'], other['tripod_q_rad']),
            'tripod_dq_rad_s': maxdiff(initial['tripod_dq_rad_s'], other['tripod_dq_rad_s']),
            'robot_root': maxdiff(initial['robot_root_state_w'], other['robot_root_state_w']),
            'tripod_root': maxdiff(initial['tripod_root_state_w'], other['tripod_root_state_w']),
            'hand_pose': maxdiff(initial['hand_pose_w'], other['hand_pose_w']),
            'handle_pose': maxdiff(initial['handle_pose_w'], other['handle_pose_w']),
            'reference_deg': abs(initial['reference_relative_deg'] - other['reference_relative_deg']),
        }

    tolerances = {
        'joint_rad': 2e-4,
        'velocity_rad_s': 2e-3,
        'root_state': 1e-4,
        'body_pose': 2e-4,
        'null_pan_deg': 0.03,
        'null_tilt_deg': 0.03,
        'null_q_rad': 0.002,
    }
    matched = all(
        d['q_rad'] <= tolerances['joint_rad']
        and d['dq_rad_s'] <= tolerances['velocity_rad_s']
        and d['tripod_q_rad'] <= tolerances['joint_rad']
        and d['tripod_dq_rad_s'] <= tolerances['velocity_rad_s']
        and d['robot_root'] <= tolerances['root_state']
        and d['tripod_root'] <= tolerances['root_state']
        and d['hand_pose'] <= tolerances['body_pose']
        and d['handle_pose'] <= tolerances['body_pose']
        and d['reference_deg'] <= 1e-6
        for d in start_diffs.values()
    )
    replay_diffs = {
        'end_pan_deg': abs(held['end_pan_abs_deg'] - repeat['end_pan_abs_deg']),
        'end_tilt_deg': abs(held['end_tilt_abs_deg'] - repeat['end_tilt_abs_deg']),
        'max_robot_joint_rad': maxdiff(held['end_robot_q_rad'], repeat['end_robot_q_rad']),
    }
    repeat_ok = (
        replay_diffs['end_pan_deg'] <= tolerances['null_pan_deg']
        and replay_diffs['end_tilt_deg'] <= tolerances['null_tilt_deg']
        and replay_diffs['max_robot_joint_rad'] <= tolerances['null_q_rad']
    )
    safe = all(b['safety_status'] == 'OK' for b in (held, repeat, actor))
    complete = all(b['steps_applied'] == HORIZON for b in (held, repeat, actor))
    feasible = actor['unfeasible_action_steps'] == 0
    valid = matched and repeat_ok and safe and complete and feasible
    improvement = held['actual_terminal_pan_error_deg'] - actor['actual_terminal_pan_error_deg']
    out_report = {
        'experiment': 'RAVEN V1.1 learned actor paired Isaac Sim physics validation',
        'comparison': 'eight-step HELD vs selected V1.1 actor; not continuous tracker',
        'valid_comparison': bool(valid),
        'start_states_matched': bool(matched),
        'physics_restore_replay_reproducible': bool(repeat_ok),
        'all_conditions_safe': bool(safe),
        'actor_commands_feasible_without_clipping': bool(feasible),
        'complete_eight_steps': bool(complete),
        'start_differences': start_diffs,
        'replay_differences': replay_diffs,
        'tolerances': tolerances,
        'held_measured_error_deg': held['actual_terminal_pan_error_deg'],
        'actor_measured_error_deg': actor['actual_terminal_pan_error_deg'],
        'candidate_error_change_deg_NOT_CLAIMABLE_IF_INVALID': improvement,
        'measured_improvement_percent': (
            100.0 * improvement / held['actual_terminal_pan_error_deg']
            if valid and held['actual_terminal_pan_error_deg'] > 1e-9 else None
        ),
        'held': held, 'held_repeat': repeat, 'actor': actor,
        'limitations': [
            'PhysX solver/contact caches may not fully restore; HELD-repeat is mandatory.',
            'Frozen RGB-D reference and actor over 8 steps (66.7 ms); not a full trajectory.',
            'Actor checkpoint trained offline with a world model omitting contact/collisions.',
            'Fixed-joint-assisted grasp is not unaided robotic finger manipulation.',
            'Safety checks are simulator-only; no real robot actuation.',
            'Actor inference timing excludes Isaac Sim camera/comms and other integration overhead.',
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_report, indent=2), encoding='utf-8')
    return out_report


def run(args, case):
    if hashlib.sha256(TRACKER.read_bytes()).hexdigest() != TRACKER_SHA256:
        raise SystemExit('Validated tracking controller SHA changed; refusing AST instrumentation')

    out = args.out.resolve() / f'case_{args.case:02d}'
    out.mkdir(parents=True, exist_ok=True)
    # The tracker initializes Isaac Sim on import. Save its original argv.
    original_argv = sys.argv[:]
    sys.argv = [str(TRACKER), '--actor-amplitude-m', '0.25',
                '--actor-period-s', '5.0', '--tracking-duration-s', '35.0',
                '--output-dir', str(out / 'tracking')]
    try:
        spec = importlib.util.spec_from_file_location('raven_actor_physics_base', TRACKER)
        if spec is None or spec.loader is None:
            raise RuntimeError('Cannot load validated tracker')
        base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = base
        spec.loader.exec_module(base)
    finally:
        sys.argv = original_argv

    # Import binary packages only AFTER the tracker launches Isaac Sim Kit.
    # This matches the validated MPC physics script import order.
    import numpy as np
    import torch
    from raven_supermimic_world_model_v2 import JOINTS
    from raven_supermimic_actor_critic_v1 import Actor

    checkpoint = torch.load(args.actor, map_location='cpu', weights_only=True)
    if (checkpoint.get('model_class') != 'RAVENOfflineActorCriticV1_1'
            or tuple(checkpoint['joint_names']) != tuple(JOINTS)
            or int(checkpoint['horizon']) != HORIZON
            or int(checkpoint['selected_epoch']) != 1
            or checkpoint['passed_validation_gate'] is not True):
        raise SystemExit('Require the selected, accepted epoch-1 V1.1 actor checkpoint')
    weights = checkpoint['actor_state_dict']
    actor = Actor(weights['state_mean'], weights['state_scale'])
    actor.load_state_dict(weights, strict=True)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor.to(device)


    class Paired:
        def __init__(self):
            self.completed = False

        @staticmethod
        def state(robot, tripod):
            rb = list(robot.data.body_names)
            tb = list(tripod.data.body_names)
            return {
                'robot_q_rad': robot.data.joint_pos[0].clone(),
                'robot_dq_rad_s': robot.data.joint_vel[0].clone(),
                'tripod_q_rad': tripod.data.joint_pos[0].clone(),
                'tripod_dq_rad_s': tripod.data.joint_vel[0].clone(),
                'robot_root_state_w': robot.data.root_state_w[0].clone(),
                'tripod_root_state_w': tripod.data.root_state_w[0].clone(),
                'hand_pose_w': robot.data.body_pose_w[0, rb.index('left_base_link')].clone(),
                'handle_pose_w': tripod.data.body_pose_w[0, tb.index('control_handle_link')].clone(),
            }

        @staticmethod
        def numbers(tensors):
            return {name: arr.detach().cpu().reshape(-1).tolist()
                    for name, arr in tensors.items()}

        def restore(self, robot, tripod, scene, sim, snapshot, full, tilt_id):
            for asset, prefix in ((robot, 'robot'), (tripod, 'tripod')):
                root = snapshot[f'{prefix}_root_state_w'].unsqueeze(0)
                asset.write_root_pose_to_sim(root[:, :7].clone())
                asset.write_root_velocity_to_sim(root[:, 7:13].clone())
                asset.write_joint_state_to_sim(
                    snapshot[f'{prefix}_q_rad'].unsqueeze(0).clone(),
                    snapshot[f'{prefix}_dq_rad_s'].unsqueeze(0).clone(),
                )
            robot.set_joint_position_target(full)
            tripod.set_joint_position_target(
                base.torch_column(0.0, sim.device), joint_ids=[tilt_id]
            )
            scene.write_data_to_sim()
            scene.update(0.0)

        def grasp_error(self, robot, tripod):
            hand_id = list(robot.data.body_names).index('left_base_link')
            handle_id = list(tripod.data.body_names).index('control_handle_link')
            handle = tripod.data.body_pose_w[:, handle_id]
            hand = robot.data.body_pose_w[:, hand_id]
            grip_target, _ = base.combine_frame_transforms(
                handle[:, :3], handle[:, 3:7], self.grip_pos, self.grip_quat
            )
            return float(torch.linalg.vector_norm(
                hand[:, :3] - grip_target, dim=-1
            )[0])

        def actor_action(self, robot, tripod, ids, ref_abs):
            tripod_joint_names = list(tripod.data.joint_names)
            pan_id = tripod_joint_names.index('pan_joint')
            tilt_id = tripod_joint_names.index('tilt_joint')
            state = torch.cat((
                robot.data.joint_pos[:, ids],
                robot.data.joint_vel[:, ids],
                tripod.data.joint_pos[:, [pan_id, tilt_id]],
                tripod.data.joint_vel[:, [pan_id, tilt_id]],
            ), dim=-1).detach().to(device)
            held = self.full[:, ids].detach().to(device)
            reference = torch.full((1, 1), ref_abs, dtype=state.dtype, device=device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter_ns()
            with torch.inference_mode():
                proposed = actor(state, held, reference)[0]
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            return proposed.to(robot.data.joint_pos.device), elapsed

        def branch(self, mode, robot, tripod, sim, scene, dt, ids, tilt_id):
            pan_id = list(tripod.data.joint_names).index('pan_joint')
            start_state = self.numbers(self.state(robot, tripod))
            start_state['reference_relative_deg'] = self.reference_deg
            tilt_start = float(tripod.data.joint_pos[0, tilt_id])
            max_grasp = self.grasp_error(robot, tripod)
            max_speed = 0.0
            max_limit = 0.0
            max_tilt = 0.0
            applied = []
            inference_ms = []
            failure = 'OK'
            infeasible = 0

            for step in range(HORIZON):
                if mode == 'actor':
                    try:
                        proposed, ms = self.actor_action(
                            robot, tripod, ids,
                            math.radians(self.pan_zero + self.reference_deg)
                        )
                        inference_ms.append(ms)
                    except Exception as exc:
                        failure = 'ACTOR_INFERENCE_FAILURE'
                        print(f'[RAVEN ACTOR] step {step+1} failed: {exc}')
                        break
                else:
                    proposed = self.full[0, ids]

                q = robot.data.joint_pos[0, ids]
                limits = robot.data.joint_pos_limits[0, ids]
                max_step = torch.full_like(q, base.MAX_ARM_STEP)
                max_step[0] = base.MAX_WAIST_STEP
                feasible = (
                    torch.isfinite(proposed)
                    & (proposed >= limits[:, 0] - 1e-6)
                    & (proposed <= limits[:, 1] + 1e-6)
                    & ((proposed - q).abs() <= max_step + 1e-6)
                )
                if not bool(feasible.all()):
                    infeasible += 1
                    failure = 'INFEASIBLE_COMMAND_NO_CLIPPING'
                    failed = [
                        {'joint': name,
                         'target_gap_rad': float(abs(proposed[j] - q[j])),
                         'max_gap_rad': float(max_step[j])}
                        for j, name in enumerate(JOINTS) if not bool(feasible[j])
                    ]
                    print(f'[RAVEN ACTOR] step {step+1}: rejected targets {failed}')
                    break

                full = self.full.clone()
                full[:, ids] = proposed.unsqueeze(0)
                robot.set_joint_position_target(full)
                tripod.set_joint_position_target(
                    base.torch_column(0.0, sim.device), joint_ids=[tilt_id]
                )
                scene.write_data_to_sim()
                sim.step(render=True)
                scene.update(dt)
                applied.append(proposed.detach().cpu().tolist())
                max_grasp = max(max_grasp, self.grasp_error(robot, tripod))
                max_speed = max(max_speed, float(robot.data.joint_vel[0, ids].abs().max()))
                max_tilt = max(max_tilt, abs(math.degrees(
                    float(tripod.data.joint_pos[0, tilt_id]) - tilt_start
                )))
                q_now = robot.data.joint_pos[0, ids]
                excess = torch.maximum(limits[:, 0] - q_now,
                                       q_now - limits[:, 1]).clamp(min=0)
                max_limit = max(max_limit, float(excess.max()))
                if max_limit > 0.01:
                    failure = 'JOINT_LIMIT'
                elif max_grasp > base.MAX_HAND_POSITION_ERROR_M:
                    failure = 'GRASP_ERROR'
                elif abs(math.degrees(float(tripod.data.joint_pos[0, pan_id]))
                         - self.pan_zero) > base.PAN_PHYSICAL_LIMIT_DEG + 1.0:
                    failure = 'PAN_LIMIT'
                elif abs(math.degrees(float(robot.data.joint_pos[0, ids[0]]))) > (
                        base.WAIST_YAW_LIMIT_DEG + 2.0):
                    failure = 'WAIST_LIMIT'
                if failure != 'OK':
                    break

            end_state = self.numbers(self.state(robot, tripod))
            terminal_pan = math.degrees(float(tripod.data.joint_pos[0, pan_id]))
            terminal_tilt = math.degrees(float(tripod.data.joint_pos[0, tilt_id]))
            return {
                'mode': mode,
                'start': start_state,
                'steps_applied': len(applied),
                'safety_status': failure,
                'unfeasible_action_steps': infeasible,
                'actor_inferences': len(inference_ms),
                'inference_wall_ms': inference_ms,
                'controller_mode': 'V1.1 direct feedback actor' if mode == 'actor'
                                   else 'fixed held targets',
                'applied_targets_rad': applied,
                'actual_terminal_pan_error_deg': abs(
                    terminal_pan - self.pan_zero - self.reference_deg
                ),
                'end_pan_abs_deg': terminal_pan,
                'end_tilt_abs_deg': terminal_tilt,
                'end_robot_q_rad': end_state['robot_q_rad'],
                'max_grasp_error_m': max_grasp,
                'max_joint_speed_rad_s': max_speed,
                'max_joint_limit_excess_rad': max_limit,
                'max_tilt_change_deg': max_tilt,
            }

        def trial(self, relative_time, phase, grip_attached, desired_pan, pan_zero,
                  grip_pos, grip_quat, robot, tripod, command, sim, scene, dt):
            if self.completed:
                return False
            if (phase != 'ACTOR_TRACKING' or not grip_attached or pan_zero is None
                    or relative_time + 1e-6 < float(case['tracking_time_s'])):
                return False
            if grip_pos is None or grip_quat is None:
                raise RuntimeError('No grasp calibration at requested tracking phase')
            self.grip_pos = grip_pos.detach().clone()
            self.grip_quat = grip_quat.detach().clone()
            self.pan_zero = math.degrees(float(pan_zero[0]))
            self.reference_deg = float(desired_pan)
            self.full = command.detach().clone()
            robot_ids = {name: i for i, name in enumerate(robot.data.joint_names)}
            ids = [robot_ids[name] for name in JOINTS]
            tilt_id = list(tripod.data.joint_names).index('tilt_joint')
            snapshot = self.state(robot, tripod)

            held = self.branch('held', robot, tripod, sim, scene, dt, ids, tilt_id)
            self.restore(robot, tripod, scene, sim, snapshot, self.full, tilt_id)
            repeat = self.branch('held_repeat', robot, tripod, sim, scene, dt, ids, tilt_id)
            self.restore(robot, tripod, scene, sim, snapshot, self.full, tilt_id)
            controlled = self.branch('actor', robot, tripod, sim, scene, dt, ids, tilt_id)

            output = out / 'actor_physics_comparison.json'
            report = compare(held, repeat, controlled, output)
            report['case_index'] = args.case
            report['tracking_time_s'] = float(relative_time)
            report['actor_checkpoint'] = str(args.actor.resolve())
            report['actor_selected_epoch'] = int(checkpoint['selected_epoch'])
            if controlled['inference_wall_ms']:
                report['actor_inference_median_ms_in_sim'] = float(np.median(
                    controlled['inference_wall_ms']))
                report['actor_inference_max_ms_in_sim'] = float(np.max(
                    controlled['inference_wall_ms']))
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')

            print('\n[RAVEN ACTOR PHYSICS] Eight-step paired results')
            print(f"  HELD:         {held['actual_terminal_pan_error_deg']:.5f} deg")
            print(f"  HELD REPEAT:  {repeat['actual_terminal_pan_error_deg']:.5f} deg")
            print(f"  ACTOR:        {controlled['actual_terminal_pan_error_deg']:.5f} deg")
            print(f"  Start matched: {report['start_states_matched']}")
            print(f"  Held repeat:   {report['physics_restore_replay_reproducible']}")
            print(f"  Actor steps:   {controlled['steps_applied']}/{HORIZON}")
            print(f"  Feasible:      {report['actor_commands_feasible_without_clipping']}")
            print(f"  All safe:      {report['all_conditions_safe']}")
            print(f"  VALID:         {report['valid_comparison']}")
            if report['valid_comparison']:
                print(f"  Error change:  {report['measured_improvement_percent']:+.2f}% reduction")
            if 'actor_inference_median_ms_in_sim' in report:
                print(f"  Actor median inference: {report['actor_inference_median_ms_in_sim']:.3f} ms")
            print(f'  Saved: {output}')
            self.completed = True
            return True

    paired = Paired()
    source = ast.parse(TRACKER.read_text(encoding='utf-8'), filename=str(TRACKER))
    functions = [n for n in source.body
                 if isinstance(n, ast.FunctionDef) and n.name == 'run']
    if len(functions) != 1:
        raise RuntimeError('Unexpected tracker run() definition')
    run_fn = functions[0]
    loops = [n for n in ast.walk(run_fn)
             if isinstance(n, ast.While)
             and 'simulation_app.is_running()' in ast.unparse(n.test)]
    if len(loops) != 1:
        raise RuntimeError('Validated tracker loop not found')
    loop = loops[0]
    positions = [i for i, node in enumerate(loop.body)
                 if isinstance(node, ast.Expr)
                 and ast.unparse(node) == 'robot.set_joint_position_target(command)']
    if len(positions) != 1:
        raise RuntimeError('Tracker safe hook not found')
    loop.body[positions[0]:positions[0]] = ast.parse('''
if raven_actor_physics_trial(relative_time, phase, grip_attached, desired_pan,
                              pan_zero, grip_pos, grip_quat, robot, tripod,
                              command, sim, scene, dt):
    status = "ACTOR_PHYSICS_WINDOW_DONE"
    break
''').body
    dynamic_run = ast.fix_missing_locations(ast.Module(body=[run_fn], type_ignores=[]))
    base.raven_actor_physics_trial = paired.trial
    exec(compile(dynamic_run, str(TRACKER), 'exec'), base.__dict__)
    try:
        base.main()
        if not paired.completed:
            raise RuntimeError('Requested tracking case not reached')
    finally:
        base.simulation_app.close()


def main():
    args = parser().parse_args()
    for required in [TRACKER, args.actor, args.metrics,
                     HERE / 'raven_supermimic_actor_critic_v1.py',
                     HERE / 'raven_supermimic_world_model_v2.py',
                     HERE / 'raven_supermimic_policy_v1.py']:
        if not required.is_file():
            raise SystemExit(f'Missing required file: {required}')
    case = load_case(args.metrics, args.case)
    run(args, case)


if __name__ == '__main__':
    main()
