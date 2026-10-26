"""RAVEN V3: resumable, bounded-budget actor/critic optimisation on Isaac Sim replay.

This trainer performs N gradient updates of EACH actor and critic per run.
It does not create N new simulator episodes. World Model V2 is fixed during
this stage; add fresh/varied simulator captures before trusting long training.

Run from RAVEN repo root (env_isaaclab), after copying to scripts/:
  python scripts/raven_supermimic_policy_train_v3.py --updates 500 --eval-every 100
  python scripts/raven_supermimic_policy_train_v3.py --updates 5000 --eval-every 250 --resume

Optional --score-alignment-weight adds a differentiable approximation to
checkpoint-selection score to the existing imagined actor-critic objective.
Choose a new --out for this experiment. All validation
metrics are world-model-predicted, not physical Isaac Sim A/B measurements.
Not approved for real G1 deployment.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from raven_supermimic_training_v2 import load_model, load_actor, finite_parameters
from raven_supermimic_actor_critic_v1 import DEFAULT_RUNS, build_samples, td_lambda_targets
from raven_supermimic_actor_critic_v1_1 import imagined_rollout, evaluate, eligible_and_score
from raven_supermimic_world_model_v2 import HORIZON, JOINTS, DEG

ROOT = Path(__file__).resolve().parent.parent
CYCLE_1 = ROOT / 'raven_training_results' / 'supermimic_online_v2' / 'cycle_01'
DEFAULT_CAPTURE = CYCLE_1 / 'experience' / 'raven_g1_state_action_120hz.jsonl'
DEFAULT_MODEL = CYCLE_1 / 'world_model_online.pt'
DEFAULT_ACTOR = CYCLE_1 / 'actor_online_selected.pt'
DEFAULT_OUT = ROOT / 'raven_training_results' / 'policy_v3_long_training'


def atomic_torch_save(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def atomic_json(value, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def append_samples(groups):
    return tuple(torch.cat([group[i] for group in groups], dim=0) for i in range(3))


def differentiable_selection_score(initial, reference, final, actions, held, actor):
    """Train-time surrogate for the EXACT selection components.

    Checkpoint selection remains based on held-out validation, not this training
    penalty. Smooth absolute values let gradients pass through small errors.
    """
    def smooth_absolute(tensor, smoothing=1e-4):
        return torch.sqrt(tensor.square() + smoothing * smoothing) - smoothing

    pan_deg = (final[:, 16] - reference[:, 0]) * DEG
    tilt_deg = (final[:, 17] - initial[:, 17]) * DEG
    previous = torch.cat((held[:, None, :], actions[:, :-1]), dim=1)
    step = (actions - previous) / actor.delta
    # Same three components and coefficients used by eligible_and_score:
    # pan_abs_deg + 2 * tilt_abs_deg + 0.03 * mean_normalised_step.
    return (smooth_absolute(pan_deg)
            + 2.0 * smooth_absolute(tilt_deg)
            + 0.03 * smooth_absolute(step).mean(dim=(1, 2))).mean()


def make_checkpoint(actor, critic, source_model, iteration, accepted, **extra):
    ckpt = {
        'model_class': 'RAVENOnlineActorCriticV2',
        'actor_state_dict': snapshot(actor),
        'critic_state_dict': snapshot(critic),
        'joint_names': JOINTS,
        'horizon': HORIZON,
        'model_source': str(source_model),
        'selected_epoch': int(iteration),
        'training_updates': int(iteration),
        'passed_validation_gate': bool(accepted),
        'training_method': 'RAVEN V3 long-budget imagined actor-critic updates',
    }
    ckpt.update(extra)
    return ckpt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--world-model', type=Path, default=DEFAULT_MODEL)
    ap.add_argument('--actor', type=Path, default=DEFAULT_ACTOR)
    ap.add_argument('--capture', type=Path, default=DEFAULT_CAPTURE,
                    help='Most recent physically collected Isaac Sim episode')
    ap.add_argument('--historical', type=Path, nargs='*', default=DEFAULT_RUNS,
                    help='Older development captures (never reserved independent test)')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--updates', type=int, default=5000,
                    help='Total actor AND critic gradient updates, not simulator episodes')
    ap.add_argument('--eval-every', type=int, default=250)
    ap.add_argument('--checkpoint-every', type=int, default=250)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--policy-stride', type=int, default=12)
    ap.add_argument('--recent-fraction', type=float, default=0.5,
                    help='Training minibatch fraction from latest capture')
    ap.add_argument('--actor-lr', type=float, default=8e-5)
    ap.add_argument('--critic-lr', type=float, default=1e-4)
    ap.add_argument('--tilt-weight', type=float, default=5.0)
    ap.add_argument('--action-weight', type=float, default=0.08)
    ap.add_argument('--rate-weight', type=float, default=0.2)
    ap.add_argument('--score-alignment-weight', type=float, default=0.0,
                    help='Experimental differentiable terminal score guidance; 0 keeps prior V3 behaviour')
    ap.add_argument('--tilt-margin-deg', type=float, default=0.02)
    ap.add_argument('--tilt-ceiling-deg', type=float, default=0.08)
    ap.add_argument('--min-pan-gain-deg', type=float, default=0.002)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    for name in ('updates', 'eval_every', 'checkpoint_every', 'batch_size', 'policy_stride'):
        if getattr(args, name) <= 0:
            ap.error(f'--{name.replace("_", "-")} must be > 0')
    if args.score_alignment_weight < 0 or not math.isfinite(args.score_alignment_weight):
        ap.error('--score-alignment-weight must be finite and nonnegative')
    if not 0.0 <= args.recent_fraction <= 1.0:
        ap.error('--recent-fraction must lie in [0,1]')
    for p in [args.world_model, args.actor, args.capture, *args.historical]:
        if not p.is_file():
            ap.error(f'Missing input file: {p}')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress = out / 'training_state.pt'
    selected = out / 'actor_v3_selected.pt'
    manifest = out / 'training_progress.json'
    if progress.exists() and not args.resume:
        ap.error(f'Existing training state at {progress}; pass --resume to continue')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[RAVEN V3] {device} | target updates: {args.updates}', flush=True)
    model, _ = load_model(args.world_model, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    actor, critic, _ = load_actor(args.actor, model, device)
    target = copy.deepcopy(critic).eval()
    for p in target.parameters():
        p.requires_grad_(False)

    print('[DATA] Loading latest and historical training/validation partitions...', flush=True)
    recent_train, recent_val, _ = build_samples([args.capture], stride=args.policy_stride)
    if args.historical:
        old_train, old_val, _ = build_samples(args.historical, stride=args.policy_stride)
    else:
        old_train, old_val = recent_train, recent_val
    combined_val = append_samples([recent_val, old_val]) if args.historical else recent_val
    print(f'[DATA] recent train={len(recent_train[0])}, old train={len(old_train[0])}, '
          f'holdout={len(combined_val[0])}', flush=True)
    cfg = SimpleNamespace(tilt_weight=args.tilt_weight,
                          action_weight=args.action_weight,
                          rate_weight=args.rate_weight,
                          tilt_margin_deg=args.tilt_margin_deg,
                          tilt_absolute_ceiling_deg=args.tilt_ceiling_deg,
                          min_pan_gain_deg=args.min_pan_gain_deg)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    gamma, lam = 0.98, 0.95
    paths = {'model': str(args.world_model.resolve()),
             'actor': str(args.actor.resolve()),
             'capture': str(args.capture.resolve()),
             'historical': [str(p.resolve()) for p in args.historical]}
    signature = {'sources': paths, 'batch_size': args.batch_size,
                 'policy_stride': args.policy_stride, 'recent_fraction': args.recent_fraction,
                 'actor_lr': args.actor_lr, 'critic_lr': args.critic_lr,
                 'tilt_weight': args.tilt_weight, 'action_weight': args.action_weight,
                 'rate_weight': args.rate_weight, 'tilt_margin_deg': args.tilt_margin_deg,
                 'tilt_ceiling_deg': args.tilt_ceiling_deg,
                 'min_pan_gain_deg': args.min_pan_gain_deg,
                 'score_alignment_weight': args.score_alignment_weight}

    if progress.exists():
        ckpt = torch.load(progress, map_location='cpu', weights_only=True)
        if ckpt['signature'] != signature:
            raise RuntimeError('Resume configuration or data sources changed. Use identical flags, '
                               'or choose a NEW --out directory for a different experiment.')
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        target.load_state_dict(ckpt['target'])
        actor_opt.load_state_dict(ckpt['actor_optimizer'])
        critic_opt.load_state_dict(ckpt['critic_optimizer'])
        # Adam state may have been serialized on CPU. Move it to the working device.
        for optimizer in (actor_opt, critic_opt):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value) and key != 'step':
                        state[key] = value.to(device)
        torch.set_rng_state(ckpt['cpu_rng'])
        if device.type == 'cuda' and ckpt['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
        iteration = int(ckpt['iteration'])
        best_score = float(ckpt['best_score'])
        best_iteration = int(ckpt['best_iteration'])
        logs = ckpt['log']
        baseline_result = ckpt['baseline_result']
        print(f'[RESUME] Restored iteration {iteration}; best accepted: {best_iteration}', flush=True)
    else:
        iteration = 0
        # Selection scores compare model-predicted holdouts, NOT unseen physics.
        baseline_result = evaluate(model, actor, combined_val, device,
                                   HORIZON, args.batch_size, cfg)
        accepted, score, _ = eligible_and_score(baseline_result, cfg)
        best_score = score if accepted else float('inf')
        best_iteration = 0 if accepted else -1
        logs = []
        print(f'[BASELINE] pan={baseline_result["actor_pan_abs_error_deg"]:.5f}deg '
              f'tilt={baseline_result["actor_tilt_delta_deg"]:.5f}deg '
              f'accepted={accepted}', flush=True)
    if iteration >= args.updates:
        print(f'[DONE] Saved training state already reached {iteration} steps.', flush=True)
        return

    def save_progress(step):
        state = {
            'signature': signature,
            'iteration': int(step),
            'actor': snapshot(actor),
            'critic': snapshot(critic),
            'target': snapshot(target),
            'actor_optimizer': actor_opt.state_dict(),
            'critic_optimizer': critic_opt.state_dict(),
            'cpu_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'best_score': best_score,
            'best_iteration': best_iteration,
            'baseline_result': baseline_result,
            'log': logs,
        }
        atomic_torch_save(state, progress)
        atomic_json({
            'status': 'training' if step < args.updates else 'completed',
            'updates_completed': step,
            'updates_target': args.updates,
            'best_accepted_iteration': best_iteration,
            'best_accepted_score': best_score if math.isfinite(best_score) else None,
            'selected_checkpoint': str(selected) if best_iteration > 0 else None,
            'recommended_checkpoint': str(selected) if best_iteration > 0 else str(args.actor.resolve()),
            'initial_validation': baseline_result,
            'history': logs,
            'source_paths': paths,
            'caution': 'Predicted validation only; requires fresh Isaac Sim A/B before hardware use.',
        }, manifest)

    recent_n = max(0, min(args.batch_size, round(args.batch_size * args.recent_fraction)))
    old_n = args.batch_size - recent_n
    log_actor, log_critic = [], []
    start = time.monotonic()
    try:
        for step in range(iteration + 1, args.updates + 1):
            parts = []
            if recent_n:
                index = torch.randint(len(recent_train[0]), (recent_n,))
                parts.append(tuple(t[index].to(device) for t in recent_train))
            if old_n:
                index = torch.randint(len(old_train[0]), (old_n,))
                parts.append(tuple(t[index].to(device) for t in old_train))
            batch = append_samples(parts) if len(parts) > 1 else parts[0]
            state, held, ref = batch

            actor.train(); critic.train()
            for par in critic.parameters():
                par.requires_grad_(False)
            reward, states, actions, final = imagined_rollout(
                model, actor, state, held, ref, HORIZON, cfg)
            bootstrap = target(final, held, ref)
            discounts = gamma ** torch.arange(HORIZON, device=device)
            actor_loss = -(torch.sum(reward * discounts[None, :], dim=1)
                           + gamma ** HORIZON * bootstrap).mean()
            # Guide the actor towards the *actual selection score* without
            # replacing the existing model-based RL / TD(lambda) training.
            # This is a differentiable train-batch surrogate, NOT validation.
            if args.score_alignment_weight:
                aligned_score = differentiable_selection_score(
                    state, ref, final, actions, held, actor)
                actor_loss = actor_loss + args.score_alignment_weight * aligned_score
            if not bool(torch.isfinite(actor_loss)):
                raise RuntimeError(f'Non-finite actor objective at step {step}')
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()

            for par in critic.parameters():
                par.requires_grad_(True)
            targets = td_lambda_targets(reward.detach(), states.detach(),
                                        final.detach(), held.detach(), ref.detach(),
                                        target, gamma=gamma, lam=lam)
            critic_prediction = critic(
                states.detach().reshape(-1, 20),
                held[:, None, :].expand(-1, HORIZON, -1).reshape(-1, 8),
                ref[:, None, :].expand(-1, HORIZON, -1).reshape(-1, 1)
            ).reshape(-1, HORIZON)
            critic_loss = nn.functional.smooth_l1_loss(critic_prediction, targets)
            if not bool(torch.isfinite(critic_loss)):
                raise RuntimeError(f'Non-finite critic objective at step {step}')
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            with torch.no_grad():
                for dest, src in zip(target.parameters(), critic.parameters()):
                    dest.lerp_(src, 0.01)
            log_actor.append(float(actor_loss.detach()))
            log_critic.append(float(critic_loss.detach()))

            if step % args.eval_every == 0 or step == args.updates:
                if not finite_parameters(actor) or not finite_parameters(critic):
                    raise RuntimeError('Non-finite policy parameters')
                report = evaluate(model, actor, combined_val, device,
                                  HORIZON, args.batch_size, cfg)
                accepted, score, tilt_limit = eligible_and_score(report, cfg)
                # Do not silently accept a candidate that fails on recent data.
                recent_report = evaluate(model, actor, recent_val, device,
                                         HORIZON, args.batch_size, cfg)
                recent_accepted, _, _ = eligible_and_score(recent_report, cfg)
                accepted = bool(accepted and recent_accepted)
                improved = accepted and score + 1e-5 < best_score
                if improved:
                    best_iteration, best_score = step, score
                    atomic_torch_save(make_checkpoint(actor, critic, args.world_model,
                                                      step, True,
                                                      validation_score=score,
                                                      validation=report), selected)
                logs.append({
                    'iteration': step,
                    'actor_loss_average': float(np.mean(log_actor)),
                    'critic_loss_average': float(np.mean(log_critic)),
                    'validation': report,
                    'recent_validation': recent_report,
                    'passed_gate': accepted,
                    'improved_selection': improved,
                    'score': score,
                    'tilt_limit_deg': tilt_limit,
                })
                print(f'[UPDATE {step:5d}/{args.updates}] '
                      f'actor_loss={np.mean(log_actor):.5f} critic_loss={np.mean(log_critic):.5f} '
                      f'pan={report["actor_pan_abs_error_deg"]:.4f}deg '
                      f'tilt={report["actor_tilt_delta_deg"]:.4f}deg '
                      f'accepted={accepted} selected={best_iteration}', flush=True)
                log_actor, log_critic = [], []
            if step % args.checkpoint_every == 0 or step == args.updates:
                save_progress(step)
    except BaseException:
        # Preserve last completed checkpoint; don't accidentally mark the active
        # step as resumable (optimiser may be half-updated on interruption).
        print('[INTERRUPTED] Resume from last saved training_state.pt using --resume.', flush=True)
        raise

    atomic_torch_save(make_checkpoint(actor, critic, args.world_model,
                                      args.updates, False),
                      out / 'actor_v3_last_unselected.pt')
    save_progress(args.updates)
    print(f'[COMPLETE] {args.updates} actor updates + {args.updates} critic updates '
          f'in {time.monotonic()-start:.1f}s of this invocation.', flush=True)
    if best_iteration > 0:
        print(f'[SELECTED] Best accepted at update {best_iteration}: {selected}', flush=True)
    else:
        print(f'[NO PROMOTION] Retain input actor checkpoint: {args.actor}', flush=True)
    print('[SAFETY] No real G1 deployment without diverse unseen simulator trials and safety review.')


if __name__ == '__main__':
    main()
