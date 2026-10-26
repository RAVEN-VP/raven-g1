"""RAVEN iterative simulator-in-the-loop model-based actor-critic learner.

Cycle: new G1-tripod Isaac Sim actor-guided episode -> actual physics replay ->
       warm-start V2 learned dynamics -> differentiated actor/TD(lambda) critic ->
       conservative checkpoint selection -> next actor-guided episode.

Preserves the original RAVEN tracker, original V2 checkpoint and V1.1 weights.
Does not run the independent third capture. Short 8-step rollouts remain in place.
Not official SuperMimic: no full 32-step formulation, no latency conditioning,
and currently no learned contact/collision features or real robot safety proof.

Run from the raven-g1 root in env_isaaclab (collector launches Isaac Sim):
  python scripts/raven_supermimic_training_v2.py --cycles 2 --headless
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from raven_supermimic_world_model_v2 import (
    HORIZON, JOINTS, ResidualDynamics, make_windows, read_run,
)
from raven_supermimic_actor_critic_v1 import (
    Actor, Critic, DEFAULT_RUNS, DEFAULT_MODEL, build_samples,
    td_lambda_targets,
)
from raven_supermimic_actor_critic_v1_1 import (
    imagined_rollout, evaluate, eligible_and_score,
)

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'scripts'
DEFAULT_ACTOR = (ROOT / 'raven_training_results' / 'policy_v2_actor_critic'
                 / 'train_v1_1' / 'actor_critic_v1_1_best.pt')
DEFAULT_OUTPUT = ROOT / 'raven_training_results' / 'supermimic_online_v2'


def finite_parameters(model):
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def load_model(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if (ck.get('model_class') != 'ResidualDynamics'
            or tuple(ck['joint_names']) != tuple(JOINTS)
            or int(ck['horizon']) != HORIZON):
        raise RuntimeError('World-model metadata incompatible with RAVEN V2')
    model = ResidualDynamics(*ck['normalization']).to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    return model, ck


def load_actor(path, model, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if (tuple(ck.get('joint_names', ())) != tuple(JOINTS)
            or ck.get('model_class') not in (
                'RAVENOfflineActorCriticV1_1', 'RAVENOnlineActorCriticV2')):
        raise RuntimeError('Expected V1.1-selected actor or online successor')
    a = Actor(model.state_mean, model.state_scale).to(device)
    c = Critic(model.state_mean, model.state_scale).to(device)
    a.load_state_dict(ck['actor_state_dict'], strict=True)
    c.load_state_dict(ck['critic_state_dict'], strict=True)
    return a, c, ck


def split_data(path, stride):
    s, a = read_run(path, min_time=27.5)
    cutoff = int(len(s) * 0.8)
    if cutoff < HORIZON + 25 or len(s) - cutoff < HORIZON + 25:
        raise RuntimeError(f'Capture has insufficient data for split: {path}')
    train = make_windows(s, a, 0, cutoff, stride)
    val = make_windows(s, a, cutoff, len(s), stride)
    return train, val


def update_dynamics(previous_ckpt, historical, new_capture, out, args, device):
    model, prev = load_model(previous_ckpt, device)
    old_runs = [split_data(p, args.model_stride) for p in historical]
    recent = split_data(new_capture, args.model_stride)
    train_parts = [run[0] for run in old_runs] + [recent[0]]
    val_parts = [run[1] for run in old_runs] + [recent[1]]
    train = tuple(torch.cat([part[i] for part in train_parts], 0).to(device)
                  for i in range(3))
    val = tuple(torch.cat([part[i] for part in val_parts], 0).to(device)
                for i in range(3))
    # Keep the original model's normalisation stable across cycles.
    # A separate future model version can learn normalisation and contact states.
    state_weights = torch.tensor([0.25]*8 + [0.20]*8 + [12., 6., 2., 1.],
                                 device=device)

    def loss_fn(x, y):
        delta = (x - y) / model.state_scale
        return (delta.square() * state_weights).mean()

    @torch.no_grad()
    def validation_loss():
        model.eval()
        total, count = 0.0, 0
        for indices in torch.arange(len(val[0]), device=device).split(256):
            prediction = model.rollout(val[0][indices], val[1][indices])
            total += float(loss_fn(prediction, val[2][indices])) * len(indices)
            count += len(indices)
        return total / count

    before = validation_loss()
    selected = {k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}
    best = before
    best_epoch = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.model_lr,
                                   weight_decay=1e-4)
    history = []
    for epoch in range(1, args.model_epochs + 1):
        model.train()
        order = torch.randperm(len(train[0]), device=device)
        training_loss = 0.0
        for indices in order.split(args.batch_size):
            estimate = model.rollout(train[0][indices], train[1][indices])
            loss = loss_fn(estimate, train[2][indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Non-finite world-model loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_loss += float(loss.detach()) * len(indices)
        val_loss = validation_loss()
        history.append({'epoch':epoch, 'training_loss':training_loss/len(train[0]),
                        'validation_loss': val_loss})
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            selected = {k:v.detach().cpu().clone()
                        for k,v in model.state_dict().items()}
        print(f'[WORLD MODEL] epoch {epoch:02d}: validation={val_loss:.5f}, best={best:.5f}')

    model.load_state_dict(selected, strict=True)
    model.eval()
    if not finite_parameters(model):
        raise RuntimeError('Model parameter is not finite')
    ckpt_path = out / 'world_model_online.pt'
    prev['state_dict'] = selected
    prev['metrics'] = {
        'training_method':'online warm-start with simulator actor-guided transition replay',
        'parent_checkpoint': str(previous_ckpt),
        'historical_sources':[str(p) for p in historical],
        'latest_actor_guided_capture':str(new_capture),
        'baseline_validation_loss':before,
        'selected_validation_loss':best,
        'best_epoch':best_epoch,
        'history':history,
        'has_actor_guided_sim_transitions':True,
        'horizon':HORIZON,
    }
    torch.save(prev, ckpt_path)
    return model, ckpt_path, prev['metrics']


def score_candidate(model, actor, validation, device, cfg, args):
    measurements = evaluate(model, actor, validation, device,
                            HORIZON, args.batch_size, cfg)
    accepted, score, ceiling = eligible_and_score(measurements, cfg)
    return measurements, accepted, score, ceiling


def update_actor(model, model_path, actor_path, new_capture, historical,
                 out, args, device):
    actor, critic, parent = load_actor(actor_path, model, device)
    target = copy.deepcopy(critic)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    new_training, new_validation, _ = build_samples([new_capture],
                                                      stride=args.policy_stride)
    # Evaluate on both newest-run holdout and unchanged development holdout.
    # DO NOT access the reserved independent third recording.
    _, old_validation, _ = build_samples(historical,
                                        stride=args.policy_stride)
    validation = tuple(torch.cat((new_validation[i], old_validation[i]), dim=0)
                       for i in range(3))
    cfg = SimpleNamespace(tilt_weight=args.tilt_weight,
                          action_weight=args.action_weight,
                          rate_weight=args.rate_weight,
                          tilt_margin_deg=args.tilt_margin_deg,
                          tilt_absolute_ceiling_deg=args.tilt_ceiling_deg,
                          min_pan_gain_deg=args.min_pan_gain_deg)
    initial, initial_accepted, initial_score, threshold = score_candidate(
        model, actor, validation, device, cfg, args)
    best = initial_score if initial_accepted else float('inf')
    best_epoch = 0
    best_weights = {k:v.detach().cpu().clone()
                    for k,v in actor.state_dict().items()}
    best_critic = {k:v.detach().cpu().clone()
                   for k,v in critic.state_dict().items()}
    best_val = initial
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    hist=[]
    gamma, lam = 0.98, 0.95
    for epoch in range(1, args.policy_epochs + 1):
        aloss, closs=[], []
        actor.train(); critic.train()
        for _ in range(args.policy_updates):
            index = torch.randint(len(new_training[0]), (args.batch_size,))
            state, held, ref = [t[index].to(device) for t in new_training]
            for par in critic.parameters():
                par.requires_grad_(False)
            reward, states, _, final = imagined_rollout(
                model, actor, state, held, ref, HORIZON, cfg)
            bootstrap = target(final, held, ref)
            discounts = gamma ** torch.arange(HORIZON, device=device)
            actor_loss = -(torch.sum(reward*discounts[None], dim=1)
                           + gamma**HORIZON*bootstrap).mean()
            if not bool(torch.isfinite(actor_loss)):
                raise RuntimeError('Non-finite actor objective')
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_opt.step()
            for par in critic.parameters():
                par.requires_grad_(True)
            targets = td_lambda_targets(reward.detach(), states.detach(),
                                        final.detach(), held.detach(), ref.detach(),
                                        target, gamma=gamma, lam=lam)
            estimate = critic(states.detach().reshape(-1,20),
                              held[:,None,:].expand(-1,HORIZON,-1).reshape(-1,8),
                              ref[:,None,:].expand(-1,HORIZON,-1).reshape(-1,1)
                              ).reshape(-1,HORIZON)
            critic_loss = nn.functional.smooth_l1_loss(estimate, targets)
            if not bool(torch.isfinite(critic_loss)):
                raise RuntimeError('Non-finite critic loss')
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()
            with torch.no_grad():
                for dest, src in zip(target.parameters(),critic.parameters()):
                    dest.lerp_(src,0.01)
            aloss.append(float(actor_loss.detach()))
            closs.append(float(critic_loss.detach()))
        values, accepted, score, threshold = score_candidate(
            model, actor, validation, device, cfg, args)
        hist.append({'epoch':epoch, 'actor_loss':float(np.mean(aloss)),
                     'critic_loss':float(np.mean(closs)),
                     'accepted':accepted, 'score':score, 'metrics':values})
        # Promotion only on a stricter BOTH-pan-and-tilt acceptance gate.
        if accepted and score + 1e-5 < best:
            best=score
            best_epoch=epoch
            best_weights={k:v.detach().cpu().clone()
                          for k,v in actor.state_dict().items()}
            best_critic={k:v.detach().cpu().clone()
                         for k,v in critic.state_dict().items()}
            best_val=values
        print(f"[ACTOR/CRITIC] epoch {epoch:02d}: "
              f"pan={values['actor_pan_abs_error_deg']:.4f}° "
              f"tilt={values['actor_tilt_delta_deg']:.4f}° "
              f"accepted={accepted} selected_epoch={best_epoch}")

    # Always save latest (research only); promote only gate-accepted best.
    final_ckpt = {
        'model_class':'RAVENOnlineActorCriticV2',
        'actor_state_dict': {k:v.detach().cpu().clone()
                             for k,v in actor.state_dict().items()},
        'critic_state_dict': {k:v.detach().cpu().clone()
                              for k,v in critic.state_dict().items()},
        'joint_names':JOINTS,
        'horizon':HORIZON,
        'model_source':str(model_path),
        'selected_epoch':args.policy_epochs,
        'passed_validation_gate':False,
    }
    torch.save(final_ckpt,out/'actor_online_last_unselected.pt')
    if best_epoch > 0:
        selected_path = out/'actor_online_selected.pt'
        final_ckpt.update({
            'actor_state_dict':best_weights,
            'critic_state_dict':best_critic,
            'selected_epoch':best_epoch,
            'passed_validation_gate':True,
        })
        torch.save(final_ckpt,selected_path)
    else:
        # Existing accepted actor remains in service; training may have failed gate.
        selected_path = actor_path
    return selected_path, {
        'baseline_actor_validation':initial,
        'baseline_passed_gate':initial_accepted,
        'baseline_score':initial_score,
        'selected_epoch':best_epoch,
        'selected_validation':best_val,
        'selected_score':best if math.isfinite(best) else None,
        'tilt_limit_deg':threshold,
        'learning_history':hist,
        'candidate_promoted':best_epoch>0,
        'policy_checkpoint_for_next_cycle':str(selected_path),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cycles',type=int,default=2)
    ap.add_argument('--model',type=Path,default=DEFAULT_MODEL)
    ap.add_argument('--actor',type=Path,default=DEFAULT_ACTOR)
    ap.add_argument('--historical',type=Path,nargs=2,default=DEFAULT_RUNS)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--model-epochs',type=int,default=8)
    ap.add_argument('--model-stride',type=int,default=8)
    ap.add_argument('--policy-epochs',type=int,default=3)
    ap.add_argument('--policy-updates',type=int,default=40)
    ap.add_argument('--policy-stride',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--model-lr',type=float,default=1e-4)
    ap.add_argument('--actor-lr',type=float,default=8e-5)
    ap.add_argument('--critic-lr',type=float,default=1e-4)
    ap.add_argument('--tilt-weight',type=float,default=5.0)
    ap.add_argument('--action-weight',type=float,default=0.08)
    ap.add_argument('--rate-weight',type=float,default=0.2)
    ap.add_argument('--tilt-margin-deg',type=float,default=0.02)
    ap.add_argument('--tilt-ceiling-deg',type=float,default=0.08)
    ap.add_argument('--min-pan-gain-deg',type=float,default=0.002)
    ap.add_argument('--blend',type=float,default=0.25)
    ap.add_argument('--explore-noise-rad',type=float,default=0.0002)
    ap.add_argument('--tracking-duration-s',type=float,default=35.0)
    ap.add_argument('--headless',action='store_true')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    for k in ('cycles','model_epochs','model_stride','policy_epochs',
              'policy_updates','policy_stride','batch_size'):
        if getattr(args,k)<1:
            ap.error(f'--{k.replace("_", "-")} must be positive')
    for path in [args.model,args.actor,*args.historical]:
        if not path.is_file():
            ap.error(f'Missing input: {path}')
    if args.tracking_duration_s < 10:
        ap.error('Use at least 10 seconds of tracking')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    current_model=args.model.resolve()
    current_actor=args.actor.resolve()
    history=[]
    for cycle in range(1,args.cycles+1):
        cycle_out=root/f'cycle_{cycle:02d}'
        cycle_out.mkdir(parents=True,exist_ok=False)
        new_capture_dir=cycle_out/'experience'
        # Alternate trajectories to create genuine state distribution variation.
        amplitude=[0.22,0.35,0.15,0.42][(cycle-1)%4]
        period=[5.0,7.0,9.0,6.0][(cycle-1)%4]
        command=[sys.executable,str(HERE/'raven_supermimic_experience_collect_v2.py'),
                 '--policy',str(current_actor),
                 '--online-out',str(new_capture_dir),
                 '--blend',str(args.blend),
                 '--explore-noise-rad',str(args.explore_noise_rad),
                 '--seed',str(args.seed+cycle),
                 '--actor-amplitude-m',str(amplitude),
                 '--actor-period-s',str(period),
                 '--tracking-duration-s',str(args.tracking_duration_s)]
        if args.headless:
            command.append('--headless')
        print(f'\n=== CYCLE {cycle}/{args.cycles}: NEW SIMULATOR EXPERIENCE ===',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        data=new_capture_dir/'raven_g1_state_action_120hz.jsonl'
        summary=new_capture_dir/'raven_g1_state_action.summary.json'
        if not data.is_file() or not summary.is_file():
            raise RuntimeError('Simulator did not create expected transition replay')
        metadata=json.loads(summary.read_text(encoding='utf-8'))
        if (metadata.get('status')!='RECORDED'
            or int(metadata.get('policy_steps_applied',0))<1):
            raise RuntimeError('No actor-guided physics transitions: refusing learning cycle')
        print(f'\n=== CYCLE {cycle}: UPDATE DYNAMICS FROM NEW PHYSICS ===',flush=True)
        model, next_model, model_report=update_dynamics(
            current_model,list(map(Path,args.historical)),data,cycle_out,
            args,device)
        print(f'\n=== CYCLE {cycle}: ACTOR-CRITIC BACKPROP ===',flush=True)
        next_actor, policy_report=update_actor(
            model,next_model,current_actor,data,list(map(Path,args.historical)),
            cycle_out,args,device)
        entry={
            'cycle':cycle,
            'new_physics_capture':str(data),
            'actor_guided_physics_steps':metadata['policy_steps_applied'],
            'model_update':model_report,
            'policy_update':policy_report,
            'next_actor':str(next_actor),
            'next_model':str(next_model),
        }
        history.append(entry)
        (cycle_out/'learning_metrics.json').write_text(
            json.dumps(entry,indent=2),encoding='utf-8')
        current_model=next_model
        current_actor=next_actor
        (root/'latest_training_summary.json').write_text(json.dumps({
            'online_learning':True,
            'note':'Experimental simulator-in-the-loop model-based learning; not official SuperMimic',
            'cycles_completed':cycle,
            'independent_third_capture_untouched':True,
            'history':history,
        },indent=2),encoding='utf-8')
        print(f'=== CYCLE {cycle} COMPLETE: model and actor updated/selected ===\n',flush=True)
    print('Training cycles complete. Check latest_training_summary.json.')
    print(f'Latest candidate actor: {current_actor}')
    print('Not deployable to a real G1; simulator-only research pipeline.')

if __name__=='__main__':
    main()
