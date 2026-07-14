"""
fleet_train_dense.py  —  fixes the #1 policy ceiling: DENSE per-stage reward.
============================================================================
Drop-in replacement trainer for the fleet policy that changes ONLY the reward
signal (and the checkpoint criterion). It reuses your existing MultiUAVPolicy
and make_features_multi unchanged, so it is a minimal, low-risk delta whose
effect is isolable.

WHAT CHANGES vs train_fleet (and why):
  1. DENSE per-stage reward.  Instead of a single terminal scalar backfilled to
     the last step, each commit (UAV k picks node j) earns its exact marginal
     objective change:
         r_t = -( J_after - J_before )
             = -( theta1 * [waoi(chain_k+[j]) - waoi(chain_k)] - theta2 * w_j ).
     This telescopes EXACTLY to the terminal reward (sum_t r_t = -J_fleet),
     verified numerically to 0 error — so it is an unbiased reshaping. It gives
     per-commit credit, which is precisely the closed-form per-stage WAoI credit
     Lemma 1 provides but the terminal-reward trainer discarded. This directly
     targets the measured priority-blindness (priority/node ~ population mean).
  2. Best-checkpoint on a GREEDY held-out eval, not the noisy exploratory
     training return.

Everything else (PPO clip, GAE, entropy, network) is identical to train_fleet.

Run (uav_env active):
    python fleet_train_dense.py --M 100 --K 4 --epochs 300 --seed 42 \
        --save models_multi_uav/fleet_M100_K4_split_seed42_DENSE.pt
Then compare its eval_fleet objective (and priority/node) to the terminal-reward
model on the SAME cell to see if the gap to SA shrinks.
"""

import os, time, argparse
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, MultiUAVPolicy, FleetState, make_features_multi,
                              fleet_rollout, fleet_post_process, eval_fleet, _gae,
                              deconfliction_schedule)


def collect_dense(policy, M, K, n_eps, rng, device, Emax_each, convention='intra'):
    """Roll out n_eps fleet episodes with DENSE per-step marginal rewards."""
    bx, bm, ba, blo, bvo, brew, bend = [], [], [], [], [], [], []
    policy.eval()
    with torch.no_grad():
        for _ in range(n_eps):
            s = int(rng.integers(0, 10_000_000)); env = Env(M=M, seed=s)
            fleet = FleetState(env, K, Emax_each)
            ep = []
            for _ in range(env.M + K + 1):
                if not fleet.any_active():
                    break
                active = [j for j in range(K) if fleet.active[j]]
                if not active:
                    break
                k = max(active, key=lambda j: fleet.E_left[j])
                x, mask, feasible = make_features_multi(fleet, k, device)
                if not feasible.any():
                    fleet.retire(k); continue
                logp, value = policy(x, mask)
                a = int(torch.distributions.Categorical(logits=logp).sample().item())
                # exact marginal objective change for this commit
                w_before = env.waoi(fleet.trajs[k])
                fleet.commit(k, a)
                w_after = env.waoi(fleet.trajs[k])
                dJ = P.theta1 * (w_after - w_before) - P.theta2 * float(env.wi[a])
                r = -dJ
                ep.append([x.cpu().numpy(), mask.cpu().numpy(), a,
                           float(logp[a].item()), float(value.item()), r])
            # common-t0: fold the (terminal) deconfliction penalty into the last reward
            if convention == 'common_t0' and ep and K >= 2:
                pen = float(deconfliction_schedule(fleet, verify=False)['aoi_penalty'])
                ep[-1][5] -= pen   # r_last -= penalty  => sum r = -(intra_J + penalty)
            for idx, (xn, mn, an, lpn, vn, rn) in enumerate(ep):
                bx.append(xn); bm.append(mn); ba.append(an); blo.append(lpn)
                bvo.append(vn); brew.append(rn); bend.append(idx == len(ep) - 1)
    policy.train()
    return (np.array(bx, np.float32), np.array(bm, bool), np.array(ba, np.int64),
            np.array(blo, np.float32), np.array(bvo, np.float32),
            np.array(brew, np.float32), np.array(bend, bool))



def eval_final(policy, M, K, n, device, Emax_each, convention):
    """Deployment metric: greedy rollout (no post-process) objective, plus the
    deconfliction penalty when the convention is common_t0. Same metric for both
    conventions so trained policies are directly comparable."""
    rng = np.random.default_rng(2025); vals = []
    with torch.no_grad():
        for _ in range(n):
            s = int(rng.integers(0, 10_000_000)); env = Env(M=M, seed=s)
            f = fleet_rollout(policy, env, K, device, Emax_each=Emax_each, greedy=True)
            J = f.fleet_objective()
            if K >= 2:   # deployment FINAL metric for BOTH conventions
                J += float(deconfliction_schedule(f, verify=False)['aoi_penalty'])
            vals.append(J)
    return float(np.mean(vals))

def train_fleet_dense(M, K, n_epochs, eps_per_epoch, device, save_path, Emax_each,
                      convention='intra', seed=42, lr=3e-4, entropy_beta=0.02, ppo_epochs=3,
                      ppo_clip=0.15, value_coef=0.5, gae_gamma=1.0, gae_lambda=0.95,
                      minibatch=64, log_every=10, eval_every=25):
    rng = np.random.default_rng(seed)
    policy = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/20)
    print(f'DENSE-reward trainer | M={M} K={K} | device={device}')
    print(f'{"ep":>5} {"trainR":>9} {"pg":>8} {"greedyObj":>10} {"s/ep":>6}')
    best = float('inf'); tw = deque(maxlen=10)
    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        bx, bm, ba, blo, bvo, brew, bend = collect_dense(
            policy, M, K, eps_per_epoch, rng, device, Emax_each, convention)
        T = len(ba)
        if T == 0:
            continue
        adv, ret = _gae(brew, bvo, bend, gae_gamma, gae_lambda)  # per-step rewards now
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv_t = torch.from_numpy(adv).to(device); ret_t = torch.from_numpy(ret).to(device)
        lpo_t = torch.from_numpy(blo).to(device); act_t = torch.from_numpy(ba).to(device)
        pg_log = []; idxs = np.arange(T)
        for _ in range(ppo_epochs):
            np.random.shuffle(idxs)
            for st in range(0, T, minibatch):
                mb = idxs[st:st + minibatch]
                if len(mb) == 0:
                    continue
                lpb, vb, eb = [], [], []
                for i in mb:
                    xi = torch.from_numpy(bx[i]).to(device); mi = torch.from_numpy(bm[i]).to(device)
                    lp_i, v_i = policy(xi, mi)
                    lpb.append(lp_i[act_t[i]]); vb.append(v_i)
                    eb.append(-(lp_i.exp() * lp_i).sum())
                lp_new = torch.stack(lpb); v_new = torch.stack(vb); ent = torch.stack(eb).mean()
                ratio = torch.exp(lp_new - lpo_t[mb]); ab = adv_t[mb]
                pg = -torch.min(ratio * ab, torch.clamp(ratio, 1-ppo_clip, 1+ppo_clip) * ab).mean()
                loss = pg + value_coef * F.mse_loss(v_new, ret_t[mb]) - entropy_beta * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
                pg_log.append(pg.item())
        sched.step(); tw.append(time.perf_counter() - t0)
        trainR = float(np.mean(brew[bend])) if bend.any() else 0.0  # not the objective; diag only
        gstr = '     -   '
        if ep % eval_every == 0:
            fobj = eval_final(policy, M, K, 40, device, Emax_each, convention)
            gstr = f'{fobj:+10.2f}'
            if fobj < best:
                best = fobj
                torch.save({'policy': policy.state_dict(), 'M': M, 'K': K,
                            'input_dim': MP.INPUT_DIM, 'Emax_each': Emax_each,
                            'reward': 'dense', 'convention': convention,
                            'best_final_obj': best}, save_path)
        if ep % log_every == 0 or ep == 1:
            print(f'{ep:>5} {trainR:>9.2f} {np.mean(pg_log):>8.4f} {gstr} {np.mean(tw):>6.1f}')
    print(f'\nBest FINAL (greedy no-pp, routing+deconflict): {best:.3f} -> {save_path}')
    return policy


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--full-battery', action='store_true')
    ap.add_argument('--convention', choices=['intra','common_t0'], default='intra')
    ap.add_argument('--save', default=None)
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Emax_each = (MP.Emax_each if a.full_battery else P.Emax / a.K)
    tag = 'DENSE' if a.convention=='intra' else 'DENSE_CT0'
    save = a.save or f'models_multi_uav/fleet_M{a.M}_K{a.K}_split_seed{a.seed}_{tag}.pt'
    os.makedirs('models_multi_uav', exist_ok=True)
    train_fleet_dense(a.M, a.K, a.epochs, 64, device, save, Emax_each,
                      convention=a.convention, seed=a.seed)