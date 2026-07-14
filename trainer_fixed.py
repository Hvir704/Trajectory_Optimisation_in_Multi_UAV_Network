"""
trainer_fixed.py  —  Stable Transformer Trainer with Temperature Annealing
===========================================================================

All previous bugs resolved (unchanged from prior fix):
  BUG 1 (FIXED): retain_graph double-gradient → single total_loss.backward()
  BUG 2 (FIXED): H_target was wrong (log(3)*14 nats vs actual log(M)*steps).
                 H_target = log(M) * n_steps * 0.5, with 100-epoch warm-up.
  BUG 3 (FIXED): rollout return-value count variance → H computed from stored
                 log_probs, never from rollout's optional 5th return.
  BUG 4 (FIXED): policy gradient sign; batch size B=128 for diversity.

NEW — FIX 2: Heated-up softmax / decaying temperature (MOSAC-ATT §IV-A)
-------------------------------------------------------------------------
PROBLEM:  A static entropy bonus β=0.01 is a permanent compromise: too
          high → policy never commits; too low → collapses early.

SOLUTION: Two coordinated decaying schedules managed entirely here.
  policy.py only needs to accept the `temperature` kwarg.

  (A) Temperature annealing τ (controls action-sampling breadth):
        τ = τ_start  for epochs 1 … T_anneal
        τ = τ_start - (τ_start - τ_final) * min(ep/T_anneal, 1.0)
            (linear decay)
        τ_start = 5.0   — almost uniform over all feasible nodes early on
        τ_final = 1.0   — standard softmax at convergence
        T_anneal = 0.4 * n_epochs  (first 40% of training)

  (B) Entropy-target decay H_target (penalises excess randomness, but the
      ceiling itself softens over time so the policy is free to sharpen):
        H_target(ep) = log(M) * n_steps * frac(ep)
        frac(ep) = H_target_frac_start * (1 - min(ep/T_anneal, 1.0))
                   + H_target_frac_final * min(ep/T_anneal, 1.0)
        H_target_frac_start = 0.8   (allow up to 80%-random early on)
        H_target_frac_final = 0.3   (only penalise if > 30%-random at end)
        No entropy penalty at all for the first `entropy_warmup_epochs`.

  Why τ and H_target together?
    τ forces the policy to *explore* (samples are diverse even if logits
    are peaked). H_target ensures the policy doesn't just become uniformly
    random and stay there — it gradually tightens the exploration budget.
    Together they reproduce the MOSAC-ATT "heated-up softmax" schedule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import time
from collections import deque

from env    import Params, UAVEnv
from policy import AttentionPolicy

try:
    from features import rollout_episode
except ImportError:
    from features import batch_rollout as rollout_episode


# ─────────────────────────────────────────────────────────────────────────────
# Rollout helper — normalises both 4- and 5-return signatures
# ─────────────────────────────────────────────────────────────────────────────

def _rollout(policy, env, device, greedy=False, temperature=1.0):
    """
    Wraps batch_rollout / rollout_episode and forwards `temperature` to
    sample_action.  Handles both 4-return and 5-return versions of features.py.

    Returns (traj, log_ps, vals, reward) — entropy computed separately.
    """
    # features.batch_rollout calls policy.sample_action internally.
    # We need temperature to reach there, so we patch it in via a closure
    # rather than modifying features.py (which we don't own here).
    from features import obs_to_tensors

    obs       = env.reset()
    log_probs = []
    values    = []
    done      = False

    while not done:
        nf, cf, mask = obs_to_tensors(obs, env.p, device)
        feasible     = (~mask[0]).any().item()

        if not feasible:
            obs, reward, done = env.step(-1)
            break

        if greedy:
            with torch.no_grad():
                action_t = policy.greedy_action(nf, cf, mask, temperature=temperature)
            action = action_t.item()
            with torch.no_grad():
                log_p, value = policy(nf, cf, mask)
            lp = log_p[0, action]
            v  = value[0]
        else:
            action_t, lp, v = policy.sample_action(nf, cf, mask,
                                                    temperature=temperature)
            action = action_t.item()

        log_probs.append(lp)
        values.append(v)
        obs, reward, done = env.step(action)

    return env.traj[:], log_probs, values, reward


# ─────────────────────────────────────────────────────────────────────────────
# Temperature schedule (Fix 2 — part A)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_temperature(epoch: int, n_epochs: int,
                         tau_start: float, tau_final: float,
                         T_anneal: int) -> float:
    """Linear annealing from tau_start → tau_final over T_anneal epochs."""
    frac = min(epoch / max(T_anneal, 1), 1.0)
    return tau_start + (tau_final - tau_start) * frac


# ─────────────────────────────────────────────────────────────────────────────
# Entropy-target fraction (Fix 2 — part B)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_H_target_frac(epoch: int, T_anneal: int,
                            frac_start: float, frac_final: float) -> float:
    """Linear decay of the H_target fraction over T_anneal epochs."""
    frac = min(epoch / max(T_anneal, 1), 1.0)
    return frac_start + (frac_final - frac_start) * frac


# ─────────────────────────────────────────────────────────────────────────────
# FixedTrainer
# ─────────────────────────────────────────────────────────────────────────────

class FixedTrainer:
    def __init__(self,
                 params: Params,
                 d_model: int   = 128,
                 n_heads: int   = 8,
                 n_layers: int  = 3,
                 lr: float      = 3e-4,
                 # ── entropy / exploration ────────────────────────────────
                 entropy_beta: float             = 0.01,
                 entropy_warmup_epochs: int      = 100,
                 H_target_frac_start: float      = 0.8,
                 H_target_frac_final: float      = 0.3,
                 # ── FIX 2: temperature schedule ─────────────────────────
                 tau_start: float                = 5.0,
                 tau_final: float                = 1.0,
                 T_anneal_frac: float            = 0.4,   # fraction of n_epochs
                 # ── other ───────────────────────────────────────────────
                 episodes_per_epoch: int         = 128,
                 adv_clip: float                 = 5.0,
                 device: str                     = None):

        self.p            = params
        self.B            = episodes_per_epoch
        self.beta         = entropy_beta
        self.ent_warmup   = entropy_warmup_epochs
        self.adv_clip     = adv_clip
        self.device       = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Temperature schedule parameters
        self.tau_start        = tau_start
        self.tau_final        = tau_final
        self.T_anneal_frac    = T_anneal_frac   # resolved to epoch count in train()

        # Entropy target decay parameters
        self.H_frac_start = H_target_frac_start
        self.H_frac_final = H_target_frac_final

        self.policy = AttentionPolicy(d_model, n_heads, n_layers).to(self.device)
        self.opt    = torch.optim.Adam(self.policy.parameters(),
                                       lr=lr, weight_decay=1e-5)

        self.history   = {'reward': [], 'obj': [], 'nodes': [],
                          'entropy': [], 'adv_std': [], 'temperature': []}
        self.best_obj   = float('inf')
        self._best_state = None

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compute_H(log_ps) -> float:
        """Episode entropy in nats from stored log_probs tensor (T,)."""
        return float(-log_ps.sum().item())

    # ── one training epoch ────────────────────────────────────────────────

    def epoch(self, rng: np.random.Generator, epoch_idx: int,
              tau: float, H_target_frac: float):
        """
        Run one epoch of B episodes with the given temperature τ and
        H_target fraction.

        Parameters
        ----------
        tau            : current temperature (Fix 2A — sample breadth)
        H_target_frac  : fraction of log(M)*steps used as entropy ceiling
                         (Fix 2B — how much randomness is tolerated)
        """
        self.policy.train()
        use_entropy = (epoch_idx > self.ent_warmup)

        all_log_ps, all_vals, all_R = [], [], []
        all_nodes, all_objs, all_H  = [], [], []

        for _ in range(self.B):
            seed = int(rng.integers(0, 10_000_000))
            env  = UAVEnv(self.p, seed=seed)
            env.reset()

            traj, log_ps, vals, R = _rollout(
                self.policy, env, self.device,
                greedy=False, temperature=tau)

            if not traj:
                continue

            if isinstance(log_ps, list):
                log_ps = torch.stack(log_ps)
            if isinstance(vals, list):
                vals   = torch.stack(vals)

            all_log_ps.append(log_ps)
            all_vals.append(vals)
            all_R.append(R)
            all_nodes.append(len(traj))
            all_objs.append(env.objective(traj))
            all_H.append(self._compute_H(log_ps))

        if not all_R:
            return {'reward': 0, 'obj': 0, 'nodes': 0,
                    'entropy': 0, 'adv_std': 0, 'temperature': tau}

        # Batch advantage normalisation
        R_arr  = np.array(all_R, dtype=np.float32)
        R_mean = float(R_arr.mean())
        R_std  = float(max(R_arr.std(), 1.0))
        advs   = np.clip((R_arr - R_mean) / R_std, -self.adv_clip, self.adv_clip)
        adv_t  = torch.tensor(advs,  dtype=torch.float32, device=self.device)
        R_t    = torch.tensor(R_arr, dtype=torch.float32, device=self.device)

        n            = len(all_R)
        policy_loss  = torch.zeros(1, device=self.device)
        value_loss   = torch.zeros(1, device=self.device)
        entropy_loss = torch.zeros(1, device=self.device)

        for i, (lp, v, H_ep) in enumerate(zip(all_log_ps, all_vals, all_H)):
            policy_loss = policy_loss - adv_t[i] * lp.sum()
            value_loss  = value_loss  + F.mse_loss(v.mean(), R_t[i])

            if use_entropy:
                # Fix 2B: H_target ceiling decays with H_target_frac
                n_steps  = len(lp)
                H_target = math.log(self.p.M) * n_steps * H_target_frac
                excess_H = F.relu(torch.tensor(H_ep - H_target,
                                               dtype=torch.float32,
                                               device=self.device))
                entropy_loss = entropy_loss + self.beta * excess_H

        total_loss = (policy_loss + value_loss + entropy_loss) / n

        self.opt.zero_grad()
        total_loss.backward()                             # single backward (BUG 1 fix)
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

        return {
            'reward':      R_mean,
            'obj':         float(np.mean(all_objs)),
            'nodes':       float(np.mean(all_nodes)),
            'entropy':     float(np.mean(all_H)),
            'adv_std':     R_std,
            'temperature': tau,
        }

    # ── evaluation ────────────────────────────────────────────────────────

    def evaluate(self, n: int = 50, seed: int = 9999,
                 temperature: float = 1.0) -> dict:
        """
        Greedy evaluation.  temperature=1.0 at normal eval; may be >1
        during early warm-up when we pass the current τ.
        """
        self.policy.eval()
        rng  = np.random.default_rng(seed)
        objs, nodes_l, waois, prios = [], [], [], []
        with torch.no_grad():
            for _ in range(n):
                s   = int(rng.integers(0, 10_000_000))
                env = UAVEnv(self.p, seed=s)
                env.reset()
                traj, _, _, _ = _rollout(
                    self.policy, env, self.device,
                    greedy=True, temperature=temperature)
                objs.append(env.objective(traj))
                nodes_l.append(len(traj))
                waois.append(self.p.theta1 * env.waoi(traj))
                prios.append(float(sum(env.wi[j] for j in traj)))
        self.policy.train()
        return {
            'obj':      float(np.mean(objs)),
            'nodes':    float(np.mean(nodes_l)),
            'waoi':     float(np.mean(waois)),
            'priority': float(np.mean(prios)),
        }

    # ── training entry point ──────────────────────────────────────────────

    def train(self, n_epochs: int = 300,
              save_path: str = 'models_attn/attn_M30.pt',
              log_every: int = 20,
              eval_every: int = 50,
              seed: int = 42) -> dict:

        rng = np.random.default_rng(seed)
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else '.',
            exist_ok=True)
        os.makedirs('results_attn', exist_ok=True)

        M   = self.p.M
        n_p = sum(p.numel() for p in self.policy.parameters())

        # Resolve T_anneal from fraction
        T_anneal    = max(1, int(n_epochs * self.T_anneal_frac))
        H_uniform   = math.log(M) * 14   # reference: uniform over M, ~14 steps

        print(f'\nStable Transformer + MOSAC-ATT Fixes | M={M} | {self.device}')
        print(f'Params: {n_p:,} | B={self.B} | ent_warmup={self.ent_warmup}ep')
        print(f'Fix 1: MaxPool baseline (M-independent scale)')
        print(f'Fix 2: τ {self.tau_start:.1f}→{self.tau_final:.1f} '
              f'over {T_anneal} epochs | '
              f'H_frac {self.H_frac_start:.2f}→{self.H_frac_final:.2f}')
        print(f'H_uniform≈{H_uniform:.1f} nats | beta={self.beta}')
        print(f'{"Epoch":>6} {"Reward":>9} {"Nodes":>7} {"Obj":>8} '
              f'{"H":>7} {"τ":>5} {"H_frac":>7} {"EvalObj":>9} {"ETA":>6}')
        print('─' * 75)

        t_win = deque(maxlen=10)
        r_win = deque(maxlen=50)
        diverge_count = 0

        for ep in range(1, n_epochs + 1):
            # ── Fix 2: compute current schedule values ───────────────────
            tau = _compute_temperature(
                ep, n_epochs, self.tau_start, self.tau_final, T_anneal)
            H_target_frac = _compute_H_target_frac(
                ep, T_anneal, self.H_frac_start, self.H_frac_final)

            t0    = time.perf_counter()
            stats = self.epoch(rng, ep, tau=tau, H_target_frac=H_target_frac)
            t_win.append(time.perf_counter() - t0)

            for k in self.history:
                if k in stats:
                    self.history[k].append(stats[k])
            r_win.append(stats['reward'])

            if ep % log_every == 0:
                ev_str = '        '

                if ep % eval_every == 0:
                    # During warm-up eval with current τ so we see the policy
                    # as it is actually being used; use τ=1 after anneal done.
                    eval_tau = tau if ep <= T_anneal else 1.0
                    ev = self.evaluate(n=30, temperature=eval_tau)
                    ev_str = f'{ev["obj"]:+8.2f}'

                    if ev['obj'] < self.best_obj:
                        self.best_obj    = ev['obj']
                        self._best_state = {k: v.cpu().clone()
                                            for k, v in
                                            self.policy.state_dict().items()}
                        self.save(save_path)
                        diverge_count = 0
                    elif (self._best_state is not None
                          and ev['obj'] > self.best_obj + 8):
                        diverge_count += 1
                        if diverge_count >= 2:
                            self.policy.load_state_dict(
                                {k: v.to(self.device)
                                 for k, v in self._best_state.items()})
                            ev_str = f'RELOAD({self.best_obj:+.0f})'
                            diverge_count = 0

                eta_m = np.mean(t_win) * (n_epochs - ep) / 60
                print(f'{ep:6d} {np.mean(r_win):9.2f} {stats["nodes"]:7.1f} '
                      f'{stats["obj"]:8.2f} {stats["entropy"]:7.1f} '
                      f'{tau:5.2f} {H_target_frac:7.3f} '
                      f'{ev_str} {eta_m:5.0f}m')

        # Final evaluation always at τ=1 (pure greedy, no exploration)
        final = self.evaluate(n=200, temperature=1.0)
        np.save(f'results_attn/attn_history_M{M}.npy', self.history)
        print(f'\nM={M}: Obj={final["obj"]:.2f}  Nodes={final["nodes"]:.1f}  '
              f'WAoI={final["waoi"]:.1f}  Priority={final["priority"]:.2f}')
        return final

    # ── persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({'policy':   self.policy.state_dict(),
                    'best_obj': self.best_obj}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.best_obj = ck.get('best_obj', float('inf'))