"""
trainer.py — REINFORCE with Learned Baseline
==============================================
Why REINFORCE (not Q-learning / D3QN):
  - The WAoI objective is a SEQUENCE-LEVEL metric. Per-step bootstrapping
    (Q-learning) requires estimating the value of partial trajectories,
    which creates credit assignment problems with the stage-weighted WAoI.
  - REINFORCE uses the FULL episode return directly:
      R = -objective = theta2*sum(wi) - theta1*WAoI
    No reward shaping, no terminal corrections, no potential functions.
  - A learned baseline V(s) reduces variance without biasing the gradient.
  - Entropy regularisation keeps the policy exploratory throughout training,
    avoiding the local-optimum collapse seen in Q-learning approaches.

Training loop:
  For each epoch:
    1. Sample B episode trajectories using current policy
    2. Compute returns R_i = -objective(traj_i)
    3. Compute advantage: A_i = R_i - V_i  (baseline-subtracted)
    4. Policy loss:  -mean(A_i * sum(log_pi))  [REINFORCE]
    5. Value loss:   MSE(V_i, R_i)             [baseline]
    6. Entropy bonus: -beta * mean(H(pi))      [exploration]
    7. Update with Adam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from typing import List

from env import Params, UAVEnv
from policy import AttentionPolicy
from features import batch_rollout, obs_to_tensors


class Trainer:
    def __init__(self, params: Params,
                 d_model=128, n_heads=8, n_layers=3,
                 lr=1e-4, entropy_beta=0.01,
                 episodes_per_epoch=64,
                 device=None):

        self.p      = params
        self.beta   = entropy_beta
        self.B      = episodes_per_epoch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = AttentionPolicy(d_model, n_heads, n_layers).to(self.device)
        self.opt    = torch.optim.Adam(self.policy.parameters(), lr=lr,
                                       weight_decay=1e-5)
        self.sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
                          self.opt, T_max=500, eta_min=1e-5)

        self.history = {
            'reward': [], 'obj': [], 'nodes': [],
            'policy_loss': [], 'value_loss': [], 'entropy': []
        }
        self.best_obj = float('inf')
        self.reward_ema = None

    # ── single epoch ──────────────────────────────────────────────────────
    def epoch(self, epoch_idx: int, rng: np.random.Generator):
        p = self.p
        policy_losses, value_losses, entropies = [], [], []
        rewards, objs, nodes = [], [], []

        for _ in range(self.B):
            seed      = int(rng.integers(0, 10_000_000))
            env       = UAVEnv(p, seed=seed)
            traj, log_probs, values, R = batch_rollout(
                self.policy, env, self.device, greedy=False)

            if not log_probs:   # empty trajectory
                continue

            R_t   = torch.tensor(R, dtype=torch.float32, device=self.device)
            lp    = torch.stack(log_probs)            # (T,)
            vals  = torch.stack(values)               # (T,)

            # ── REINFORCE with baseline ──────────────────────────────────
            # Use last value as episode value estimate (no discounting:
            # all rewards come at episode end)
            V_ep = vals.mean()
            adv  = R_t - V_ep.detach()

            # Entropy of each step's distribution (for regularisation)
            # We re-run forward to get full distributions for entropy
            H_total = self._episode_entropy(env, traj, seed)

            policy_loss = -(adv * lp.sum())
            value_loss  = F.mse_loss(V_ep, R_t)
            entropy_loss= -self.beta * H_total

            loss = policy_loss + value_loss + entropy_loss
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(H_total.item())
            rewards.append(R)
            objs.append(env.objective(traj))
            nodes.append(len(traj))

        self.sched.step()
        return {
            'reward':      float(np.mean(rewards)) if rewards else 0,
            'obj':         float(np.mean(objs)) if objs else 0,
            'nodes':       float(np.mean(nodes)) if nodes else 0,
            'policy_loss': float(np.mean(policy_losses)) if policy_losses else 0,
            'value_loss':  float(np.mean(value_losses)) if value_losses else 0,
            'entropy':     float(np.mean(entropies)) if entropies else 0,
        }

    def _episode_entropy(self, env: UAVEnv, traj: List[int], seed: int):
        """Re-run episode deterministically to collect entropy at each step."""
        env2  = UAVEnv(env.p, seed=seed)
        obs   = env2.reset()
        H_sum = torch.tensor(0.0, device=self.device)
        done  = False
        for action in traj:
            nf, cf, mask = obs_to_tensors(obs, env.p, self.device)
            log_p, _     = self.policy(nf, cf, mask)
            probs        = log_p.exp()
            H_sum        = H_sum - (probs * log_p).sum()
            obs, _, done = env2.step(action)
        return H_sum

    # ── evaluation ────────────────────────────────────────────────────────
    def evaluate(self, n=200, seed=9999) -> dict:
        rng  = np.random.default_rng(seed)
        objs, nodes, waois, prios = [], [], [], []
        self.policy.eval()
        with torch.no_grad():
            for _ in range(n):
                s   = int(rng.integers(0, 10_000_000))
                env = UAVEnv(self.p, seed=s)
                traj, _, _, R = batch_rollout(
                    self.policy, env, self.device, greedy=True)
                objs.append(env.objective(traj))
                nodes.append(len(traj))
                waois.append(env.p.theta1 * env.waoi(traj))
                prios.append(sum(env.wi[j] for j in traj))
        self.policy.train()
        return {
            'obj':      float(np.mean(objs)),
            'nodes':    float(np.mean(nodes)),
            'waoi':     float(np.mean(waois)),
            'priority': float(np.mean(prios)),
            'obj_std':  float(np.std(objs)),
        }

    # ── training entry point ──────────────────────────────────────────────
    def train(self, n_epochs=500, save_path='models(attn)/attention_policy.pt',
              log_every=20, eval_every=50, seed=42):

        rng = np.random.default_rng(seed)

        # Safely create output directories regardless of path structure
        model_dir = os.path.dirname(save_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        os.makedirs('results(attn)', exist_ok=True)

        M = self.p.M
        print(f"\nAttention Policy | M={M} | device={self.device}")
        print(f"Params: {sum(p.numel() for p in self.policy.parameters()):,} parameters")
        print(f"Epochs: {n_epochs} × {self.B} episodes = "
              f"{n_epochs*self.B:,} total episodes\n")
        print(f"{'Epoch':>6} {'Reward':>8} {'Nodes':>7} {'Obj':>8} "
              f"{'π_loss':>9} {'H':>7} {'ε_eval':>8}")
        print('─' * 65)

        reward_win = deque(maxlen=50)

        for ep in range(1, n_epochs + 1):
            stats = self.epoch(ep, rng)

            for k, v in stats.items():
                self.history[k].append(v)
            reward_win.append(stats['reward'])

            if ep % log_every == 0:
                eval_obj = '   —  '
                if ep % eval_every == 0:
                    ev = self.evaluate(n=100)
                    eval_obj = f"{ev['obj']:+.2f}"
                    # Save whenever this evaluation beats the stored best
                    if ev['obj'] < self.best_obj:
                        self.best_obj = ev['obj']
                        self.save(save_path)
                print(f"{ep:6d} {np.mean(reward_win):8.2f} "
                      f"{stats['nodes']:7.1f} {stats['obj']:8.2f} "
                      f"{stats['policy_loss']:9.3f} {stats['entropy']:7.3f} "
                      f"{eval_obj:>8}")

        # ── Persist training history keyed by M so runs don't overwrite ──
        history_path = f"results(attn)/attention_history_M{M}.npy"
        np.save(history_path, self.history)

        # Final full eval — returned to caller; saving is the caller's job
        # (run.py does a final save if this beats best_obj)
        final = self.evaluate(n=200)
        print(f"\n{'─'*65}")
        print(f"Final evaluation (200 instances):")
        print(f"  Obj={final['obj']:.4f}  Nodes={final['nodes']:.1f}  "
              f"WAoI={final['waoi']:.4f}  Priority={final['priority']:.4f}  "
              f"Obj_std={final['obj_std']:.4f}")
        print(f"  Best obj seen during training: {self.best_obj:.4f}")
        print(f"  History saved → {history_path}")
        return final

    def save(self, path):
        torch.save({'policy':   self.policy.state_dict(),
                    'opt':      self.opt.state_dict(),
                    'best_obj': self.best_obj,
                    'M':        self.p.M}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.best_obj = ck.get('best_obj', float('inf'))