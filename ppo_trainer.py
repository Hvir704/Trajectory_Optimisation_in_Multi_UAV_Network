"""
ppo_trainer.py - Proximal Policy Optimization for the attention policy.

PPO uses the same AttentionPolicy architecture as trainer.py, but updates from
batches of on-policy rollout steps over multiple clipped-objective epochs.
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import Params, UAVEnv
from features import batch_rollout, obs_to_tensors
from policy import AttentionPolicy


class RolloutBuffer:
    def __init__(self):
        self.node_feats = []
        self.ctx_feats = []
        self.masks = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, node_feats, ctx_feats, mask, action, log_prob, reward, value, done):
        self.node_feats.append(node_feats.detach().squeeze(0).cpu())
        self.ctx_feats.append(ctx_feats.detach().squeeze(0).cpu())
        self.masks.append(mask.detach().squeeze(0).cpu())
        self.actions.append(int(action))
        self.old_log_probs.append(float(log_prob.detach().cpu()))
        self.rewards.append(float(reward))
        self.values.append(float(value.detach().cpu()))
        self.dones.append(bool(done))

    def __len__(self):
        return len(self.actions)

    def as_tensors(self, device):
        return {
            'node_feats': torch.stack(self.node_feats).to(device),
            'ctx_feats': torch.stack(self.ctx_feats).to(device),
            'masks': torch.stack(self.masks).to(device),
            'actions': torch.tensor(self.actions, dtype=torch.long, device=device),
            'old_log_probs': torch.tensor(self.old_log_probs, dtype=torch.float32, device=device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'values': torch.tensor(self.values, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
        }


def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(0.0, device=rewards.device)

    for t in reversed(range(len(rewards))):
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    def __init__(self, params: Params,
                 d_model=128, n_heads=8, n_layers=3,
                 lr=1e-4, clip_eps=0.2, entropy_beta=0.01,
                 value_coef=0.5, gamma=1.0, gae_lambda=0.95,
                 ppo_epochs=4, minibatch_size=256,
                 episodes_per_epoch=64, device=None):

        self.p = params
        self.clip_eps = clip_eps
        self.beta = entropy_beta
        self.value_coef = value_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.B = episodes_per_epoch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy = AttentionPolicy(d_model, n_heads, n_layers).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-5)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=500, eta_min=1e-5)

        self.history = {
            'reward': [], 'obj': [], 'nodes': [],
            'policy_loss': [], 'value_loss': [], 'entropy': []
        }
        self.best_obj = float('inf')

    def _collect_episode(self, seed: int, buffer: RolloutBuffer):
        env = UAVEnv(self.p, seed=seed)
        obs = env.reset()
        ep_reward = 0.0
        start_len = len(buffer)

        while True:
            nf, cf, mask = obs_to_tensors(obs, env.p, self.device)
            feasible = (~mask[0]).any().item()

            if not feasible:
                obs, terminal_reward, done = env.step(-1)
                ep_reward = terminal_reward
                if len(buffer.rewards) > start_len and buffer.dones[-1] is False:
                    buffer.rewards[-1] = terminal_reward
                    buffer.dones[-1] = True
                break

            with torch.no_grad():
                log_p, value = self.policy(nf, cf, mask)
                dist = torch.distributions.Categorical(logits=log_p)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)

            action = action_t.item()
            buffer.add(nf, cf, mask, action, log_prob[0], 0.0, value[0], False)
            obs, reward, done = env.step(action)
            if done:
                ep_reward = reward
                buffer.rewards[-1] = reward
                buffer.dones[-1] = True
                break

        return env.traj[:], ep_reward, env.objective(env.traj)

    def _ppo_update(self, buffer: RolloutBuffer):
        data = buffer.as_tensors(self.device)
        advantages, returns = compute_gae(
            data['rewards'], data['values'], data['dones'],
            gamma=self.gamma, lam=self.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        n = len(buffer)
        policy_losses, value_losses, entropies = [], [], []

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.minibatch_size):
                idx = perm[start:start + self.minibatch_size]

                log_p, values = self.policy(
                    data['node_feats'][idx],
                    data['ctx_feats'][idx],
                    data['masks'][idx])
                dist = torch.distributions.Categorical(logits=log_p)
                new_log_probs = dist.log_prob(data['actions'][idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - data['old_log_probs'][idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[idx]

                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, returns[idx])
                loss = policy_loss + self.value_coef * value_loss - self.beta * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.opt.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        self.sched.step()
        return {
            'policy_loss': float(np.mean(policy_losses)) if policy_losses else 0.0,
            'value_loss': float(np.mean(value_losses)) if value_losses else 0.0,
            'entropy': float(np.mean(entropies)) if entropies else 0.0,
        }

    def epoch(self, epoch_idx: int, rng: np.random.Generator):
        buffer = RolloutBuffer()
        rewards, objs, nodes = [], [], []

        for _ in range(self.B):
            seed = int(rng.integers(0, 10_000_000))
            traj, reward, obj = self._collect_episode(seed, buffer)
            if traj:
                rewards.append(reward)
                objs.append(obj)
                nodes.append(len(traj))

        if len(buffer) == 0:
            return {
                'reward': 0.0, 'obj': 0.0, 'nodes': 0.0,
                'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
            }

        update_stats = self._ppo_update(buffer)
        return {
            'reward': float(np.mean(rewards)) if rewards else 0.0,
            'obj': float(np.mean(objs)) if objs else 0.0,
            'nodes': float(np.mean(nodes)) if nodes else 0.0,
            **update_stats,
        }

    def evaluate(self, n=200, seed=9999) -> dict:
        objs, nodes, waois, prios = [], [], [], []
        self.policy.eval()
        with torch.no_grad():
            for s in range(n):
                env = UAVEnv(self.p, seed=s)
                traj, _, _, _ = batch_rollout(
                    self.policy, env, self.device, greedy=True)
                objs.append(env.objective(traj))
                nodes.append(len(traj))
                waois.append(env.p.theta1 * env.waoi(traj))
                prios.append(sum(env.wi[j] for j in traj))
        self.policy.train()
        return {
            'obj': float(np.mean(objs)),
            'nodes': float(np.mean(nodes)),
            'waoi': float(np.mean(waois)),
            'priority': float(np.mean(prios)),
            'obj_std': float(np.std(objs)),
        }

    def train(self, n_epochs=500, save_path=None,
              log_every=20, eval_every=50, seed=4242):

        rng = np.random.default_rng(seed)
        if save_path is None:
            save_path = f'models_attn/attn_M{self.p.M}.pt'
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        os.makedirs('results(attn)', exist_ok=True)

        print(f"\nPPO Attention Policy | M={self.p.M} | device={self.device}")
        print(f"Params: {sum(p.numel() for p in self.policy.parameters()):,} parameters")
        print(f"Epochs: {n_epochs} x {self.B} episodes, {self.ppo_epochs} PPO epochs/update\n")
        print(f"{'Epoch':>6} {'Reward':>8} {'Nodes':>7} {'Obj':>8} "
              f"{'pi_loss':>9} {'H':>7} {'eval':>8}")
        print('-' * 65)

        reward_win = deque(maxlen=50)

        for ep in range(1, n_epochs + 1):
            stats = self.epoch(ep, rng)

            for k, v in stats.items():
                self.history[k].append(v)
            reward_win.append(stats['reward'])

            if ep % log_every == 0:
                eval_obj = '   -  '
                if ep % eval_every == 0:
                    ev = self.evaluate(n=100)
                    eval_obj = f"{ev['obj']:+.2f}"
                    if ev['obj'] < self.best_obj:
                        self.best_obj = ev['obj']
                        self.save(save_path)
                print(f"{ep:6d} {np.mean(reward_win):8.2f} "
                      f"{stats['nodes']:7.1f} {stats['obj']:8.2f} "
                      f"{stats['policy_loss']:9.3f} {stats['entropy']:7.3f} "
                      f"{eval_obj:>8}")

        final = self.evaluate(n=200)
        if self.best_obj == float('inf') or final['obj'] < self.best_obj:
            self.best_obj = final['obj']
            self.save(save_path)

        print(f"\n{'-'*65}")
        print("PPO final evaluation (200 instances):")
        print(f"  Obj={final['obj']:.2f}  Nodes={final['nodes']:.1f}  "
              f"WAoI={final['waoi']:.1f}  Priority={final['priority']:.1f}")
        np.save(f"results(attn)/ppo_history_M{self.p.M}.npy", self.history)
        return final

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'opt': self.opt.state_dict(),
            'best_obj': self.best_obj,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck['policy'])
        self.best_obj = ck.get('best_obj', float('inf'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--M_list', type=int, nargs='+', default=[8, 10, 12, 15])
    parser.add_argument('--epochs', type=int, default=500)
    args = parser.parse_args()

    for M in args.M_list:
        print(f"\n{'='*50}\n  Starting PPO Training for M={M}\n{'='*50}")
        p = Params(M=M)
        trainer = PPOTrainer(p)
        trainer.train(n_epochs=args.epochs)