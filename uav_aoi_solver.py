"""
uav_aoi_solver.py  —  UAV AoI Priority-Aware Trajectory Optimisation  (v2)
============================================================================
Improvements over v1:
  1. Richer 16-dim features: adds marginal WAoI cost, priority/WAoI ratio,
     energy slack, and current WAoI rate — directly encoding Lemma 1 structure.
  2. PPO-style training: multi-epoch updates over a rollout buffer with
     clipped surrogate objective and GAE advantages. More stable than REINFORCE.
  3. Beam search at test time: width-5 search over trajectories (no retraining).
  4. 2-opt local search: post-hoc ordering improvement (test time only).
  5. Node insertion: greedily inserts skipped feasible nodes after 2-opt.

Run:
    python uav_aoi_solver.py            # full run
    python uav_aoi_solver.py --quick    # fast demo
"""

import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results',    exist_ok=True)
os.makedirs('models',     exist_ok=True)
os.makedirs('models_mlp', exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROBLEM PARAMETERS  (paper Section 3)
# ══════════════════════════════════════════════════════════════════════════════
class P:
    M       = 30
    area    = 1000.0       # m
    H       = 100.0        # m altitude
    v       = 20.0         # m/s cruise speed
    Ph      = 200.0        # W hover power
    Pf      = 150.0        # W cruise power
    Emax    = 50_000.0     # J battery
    W_bw    = 1e6          # Hz
    Ps      = 0.1          # W sensor TX
    kappa0  = 1e-3         # ref channel gain
    sigma2  = 1e-14        # noise power
    Di_lo   = 0.5e6        # bits min
    Di_hi   = 5.0e6        # bits max
    wi_lo   = 1.0
    wi_hi   = 10.0
    theta1  = 0.01
    theta2  = 1.0
    home    = np.array([500., 500.])

    # derived
    R = W_bw * np.log2(1 + kappa0 * Ps / (H ** 2 * sigma2))   # uniform rate


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
class Env:
    def __init__(self, M=None, seed=None):
        self.M   = M or P.M
        rng      = np.random.default_rng(seed)
        self.pos = rng.uniform(0, P.area, (self.M, 2)).astype(np.float32)
        self.wi  = rng.uniform(P.wi_lo, P.wi_hi, self.M).astype(np.float32)
        self.Di  = rng.uniform(P.Di_lo, P.Di_hi, self.M).astype(np.float32)
        self.tcd = (self.Di / P.R).astype(np.float32)

    def dist(self, a, b):  return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    def tf(self, a, b):    return self.dist(a, b) / P.v
    def e_segment(self, a, b, j):
        return P.Pf * self.tf(a, self.pos[j]) + P.Ph * self.tcd[j] + P.Pf * self.tf(self.pos[j], b)

    def feasible_mask(self, curr, E_left, visited):
        d_to   = np.linalg.norm(self.pos - curr,   axis=1)
        d_home = np.linalg.norm(self.pos - P.home, axis=1)
        e_need = P.Pf * (d_to + d_home) / P.v + P.Ph * self.tcd
        return (~visited) & (e_need <= E_left)

    def waoi(self, traj):
        if not traj: return 0.0
        W = 0.0; val = 0.0
        for k, j in enumerate(traj):
            W   += self.wi[j]
            nxt  = self.pos[traj[k+1]] if k < len(traj)-1 else P.home
            val += W * (self.tcd[j] + self.tf(self.pos[j], nxt))
        return float(val)

    def objective(self, traj):
        return P.theta1 * self.waoi(traj) - P.theta2 * float(sum(self.wi[j] for j in traj))

    def reward(self, traj):
        return -self.objective(traj)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  POLICY NETWORK  (16-dim input — improved features)
# ══════════════════════════════════════════════════════════════════════════════
class Policy(nn.Module):
    """
    Per-step scores every feasible node with a shared MLP.

    Node features (10 dims):
      0  x / area
      1  y / area
      2  w / wi_hi
      3  tcd / tcd_max
      4  dist_to_curr / area
      5  dist_to_home / area
      6  feasible flag
      7  marginal_waoi_j  (Lemma 1 cost of visiting j next)  [NEW]
      8  priority/waoi ratio                                  [NEW]
      9  energy slack after visiting j and returning home     [NEW]

    Context features (6 dims):
      0  curr_x / area
      1  curr_y / area
      2  E_left / Emax
      3  W_cum / (M * wi_hi)
      4  n_visited / M
      5  waoi_rate = theta1 * W_cum / (M * wi_hi)            [NEW]

    Total: 16 dims per node.
    ~100K parameters.
    """
    def __init__(self, hidden=256, input_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.score_head = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        """
        x    : (M, input_dim) float  — per-node feature vectors
        mask : (M,) bool             — True = invalid (skip)
        Returns log_probs (M,), value (scalar)
        """
        h      = self.encoder(x)                # (M, hidden)
        scores = self.score_head(h).squeeze(-1)  # (M,)
        scores = scores.masked_fill(mask, -1e9)
        log_p  = F.log_softmax(scores, dim=0)

        valid_h = h[~mask]
        value   = self.value_head(valid_h.mean(0)).squeeze() if valid_h.shape[0] > 0 \
                  else torch.zeros(1, device=x.device).squeeze()
        return log_p, value


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FEATURE EXTRACTION  (16-dim, vectorised)
# ══════════════════════════════════════════════════════════════════════════════
def make_features(env: Env, curr: np.ndarray, E_left: float,
                  W_cum: float, n_vis: int,
                  visited: np.ndarray, device: str):
    """
    Build (M, 16) feature tensor and (M,) invalid mask.
    Improvement 1: adds marginal WAoI, priority/WAoI ratio, energy slack,
    and WAoI rate to give the policy direct access to Lemma 1 structure.
    """
    M = env.M

    # ── distances (vectorised) ──────────────────────────────────────────────
    diff_curr = env.pos - curr
    diff_home = env.pos - P.home
    d_curr_m  = np.linalg.norm(diff_curr, axis=1)   # metres
    d_home_m  = np.linalg.norm(diff_home, axis=1)   # metres
    d_curr    = d_curr_m / P.area                    # normalised
    d_home    = d_home_m / P.area

    # ── feasibility ─────────────────────────────────────────────────────────
    e_need = P.Pf * (d_curr_m + d_home_m) / P.v + P.Ph * env.tcd
    feasible = (~visited) & (e_need <= E_left)

    # ── NEW: marginal WAoI cost of visiting node j next (Lemma 1) ──────────
    # ΔWAoI_j = W_cum * tf_curr→j  +  (W_cum + w_j) * tcd_j
    tf_curr_j   = d_curr_m / P.v                              # (M,)
    marginal_w  = W_cum * tf_curr_j + (W_cum + env.wi) * env.tcd  # (M,)
    tcd_max     = float(P.Di_hi / P.R)
    norm_mw     = float(M * P.wi_hi * tcd_max)
    marginal_w_n = marginal_w / (norm_mw + 1e-9)              # normalised

    # ── NEW: priority-to-WAoI ratio ─────────────────────────────────────────
    ratio = P.theta2 * env.wi / (P.theta1 * marginal_w + 1e-6)
    ratio_n = np.clip(ratio / 20.0, 0.0, 1.0)                 # clip [0,20]→[0,1]

    # ── NEW: energy slack after visiting j and returning ────────────────────
    slack = (E_left - e_need) / P.Emax                        # (M,)
    slack = np.clip(slack, -1.0, 1.0)                         # can be <0 if infeasible

    # ── node feature matrix (M, 10) ─────────────────────────────────────────
    node_f = np.stack([
        env.pos[:, 0] / P.area,      # 0
        env.pos[:, 1] / P.area,      # 1
        env.wi / P.wi_hi,             # 2
        env.tcd / tcd_max,            # 3
        d_curr,                       # 4
        d_home,                       # 5
        feasible.astype(np.float32),  # 6
        marginal_w_n.astype(np.float32),  # 7  [NEW]
        ratio_n.astype(np.float32),       # 8  [NEW]
        slack.astype(np.float32),         # 9  [NEW]
    ], axis=1)   # (M, 10)

    # ── NEW: context (6 dims) ───────────────────────────────────────────────
    waoi_rate = P.theta1 * W_cum / (M * P.wi_hi + 1e-9)
    ctx = np.array([
        curr[0] / P.area,              # 0
        curr[1] / P.area,              # 1
        E_left / P.Emax,              # 2
        W_cum / (M * P.wi_hi),        # 3
        n_vis / M,                     # 4
        float(waoi_rate),              # 5  [NEW]
    ], dtype=np.float32)
    ctx_rep = np.broadcast_to(ctx, (M, 6))   # (M, 6)

    x    = np.concatenate([node_f, ctx_rep], axis=1).astype(np.float32)  # (M, 16)
    mask = ~feasible  # True = invalid

    x_t    = torch.from_numpy(x   ).to(device)
    mask_t = torch.from_numpy(mask).to(device)
    return x_t, mask_t, feasible


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EPISODE ROLLOUT  (stochastic — used during training)
# ══════════════════════════════════════════════════════════════════════════════
def rollout(policy: Policy, env: Env, device: str, greedy: bool = False):
    """
    Roll out one episode stochastically (or greedily if greedy=True).
    Returns traj, log_ps, values, entropy, reward.
    Used during PPO buffer collection and by optimality_gap.py.
    """
    visited   = np.zeros(env.M, dtype=bool)
    curr      = P.home.copy()
    E_left    = P.Emax
    W_cum     = 0.0
    traj      = []
    log_ps    = []
    values    = []
    entropies = []

    for _ in range(env.M):
        x, mask, feasible = make_features(env, curr, E_left, W_cum,
                                           int(visited.sum()), visited, device)
        if not feasible.any():
            break

        log_p, value = policy(x, mask)

        if greedy:
            action = int(log_p.argmax().item())
        else:
            dist   = torch.distributions.Categorical(logits=log_p)
            action = int(dist.sample().item())

        lp = log_p[action]
        H  = -(log_p.exp() * log_p).sum()

        log_ps.append(lp)
        values.append(value)
        entropies.append(H)

        tf_to   = float(np.linalg.norm(env.pos[action] - curr)) / P.v
        E_left -= P.Pf * tf_to + P.Ph * env.tcd[action]
        curr    = env.pos[action].copy()
        visited[action] = True
        W_cum  += env.wi[action]
        traj.append(action)

    R = env.reward(traj)

    if not log_ps:
        z = torch.zeros(1, device=device, requires_grad=True)
        return traj, z, z, z.squeeze(), R

    return (traj,
            torch.stack(log_ps),
            torch.stack(values),
            torch.stack(entropies).sum(),
            R)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  TEST-TIME IMPROVEMENTS
# ══════════════════════════════════════════════════════════════════════════════

def rollout_beam(policy: Policy, env: Env, device: str, beam_width: int = 5):
    """
    Improvement 3 — Beam search rollout (test time only, no gradients).
    Maintains top-k partial trajectories by cumulative log-probability,
    then returns the trajectory with the best objective among all beams.
    """
    # Each beam: (log_prob_sum, traj, curr, E_left, W_cum, visited_array)
    init_visited = np.zeros(env.M, dtype=bool)
    beams = [(0.0, [], P.home.copy(), P.Emax, 0.0, init_visited.copy())]
    completed = []

    policy.eval()
    with torch.no_grad():
        for _ in range(env.M):
            if not beams:
                break
            candidates = []
            all_terminal = True

            for lp_sum, traj, curr, E_left, W_cum, visited in beams:
                x, mask, feasible = make_features(env, curr, E_left, W_cum,
                                                   int(visited.sum()), visited, device)
                if not feasible.any():
                    completed.append(traj)
                    continue

                all_terminal = False
                log_p, _ = policy(x, mask)
                lp_np = log_p.cpu().numpy()

                # Expand top beam_width feasible actions
                feasible_idx = np.where(feasible)[0]
                lp_feasible  = lp_np[feasible_idx]
                top_k_local  = np.argsort(lp_feasible)[-beam_width:]

                for ki in top_k_local:
                    j   = int(feasible_idx[ki])
                    new_visited = visited.copy()
                    new_visited[j] = True
                    tf_j  = float(np.linalg.norm(env.pos[j] - curr)) / P.v
                    e_j   = P.Pf * tf_j + P.Ph * env.tcd[j]
                    candidates.append((
                        lp_sum + float(lp_np[j]),
                        traj + [j],
                        env.pos[j].copy(),
                        E_left - e_j,
                        W_cum + env.wi[j],
                        new_visited
                    ))

            if all_terminal:
                break

            # Keep top beam_width by log_prob_sum
            candidates.sort(key=lambda b: -b[0])
            beams = candidates[:beam_width]

    # Collect all trajectories (beams + completed)
    all_trajs = [b[1] for b in beams] + completed
    if not all_trajs:
        return []

    # Return trajectory with best (lowest) objective
    return min(all_trajs, key=lambda t: env.objective(t) if t else float('inf'))


def _energy_cost(env: Env, traj: list) -> float:
    """Total propulsion energy for a trajectory (including return to home)."""
    E = 0.0
    prev = P.home
    for j in traj:
        E += P.Pf * float(np.linalg.norm(env.pos[j] - prev)) / P.v + P.Ph * env.tcd[j]
        prev = env.pos[j]
    E += P.Pf * float(np.linalg.norm(prev - P.home)) / P.v
    return E


def two_opt_improve(env: Env, traj: list) -> list:
    """
    Improvement 4 — 2-opt local search (test time only).
    Repeatedly reverses sub-segments until no improvement found.
    Effective for fixing crossing paths, especially at small M.
    """
    if len(traj) < 3:
        return traj

    best = list(traj)
    improved = True
    while improved:
        improved = False
        best_obj = env.objective(best)
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if env.objective(candidate) < best_obj - 1e-6:
                    best = candidate
                    best_obj = env.objective(candidate)
                    improved = True
                    break
            if improved:
                break
    return best


def try_insert_nodes(env: Env, traj: list) -> list:
    """
    Improvement 5 — cheapest insertion of skipped feasible nodes (test time only).
    After the policy terminates, greedily tries inserting each unvisited node
    at the best position in the route. Repeats until no beneficial insertion exists.
    Directly addresses the under-visitation gap revealed by optimality analysis.
    """
    visited = set(traj)
    changed = True

    while changed:
        changed = False
        curr_obj = env.objective(traj)
        best_gain      = 0.0
        best_candidate = None

        for j in range(env.M):
            if j in visited:
                continue
            for pos in range(len(traj) + 1):
                candidate = traj[:pos] + [j] + traj[pos:]
                # Energy feasibility check
                if _energy_cost(env, candidate) > P.Emax:
                    continue
                gain = curr_obj - env.objective(candidate)
                if gain > best_gain + 1e-6:
                    best_gain      = gain
                    best_candidate = (candidate, j)

        if best_candidate is not None:
            traj, j = best_candidate
            visited.add(j)
            changed = True

    return traj


def post_process(env: Env, traj: list) -> list:
    """
    Full test-time post-processing pipeline:
      1. 2-opt reordering
      2. Node insertion
    Call this on any trajectory before reporting its objective.
    """
    traj = two_opt_improve(env, traj)
    traj = try_insert_nodes(env, traj)
    return traj


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BASELINES
# ══════════════════════════════════════════════════════════════════════════════
def run_baseline(env: Env, method: str) -> list:
    visited = np.zeros(env.M, dtype=bool)
    curr    = P.home.copy()
    E_left  = P.Emax
    traj    = []

    for _ in range(env.M):
        fm = env.feasible_mask(curr, E_left, visited)
        if not fm.any():
            break

        if method == 'random':
            j = int(np.random.choice(np.where(fm)[0]))
        elif method == 'greedy_priority':
            j = int(np.where(fm)[0][np.argmax(env.wi[fm])])
        elif method == 'nearest_neighbor':
            dists = np.linalg.norm(env.pos - curr, axis=1)
            dists[~fm] = np.inf
            j = int(np.argmin(dists))
        elif method == 'pdr':
            dists  = np.maximum(np.linalg.norm(env.pos - curr, axis=1), 1e-9)
            scores = env.wi / dists
            scores[~fm] = -np.inf
            j = int(np.argmax(scores))
        else:
            raise ValueError(method)

        tf_to  = float(np.linalg.norm(env.pos[j] - curr)) / P.v
        E_left -= P.Pf * tf_to + P.Ph * env.tcd[j]
        curr    = env.pos[j].copy()
        visited[j] = True
        traj.append(j)

    return traj


BASELINES = {
    'Random':           'random',
    'Nearest-Neighbor': 'nearest_neighbor',
    'Greedy-Priority':  'greedy_priority',
    'PDR':              'pdr',
}
COLORS = {
    'Random':           '#888888',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'Attention (Ours)': '#009E73',
}
MARKERS = {'Random': 's', 'Nearest-Neighbor': '^', 'Greedy-Priority': 'D',
           'PDR': 'v', 'Attention (Ours)': 'o'}
ALL_METHODS = ['Random', 'Nearest-Neighbor', 'Greedy-Priority', 'PDR', 'Attention (Ours)']


def eval_baselines(M=30, n=200, seed=42):
    rng = np.random.default_rng(seed)
    out = {name: {'obj': [], 'nodes': [], 'waoi': [], 'priority': []}
           for name in BASELINES}
    for _ in range(n):
        s = int(rng.integers(0, 10_000_000))
        for name, key in BASELINES.items():
            env  = Env(M=M, seed=s)
            traj = run_baseline(env, key)
            out[name]['obj'].append(env.objective(traj))
            out[name]['nodes'].append(len(traj))
            out[name]['waoi'].append(P.theta1 * env.waoi(traj))
            out[name]['priority'].append(float(sum(env.wi[j] for j in traj)))
    return {k: {m: float(np.mean(v)) for m, v in vals.items()}
            for k, vals in out.items()}


def eval_policy(policy: Policy, M=30, n=200, seed=9999, device='cpu',
                use_beam: bool = True, beam_width: int = 5,
                use_postprocess: bool = True):
    """
    Evaluate policy with beam search + post-processing (test time).
    Set use_beam=False and use_postprocess=False for training-time eval (faster).
    """
    rng = np.random.default_rng(seed)
    out = {'obj': [], 'nodes': [], 'waoi': [], 'priority': []}
    policy.eval()

    with torch.no_grad():
        for _ in range(n):
            s   = int(rng.integers(0, 10_000_000))
            env = Env(M=M, seed=s)

            if use_beam:
                traj = rollout_beam(policy, env, device, beam_width=beam_width)
            else:
                traj, *_ = rollout(policy, env, device, greedy=True)

            if use_postprocess and traj:
                traj = post_process(env, traj)

            out['obj'].append(env.objective(traj))
            out['nodes'].append(len(traj))
            out['waoi'].append(P.theta1 * env.waoi(traj))
            out['priority'].append(float(sum(env.wi[j] for j in traj)))

    policy.train()
    return {k: float(np.mean(v)) for k, v in out.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 8.  PPO TRAINING LOOP  (Improvement 2)
# ══════════════════════════════════════════════════════════════════════════════
def _collect_rollout_buffer(policy: Policy, M: int, n_episodes: int,
                             rng: np.random.Generator, device: str):
    """
    Collect n_episodes into a flat buffer of transitions.
    Each transition: (x, mask, action, log_prob_old, value_old, ret=0)
    The episode return is terminal-only; ret is backfilled after collection.
    """
    buf_x       = []   # (M, 16) per step
    buf_mask    = []   # (M,)
    buf_action  = []   # int
    buf_log_old = []   # float
    buf_val_old = []   # float
    buf_ret     = []   # float  (filled below)
    buf_ep_end  = []   # bool   (True at last step of each episode)

    policy.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            visited = np.zeros(M, dtype=bool)
            curr    = P.home.copy()
            E_left  = P.Emax
            W_cum   = 0.0
            traj    = []
            ep_transitions = []   # (x, mask, action, lp, val)

            s   = int(rng.integers(0, 10_000_000))
            env = Env(M=M, seed=s)

            for _ in range(env.M):
                x, mask, feasible = make_features(env, curr, E_left, W_cum,
                                                   int(visited.sum()), visited, device)
                if not feasible.any():
                    break

                log_p, value = policy(x, mask)
                dist   = torch.distributions.Categorical(logits=log_p)
                action = int(dist.sample().item())

                ep_transitions.append((
                    x.cpu().numpy(),
                    mask.cpu().numpy(),
                    action,
                    float(log_p[action].item()),
                    float(value.item()),
                ))

                tf_to   = float(np.linalg.norm(env.pos[action] - curr)) / P.v
                E_left -= P.Pf * tf_to + P.Ph * env.tcd[action]
                curr    = env.pos[action].copy()
                visited[action] = True
                W_cum  += env.wi[action]
                traj.append(action)

            R = env.reward(traj)   # terminal reward for entire episode

            # Backfill: only the last transition gets R; all others get 0
            for idx, (xn, mn, an, lpn, vn) in enumerate(ep_transitions):
                is_last = (idx == len(ep_transitions) - 1)
                buf_x.append(xn)
                buf_mask.append(mn)
                buf_action.append(an)
                buf_log_old.append(lpn)
                buf_val_old.append(vn)
                buf_ret.append(R if is_last else 0.0)
                buf_ep_end.append(is_last)

    policy.train()
    return (np.array(buf_x,       dtype=np.float32),    # (T, M, 16)
            np.array(buf_mask,    dtype=bool),           # (T, M)
            np.array(buf_action,  dtype=np.int64),       # (T,)
            np.array(buf_log_old, dtype=np.float32),     # (T,)
            np.array(buf_val_old, dtype=np.float32),     # (T,)
            np.array(buf_ret,     dtype=np.float32),     # (T,)
            np.array(buf_ep_end,  dtype=bool))           # (T,)


def _compute_gae(buf_ret, buf_val, buf_ep_end,
                 gamma: float = 1.0, lam: float = 0.95):
    """
    Generalised Advantage Estimation.
    With gamma=1.0 (undiscounted) and terminal-only reward,
    this reduces to: adv[t] = (R_ep - val[t]) propagated with λ.
    """
    T    = len(buf_ret)
    adv  = np.zeros(T, dtype=np.float32)
    gae  = 0.0

    for t in reversed(range(T)):
        next_val = 0.0 if buf_ep_end[t] else buf_val[min(t+1, T-1)]
        delta    = buf_ret[t] + gamma * next_val - buf_val[t]
        gae      = delta + gamma * lam * (0.0 if buf_ep_end[t] else gae)
        adv[t]   = gae

    returns = adv + buf_val
    return adv, returns


def train(M=30, n_epochs=500, eps_per_epoch=128, lr=3e-4,
          entropy_beta=0.02, device='cpu', seed=42,
          save_path='models_mlp/policy_M30.pt',
          log_every=10, eval_every=50,
          # PPO hyperparameters
          ppo_epochs=3, ppo_clip=0.15, value_coef=0.5,
          gae_gamma=1.0, gae_lambda=0.95,
          minibatch_size=64):
    """
    Improvement 2 — PPO training with the lightweight MLP policy.

    Each epoch:
      1. Collect eps_per_epoch episodes into a buffer (no grad)
      2. Compute GAE advantages
      3. Run ppo_epochs update passes over the buffer in mini-batches
         with clipped surrogate objective

    Training eval uses greedy rollout (no beam/post-process) for speed.
    Final eval uses full beam + post-processing pipeline.
    """
    rng    = np.random.default_rng(seed)
    policy = Policy(hidden=256, input_dim=16).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/20)

    n_p = sum(x.numel() for x in policy.parameters())
    print(f'\nPolicy (v2): {n_p:,} params | M={M} | device={device}')
    print(f'Training (PPO): {n_epochs} epochs × {eps_per_epoch} eps '
          f'× {ppo_epochs} PPO epochs = {n_epochs*eps_per_epoch:,} rollouts')
    print(f'\n{"Ep":>5} {"Reward":>8} {"Nodes":>7} {"Obj":>8} {"PGLoss":>8}'
          f' {"EvalObj":>9} {"s/ep":>6}')
    print('─' * 60)

    history  = {'reward': [], 'nodes': [], 'obj': [], 'entropy': []}
    best_obj = float('inf')
    t_win    = deque(maxlen=10)
    rwd_win  = deque(maxlen=50)

    for ep in range(1, n_epochs + 1):
        t0 = time.perf_counter()

        # ── Step 1: Collect rollout buffer ───────────────────────────────
        (buf_x, buf_mask, buf_action, buf_log_old,
         buf_val, buf_ret, buf_ep_end) = _collect_rollout_buffer(
            policy, M, eps_per_epoch, rng, device)

        T = len(buf_action)
        if T == 0:
            continue

        # ── Step 2: GAE advantages ───────────────────────────────────────
        adv_np, ret_np = _compute_gae(buf_ret, buf_val, buf_ep_end,
                                       gamma=gae_gamma, lam=gae_lambda)
        # Normalise advantages
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        adv_t   = torch.from_numpy(adv_np).to(device)
        ret_t   = torch.from_numpy(ret_np).to(device)
        lpo_t   = torch.from_numpy(buf_log_old).to(device)
        act_t   = torch.from_numpy(buf_action).to(device)

        ep_pg_losses = []
        ep_entropies = []

        # ── Step 3: PPO update epochs ────────────────────────────────────
        indices = np.arange(T)
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, minibatch_size):
                idx = indices[start: start + minibatch_size]
                if len(idx) == 0:
                    continue

                # Re-evaluate current policy on stored states
                lp_batch  = []
                val_batch = []
                ent_batch = []

                for i in idx:
                    xi   = torch.from_numpy(buf_x[i]).to(device)
                    mi   = torch.from_numpy(buf_mask[i]).to(device)
                    lp_i, val_i = policy(xi, mi)
                    lp_batch.append(lp_i[act_t[i]])
                    val_batch.append(val_i)
                    ent_batch.append(-(lp_i.exp() * lp_i).sum())

                lp_new  = torch.stack(lp_batch)
                val_new = torch.stack(val_batch)
                ent_new = torch.stack(ent_batch).mean()

                # Importance ratio
                ratio  = torch.exp(lp_new - lpo_t[idx])
                adv_b  = adv_t[idx]

                # Clipped surrogate
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - ppo_clip, 1 + ppo_clip) * adv_b
                pg_loss = -torch.min(surr1, surr2).mean()

                val_loss = F.mse_loss(val_new, ret_t[idx])
                loss     = pg_loss + value_coef * val_loss - entropy_beta * ent_new

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

                ep_pg_losses.append(pg_loss.item())
                ep_entropies.append(ent_new.item())

        sched.step()
        t_win.append(time.perf_counter() - t0)

        # ── Logging ──────────────────────────────────────────────────────
        # Compute episode-level stats from the buffer's terminal returns
        ep_terminal_r = buf_ret[buf_ep_end]
        ep_terminal_n = []  # count nodes from buf (non-zero returns)
        mean_r  = float(np.mean(ep_terminal_r)) if len(ep_terminal_r) else 0.0
        mean_pg = float(np.mean(ep_pg_losses))  if ep_pg_losses else 0.0
        mean_H  = float(np.mean(ep_entropies))  if ep_entropies else 0.0
        rwd_win.append(mean_r)

        # Node count proxy: average trajectory length from buffer
        ep_lens = []
        cur_len = 0
        for end in buf_ep_end:
            cur_len += 1
            if end:
                ep_lens.append(cur_len)
                cur_len = 0
        mean_n = float(np.mean(ep_lens)) if ep_lens else 0.0
        mean_o = -mean_r  # objective = -reward

        history['reward'].append(mean_r)
        history['nodes'].append(mean_n)
        history['obj'].append(mean_o)
        history['entropy'].append(mean_H)

        if ep % log_every == 0:
            eval_str = '      —  '
            if ep % eval_every == 0:
                # Fast eval during training (greedy, no post-process)
                ev = eval_policy(policy, M=M, n=30, device=device,
                                 use_beam=False, use_postprocess=False)
                eval_str = f'{ev["obj"]:+9.2f}'
                if ev['obj'] < best_obj:
                    best_obj = ev['obj']
                    torch.save({'policy': policy.state_dict(),
                                'best_obj': best_obj}, save_path)

            eta = np.mean(t_win) * (n_epochs - ep) / 60
            print(f'{ep:5d} {np.mean(rwd_win):8.2f} {mean_n:7.1f} '
                  f'{mean_o:8.2f} {mean_pg:8.4f} {eval_str} '
                  f'{np.mean(t_win):6.1f}s  ETA:{eta:.0f}m')

    # Save final
    torch.save({'policy': policy.state_dict(), 'best_obj': best_obj}, save_path)
    np.save(f'results/history_M{M}.npy', history)
    print(f'\nBest eval obj (greedy): {best_obj:.2f}')
    return policy, history


# ══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def fig_convergence(history, M):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f'PPO Training Convergence  M={M}', fontsize=13)
    for ax, key, lbl, col in [
        (axes[0], 'reward',  'Episode Reward',   '#009E73'),
        (axes[1], 'nodes',   'Nodes Visited',    '#0072B2'),
        (axes[2], 'entropy', 'Policy Entropy H', '#D55E00'),
    ]:
        y  = np.array(history[key])
        k  = min(30, len(y))
        sm = np.convolve(y, np.ones(k) / k, 'same')
        ax.plot(y,  color=col, alpha=0.25, lw=0.8)
        ax.plot(sm, color=col, lw=2.0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(lbl)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/fig1_convergence_M{M}.png', dpi=150)
    plt.close()
    print(f'  Saved results/fig1_convergence_M{M}.png')


def fig_comparison(all_res, M, n):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Performance Comparison  M={M},  {n} instances', fontsize=13)
    for ax, key, title, ylab in [
        (axes[0], 'waoi',     '(a) Weighted AoI',       'θ₁·WAoI'),
        (axes[1], 'priority', '(b) Total Priority',      'Σwᵢ'),
        (axes[2], 'obj',      '(c) Composite Objective', 'θ₁WAoI−θ₂Σwᵢ'),
    ]:
        vals = [all_res[m][key] for m in ALL_METHODS]
        bars = ax.bar(range(len(ALL_METHODS)), vals,
                      color=[COLORS[m] for m in ALL_METHODS],
                      edgecolor='black', lw=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(len(ALL_METHODS)))
        ax.set_xticklabels(ALL_METHODS, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel(ylab, fontsize=10)
        ax.axhline(0, color='black', lw=0.5)
        ax.grid(axis='y', alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -2),
                    f'{v:.1f}', ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'results/fig2_comparison_M{M}.png', dpi=150)
    plt.close()
    print(f'  Saved results/fig2_comparison_M{M}.png')


def fig_reward_dist(all_res_raw, M, n):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Reward & Nodes Distribution  M={M},  {n} instances', fontsize=13)
    rewards_list = [all_res_raw[m]['reward_list'] for m in ALL_METHODS]
    nodes_list   = [all_res_raw[m]['nodes_list']  for m in ALL_METHODS]
    for ax, data, ylabel, title in [
        (ax1, rewards_list, 'Episode Reward', '(a) Reward Distribution'),
        (ax2, nodes_list,   'Nodes Visited',  '(b) Nodes Visited'),
    ]:
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', lw=2))
        for patch, m in zip(bp['boxes'], ALL_METHODS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(ALL_METHODS) + 1))
        ax.set_xticklabels(ALL_METHODS, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/fig3_distributions_M{M}.png', dpi=150)
    plt.close()
    print(f'  Saved results/fig3_distributions_M{M}.png')


def fig_scalability(scale_res, M_list):
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in ALL_METHODS:
        ax.plot(M_list, [scale_res[m][M] for M in M_list],
                marker=MARKERS[m], color=COLORS[m], label=m, lw=1.8, markersize=7)
    ax.set_xlabel('Number of Nodes M', fontsize=12)
    ax.set_ylabel('Composite Objective', fontsize=12)
    ax.set_title('Scalability: Objective vs Network Size  (beam+post-process)', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig4_scalability.png', dpi=150)
    plt.close()
    print('  Saved results/fig4_scalability.png')


def fig_battery(battery_res):
    Emax_kJ = [E / 1000 for E in battery_res['Emax_list']]
    fig, ax  = plt.subplots(figsize=(9, 5))
    for m in ALL_METHODS:
        ax.plot(Emax_kJ, battery_res[m], marker=MARKERS[m],
                color=COLORS[m], label=m, lw=1.8, markersize=7)
    ax.set_xlabel('Battery Emax (kJ)', fontsize=12)
    ax.set_ylabel('Composite Objective', fontsize=12)
    ax.set_title('Battery Sensitivity  M=30', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/fig5_battery.png', dpi=150)
    plt.close()
    print('  Saved results/fig5_battery.png')


def fig_trajectory(policy, M, device, seed=7):
    env  = Env(M=M, seed=seed)
    # Use beam search + post-processing for the figure
    traj = rollout_beam(policy, env, device, beam_width=5)
    traj = post_process(env, traj)

    fig, ax = plt.subplots(figsize=(8, 7))
    for i in range(M):
        sz  = 40 + (env.wi[i] / P.wi_hi) * 140
        clr = COLORS['Attention (Ours)'] if i in traj else '#cccccc'
        ew  = 1.3 if i in traj else 0.3
        ax.scatter(*env.pos[i], s=sz, c=clr, edgecolors='black', lw=ew, zorder=4)
        ax.text(env.pos[i, 0] + 12, env.pos[i, 1] + 12, f'w={env.wi[i]:.1f}', fontsize=6.5)
    px = [P.home[0]] + [env.pos[j, 0] for j in traj] + [P.home[0]]
    py = [P.home[1]] + [env.pos[j, 1] for j in traj] + [P.home[1]]
    ax.plot(px, py, '-', color=COLORS['Attention (Ours)'], lw=1.8, alpha=0.75, zorder=3)
    for k in range(len(px) - 1):
        ax.annotate('', xy=(px[k+1], py[k+1]), xytext=(px[k], py[k]),
                    arrowprops=dict(arrowstyle='->', color=COLORS['Attention (Ours)'],
                                    lw=1.2, mutation_scale=12))
    for order, j in enumerate(traj):
        ax.text(env.pos[j, 0] - 18, env.pos[j, 1] - 18, str(order + 1), fontsize=8,
                color='white', fontweight='bold',
                bbox=dict(boxstyle='circle', facecolor=COLORS['Attention (Ours)'],
                          edgecolor='none', pad=0.1))
    ax.scatter(*P.home, s=220, marker='*', c='gold', edgecolors='black', lw=1.5, zorder=6)
    obj  = env.objective(traj)
    waoi = P.theta1 * env.waoi(traj)
    ax.set_title(f'Learned Trajectory (beam+2opt+insert)  M={M}, nodes={len(traj)}\n'
                 f'WAoI={waoi:.1f}  Obj={obj:.2f}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.grid(alpha=0.2)
    ax.set_xlim(-30, P.area + 30)
    ax.set_ylim(-30, P.area + 30)
    plt.tight_layout()
    plt.savefig(f'results/fig6_trajectory_M{M}.png', dpi=150)
    plt.close()
    print(f'  Saved results/fig6_trajectory_M{M}.png')


# ══════════════════════════════════════════════════════════════════════════════
# 10.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick',   action='store_true', help='Fast demo: fewer epochs/instances')
    ap.add_argument('--epochs',  type=int, default=0)
    ap.add_argument('--M',       type=int, nargs='+', default=[], help='Override M list')
    ap.add_argument('--no-beam', action='store_true', help='Disable beam search at eval')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.quick:
        n_epochs      = args.epochs or 100
        eps_per_epoch = 32
        M_list        = args.M or [20, 30]
        n_eval        = 30
    else:
        n_epochs      = args.epochs or 500
        eps_per_epoch = 128
        M_list        = args.M or [20, 30, 40, 50, 60, 70, 80, 90, 100]
        n_eval        = 200

    use_beam = not args.no_beam

    print('=' * 65)
    print('  UAV AoI — PPO MLP Policy v2 (richer features + post-process)')
    print(f'  device={device}  epochs={n_epochs}  eps/epoch={eps_per_epoch}')
    print(f'  M values: {M_list}')
    print(f'  Beam search at eval: {use_beam}')
    print('=' * 65)

    os.makedirs('models_mlp', exist_ok=True)

    for Mv in M_list:
        print(f"\n{'='*65}")
        print(f"  Training MLP v2 for M={Mv}")
        print(f"{'='*65}")

        save_path = f'models_mlp/policy_M{Mv}.pt'

        pol_m, hist = train(
            M=Mv,
            n_epochs=n_epochs,
            eps_per_epoch=eps_per_epoch,
            lr=3e-4,
            entropy_beta=0.02,
            device=device,
            save_path=save_path,
            log_every=10,
            eval_every=50,
            ppo_epochs=3,
            ppo_clip=0.15,
            value_coef=0.5,
            seed=42,
        )

        # Load best checkpoint
        ckpt = torch.load(save_path, map_location=device)
        pol_m.load_state_dict(ckpt['policy'])

        # Final evaluation with full pipeline
        print(f'\n  Final evaluation (beam={use_beam}, post-process=True) ...')
        pr = eval_policy(pol_m, M=Mv, n=n_eval, device=device,
                         use_beam=use_beam, beam_width=5,
                         use_postprocess=True)
        print(f'  -> [M={Mv}] Obj={pr["obj"]:.3f}  Nodes={pr["nodes"]:.1f}  '
              f'WAoI={pr["waoi"]:.2f}  Priority={pr["priority"]:.2f}')

        # Figures
        fig_convergence(hist, Mv)
        fig_trajectory(pol_m, Mv, device)

    print('\n' + '=' * 65)
    print('  All models trained and saved to models_mlp/')
    print('  Note: optimality_gap.py needs rerunning — models use 16-dim features now.')
    print('=' * 65)