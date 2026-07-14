"""features.py — Feature extraction and rollout for the attention policy.

Includes explicit spatial geometry (d_curr, d_home) to match MLP
performance and accelerate REINFORCE convergence.

MOSAC-ATT update: batch_rollout now accepts a `temperature` parameter
that is forwarded to policy.sample_action() during stochastic rollouts.
At evaluation (greedy=True) temperature is ignored — argmax is always
deterministic. Default temperature=1.0 preserves identical behaviour
for all callers that don't pass it explicitly.
"""

import numpy as np
import torch
from env import Params, UAVEnv

# ─────────────────────────────────────────────────────────────────────────────
# Feature construction
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_tensors(obs: dict, p: Params, device='cpu'):
    """
    Convert environment observation dict to (node_feats, ctx_feats, mask) tensors.

    Node features — 5-dim per node:
        [x/area, y/area, wi/wi_hi, tcd/tcd_max, feasible_flag]
    Context features — 6-dim global:
        [cx/area, cy/area, E_left/Emax, W_cumul/W_max, 
         n_visited/M, dist_home/area]
    Mask — bool (B, M): True = node cannot be visited.
    """
    tcd_max = p.tcd(p.Di_hi)
    M       = p.M

    # Node features
    nf = np.zeros((M, 5), dtype=np.float32)
    nf[:, 0] = obs['node_pos'][:, 0] / p.area
    nf[:, 1] = obs['node_pos'][:, 1] / p.area
    nf[:, 2] = obs['wi'] / p.wi_hi
    nf[:, 3] = obs['tcd'] / tcd_max

    # Feasibility flag: unvisited AND energy-feasible
    for j in range(M):
        if obs['visited'][j]:
            continue
        e_need = (p.e_fly(obs['curr_pos'], obs['node_pos'][j])
                  + p.Ph * obs['tcd'][j]
                  + p.e_fly(obs['node_pos'][j], p.home))
        if e_need <= obs['E_left']:
            nf[j, 4] = 1.0

    # Context features
    dist_home = float(np.linalg.norm(obs['curr_pos'] - np.array(p.home)))
    W_cumul   = float(np.sum(obs['wi'] * obs['visited']))
    n_visited = int(obs['visited'].sum())

    cf = np.array([
        obs['curr_pos'][0] / p.area,
        obs['curr_pos'][1] / p.area,
        obs['E_left'] / p.Emax,
        W_cumul / (M * p.wi_hi),
        n_visited / M,
        dist_home / p.area,
    ], dtype=np.float32)

    # Mask: True = cannot visit (visited or energy-infeasible)
    mask = np.ones(M, dtype=bool)
    for j in range(M):
        if nf[j, 4] > 0.5:
            mask[j] = False

    nf_t   = torch.tensor(nf,   dtype=torch.float32, device=device).unsqueeze(0)
    cf_t   = torch.tensor(cf,   dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.bool,    device=device).unsqueeze(0)

    return nf_t, cf_t, mask_t

# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

def batch_rollout(policy, env: UAVEnv, device,
                  greedy: bool = False,
                  temperature: float = 1.0):
    """
    Run one complete episode with the given policy.

    Parameters
    ----------
    policy      : AttentionPolicy
    env         : UAVEnv (already constructed; reset() is called internally)
    device      : torch device string
    greedy      : if True use argmax (no sampling); temperature is ignored
    temperature : tau for Fix 2 heated-up softmax. Forwarded to
                  policy.sample_action() during stochastic rollouts.
                  Default 1.0 -> standard categorical sampling (no change).
                  Values >1 flatten the distribution (more exploration).

    Returns
    -------
    traj      : list[int]       visited node indices in order
    log_probs : list[Tensor]    per-step log-probabilities (scalar tensors)
    values    : list[Tensor]    per-step baseline values   (scalar tensors)
    reward    : float           episode return (= -objective, at terminal step)

    Notes
    -----
    - log_probs are always computed under the UNSCALED (tau=1) policy so that
      the REINFORCE gradient is unbiased. Temperature only governs which
      action is sampled. This invariant is enforced inside policy.sample_action().
    - At greedy=True the call is wrapped in torch.no_grad() and temperature
      is ignored — argmax is always deterministic.
    """
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
                log_p, value = policy(nf, cf, mask)
            action = log_p.argmax(dim=-1).item()
            lp     = log_p[0, action]
            v      = value[0]
        else:
            # Fix 2: pass temperature to sample_action
            action_t, lp, v = policy.sample_action(
                nf, cf, mask, temperature=temperature)
            action = action_t.item()

        log_probs.append(lp)
        values.append(v)

        obs, reward, done = env.step(action)

    return env.traj[:], log_probs, values, reward

# Alias for backwards compatibility with trainer_fixed.py's try/except import
rollout_episode = batch_rollout