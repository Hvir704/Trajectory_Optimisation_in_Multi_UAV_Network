"""
baselines.py — Deterministic baselines (no learning).
All use the same energy feasibility check as the policy.
"""

import numpy as np
from env import Params, UAVEnv
from typing import List


def _run(env: UAVEnv, score_fn) -> List[int]:
    """Generic greedy runner using a per-step scoring function."""
    obs  = env.reset()
    done = False
    while not done:
        feasible = [j for j in range(env.p.M)
                    if not obs['visited'][j]
                    and _is_feasible(j, obs, env.p)]
        if not feasible:
            obs, _, done = env.step(-1)
            break
        j    = max(feasible, key=lambda j: score_fn(j, obs, env))
        obs, _, done = env.step(j)
    return env.traj[:]


def _is_feasible(j, obs, p):
    e = (p.e_fly(obs['curr_pos'], obs['node_pos'][j])
         + p.Ph * obs['tcd'][j]
         + p.e_fly(obs['node_pos'][j], p.home))
    return e <= obs['E_left']


def greedy_priority(env):
    return _run(env, lambda j, obs, e: obs['wi'][j])


def nearest_neighbor(env):
    return _run(env, lambda j, obs, e:
                -np.linalg.norm(obs['curr_pos'] - obs['node_pos'][j]))


def pdr(env):
    def score(j, obs, e):
        d = max(np.linalg.norm(obs['curr_pos'] - obs['node_pos'][j]), 1e-9)
        return obs['wi'][j] / d
    return _run(env, score)


def random_policy(env):
    return _run(env, lambda j, obs, e: np.random.random())


BASELINES = {
    'Random':           random_policy,
    'Nearest-Neighbor': nearest_neighbor,
    'Greedy-Priority':  greedy_priority,
    'PDR':              pdr,
}


def evaluate_all(params: Params, n=200, seed=42) -> dict:
    rng     = np.random.default_rng(seed)
    results = {name: {'obj': [], 'nodes': [], 'waoi': [], 'priority': []}
               for name in BASELINES}
    for _ in range(n):
        s = int(rng.integers(0, 10_000_000))
        for name, fn in BASELINES.items():
            env  = UAVEnv(params, seed=s)
            traj = fn(env)
            results[name]['obj'].append(env.objective(traj))
            results[name]['nodes'].append(len(traj))
            results[name]['waoi'].append(params.theta1 * env.waoi(traj))
            results[name]['priority'].append(sum(env.wi[j] for j in traj))
    return {name: {k: float(np.mean(v)) for k, v in vals.items()}
            for name, vals in results.items()}
