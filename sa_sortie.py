"""
sa_sortie.py -- SA sortie planner for the dynamic simulator.

PORTED, NOT IMPORTED, from sa_repair2.py. Reasons:
  - sa_repair2 / compare_baseline solve a K-WAY SIMULTANEOUS PARTITION over a
    fixed node set with fixed tcd (data volume). A dynamic sortie is ONE drone,
    ONE route, chosen at launch time from live ages/weights with an ESTIMATED
    dwell -- the K=1 special case of the same combinatorial problem, but a
    different objective and different inputs.
  - compare_baseline's module-level constants (AREA=1000, Emax=50000, Pf=150,
    Ph=200) are the OLD static scale. This uses DynParams throughout.
  - The valuable, tested part of sa_repair2 is the k-for-1 COST-AWARE REPAIR
    OPERATOR (CONTEXT_16): pick the insertion target first, eject the
    cheapest-value-per-joule served nodes until it fits. That is what unfroze
    SA at cap-packed cells, and dynamic sorties are cap-packed by construction
    (CONTEXT_65 measured 92-96% energy utilisation under the stub). Ported here
    verbatim in spirit, adapted to a single chain.

OBJECTIVE -- a documented simplification, not a full port of fleet_obj.
  fleet_obj's chain_waoi accumulates CUMULATIVE weighted delay: a node visited
  late in a chain is charged for the priority-weight of every node visited
  before it, because those earlier stops delayed it. That structure is correct
  for a single static mission with fixed payloads.

  Here we instead maximise
        value(route) = sum_j  weight_est[j] * age[j]
  i.e. total priority-weighted age reset, independent of visit order within the
  sortie. Order is chosen by the move set to fit as much value as possible in
  the energy budget, but is NOT charged for intra-sortie delay.

  This is deliberately simpler than porting the full cumulative-delay objective,
  because within one ~15-min sortie the ordering effect on AGE AT COLLECTION is
  second-order next to the SET of nodes chosen -- but it is a simplification,
  not a proof, and should be revisited if C4 numbers look sensitive to it.

SEEDING -- CONTEXT_60 §5.4 / this session's finding.
  sa_repair2.py's own main() seeds BOTH the baseline and repair arms by RESTART
  INDEX (0 or 1), not by instance. That bug is confirmed present in the file as
  uploaded, unfixed. It is NOT ported here: every call below takes an explicit,
  distinct seed built from (sortie index, restart index), never a bare small
  integer reused across restarts.
"""

from __future__ import annotations

import numpy as np
from typing import List

from dyn_env import DynParams, SortieRequest


def _tf(a, b, v):
    return float(np.linalg.norm(a - b)) / v


def _chain_energy(chain, pos, home, dwell, p: DynParams):
    E = 0.0
    prev = home
    for j in chain:
        E += p.e_fly(_tf(prev, pos[j], p.v) * p.v) + p.e_hover(dwell[j])
        # e_fly takes distance, tf returns time -> convert back: d = v*t
        prev = pos[j]
    E += p.e_fly(_tf(prev, home, p.v) * p.v)
    return E


def _chain_value(chain, weight, age):
    return sum(weight[j] * age[j] for j in chain)


def _ins_cost(chain, j, pos, dwell, home, p: DynParams, at: int):
    a = home if at == 0 else pos[chain[at - 1]]
    b = home if at == len(chain) else pos[chain[at]]
    d_a = np.linalg.norm(a - pos[j]); d_b = np.linalg.norm(pos[j] - b); d_ab = np.linalg.norm(a - b)
    return p.e_fly(d_a + d_b - d_ab) + p.e_hover(dwell[j])


def _best_ins(chain, j, pos, dwell, home, p: DynParams):
    best, bp = None, 0
    for at in range(len(chain) + 1):
        d = _ins_cost(chain, j, pos, dwell, home, p, at)
        if best is None or d < best:
            best, bp = d, at
    return best, bp


def _rem_gain(chain, i, pos, dwell, home, p: DynParams):
    a = home if i == 0 else pos[chain[i - 1]]
    b = home if i == len(chain) - 1 else pos[chain[i + 1]]
    j = chain[i]
    d_a = np.linalg.norm(a - pos[j]); d_b = np.linalg.norm(pos[j] - b); d_ab = np.linalg.norm(a - b)
    return p.e_fly(d_a + d_b - d_ab) + p.e_hover(dwell[j])


def sa_sortie(pos: np.ndarray, weight: np.ndarray, dwell: np.ndarray, home: np.ndarray,
              E_budget: float, p: DynParams, iters: int, seed: int,
              repair_p: float = 0.35) -> List[int]:
    """Single-chain SA with the ported k-for-1 repair operator.

    Greedy-constructs a feasible start (reuses the stub's scoring, since it is
    already a decent constructive heuristic), then improves by local search.
    """
    rng = np.random.default_rng(seed)
    M = len(pos)

    # --- greedy construction (reuse the stub's ratio rule as the seed tour) ---
    chain: List[int] = []
    cur = home.copy()
    E = E_budget
    used = np.zeros(M, dtype=bool)
    while True:
        d_to = np.linalg.norm(pos - cur, axis=1)
        d_home = np.linalg.norm(pos - home, axis=1)
        e_need = p.e_fly(d_to + d_home) + p.e_hover(dwell)
        feas = (~used) & (e_need <= E)
        if not feas.any():
            break
        e_marg = np.maximum(p.e_fly(d_to) + p.e_hover(dwell), 1.0)
        score = np.where(feas, weight * np.maximum(1e-9, np.zeros(M)) if False else
                          weight / e_marg, -np.inf)
        # NOTE: greedy seed ranks purely by value/energy, independent of the age
        # term -- age enters only via `weight` argument, which callers already
        # multiply appropriately (see build_sa_planner below).
        j = int(np.argmax(score))
        E -= p.e_fly(d_to[j]) + p.e_hover(dwell[j])
        cur = pos[j].copy()
        used[j] = True
        chain.append(j)

    served = set(chain)
    cur_val = _chain_value(chain, weight, np.ones(M))  # placeholder, replaced below
    # value function passed in already has age folded into `weight` (see
    # build_sa_planner) so `age` here is unity -- keeps this function generic.
    cur_val = _chain_value(chain, weight, np.ones(M))
    best_val, best_chain = cur_val, chain[:]

    T0 = max(cur_val * 0.05, 1e-3)
    T1 = 1e-4

    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        nc = chain[:]
        uns = [j for j in range(M) if j not in served]
        is_repair = (rng.random() < repair_p) and uns and nc

        if is_repair:
            # --- ported k-for-1 cost-aware eject-insert ---
            w_u = weight[uns]
            tot = w_u.sum()
            if tot <= 0:
                pr = np.ones(len(uns)) / len(uns)
            else:
                pr = w_u / tot
            j = int(rng.choice(uns, p=pr))
            need, at = _best_ins(nc, j, pos, dwell, home, p)
            slack = E_budget - _chain_energy(nc, pos, home, dwell, p)
            if need > slack:
                cand = []
                for i in range(len(nc)):
                    gfree = _rem_gain(nc, i, pos, dwell, home, p)
                    if gfree > 1e-9:
                        cand.append((weight[nc[i]] / gfree, i, gfree))
                cand.sort()
                freed, drop = 0.0, []
                for _, i, gfree in cand:
                    drop.append(i); freed += gfree
                    if slack + freed >= need:
                        break
                if slack + freed < need:
                    continue
                for i in sorted(drop, reverse=True):
                    del nc[i]
                need, at = _best_ins(nc, j, pos, dwell, home, p)
                if _chain_energy(nc, pos, home, dwell, p) + need > E_budget + 1e-9:
                    continue
            nc = nc[:at] + [j] + nc[at:]
        else:
            op = rng.integers(0, 4)
            if op == 0 and uns:
                j = int(rng.choice(uns))
                at = int(rng.integers(0, len(nc) + 1))
                nc = nc[:at] + [j] + nc[at:]
            elif op == 1 and nc:
                i = int(rng.integers(0, len(nc))); del nc[i]
            elif op == 2 and len(nc) >= 2:
                a, b = sorted(rng.choice(len(nc), 2, replace=False))
                nc[a:b + 1] = nc[a:b + 1][::-1]
            elif op == 3 and uns and nc:
                i = int(rng.integers(0, len(nc)))
                nc[i] = int(rng.choice(uns))

        if _chain_energy(nc, pos, home, dwell, p) > E_budget + 1e-6:
            continue
        val = _chain_value(nc, weight, np.ones(M))
        d = cur_val - val  # SA framework minimises; we want to MAXIMISE value
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            chain = nc
            cur_val = val
            served = set(chain)
            if val > best_val:
                best_val, best_chain = val, chain[:]

    return best_chain


def build_sa_planner(iters: int = 1200, repair_p: float = 0.35):
    """Factory -> a SortiePlanner closure for DynSim.

    SEEDING: each call gets seed = hash(t_launch) so distinct sorties never
    collide, without relying on any small reused index (CONTEXT_60 §5.4).
    """
    _counter = {"n": 0}

    def planner(req: SortieRequest) -> List[int]:
        p = req.p
        # value-per-node folded into `weight` so sa_sortie's generic value
        # function (weight * 1) equals weight_est[j] * age[j] as intended.
        folded_weight = req.weight_est * req.age
        _counter["n"] += 1
        seed = (_counter["n"] * 7919 + int(req.E_usable) % 1000) % (2**31 - 1)
        return sa_sortie(req.pos, folded_weight, req.dwell_est, req.home,
                          req.E_usable, p, iters=iters, seed=seed, repair_p=repair_p)

    return planner


if __name__ == "__main__":
    # smoke test: SA planner vs stub, same cell, same seed stream
    from dyn_env import DynSim, greedy_ratio_planner
    import time

    p = DynParams(M=100, K=4, T_horizon=6 * 3600, T_burnin=1.5 * 3600)

    t0 = time.time()
    m_stub = DynSim(p, planner=greedy_ratio_planner, seed=1).run()
    t1 = time.time()
    m_sa = DynSim(p, planner=build_sa_planner(iters=800), seed=1).run()
    t2 = time.time()

    print(f"stub : J={m_stub['J_timeavg']:12.1f}  Ts/tc={m_stub['T_s_over_t_c']:.3f}  "
          f"n={m_stub['mean_n_visited']:.1f}  ({t1-t0:.1f}s)")
    print(f"SA   : J={m_sa['J_timeavg']:12.1f}  Ts/tc={m_sa['T_s_over_t_c']:.3f}  "
          f"n={m_sa['mean_n_visited']:.1f}  ({t2-t1:.1f}s)")
