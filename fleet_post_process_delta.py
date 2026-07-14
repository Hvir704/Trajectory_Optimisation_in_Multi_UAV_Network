"""
fleet_post_process_delta.py  --  O(1)-per-candidate insertion via delta evaluation.
=================================================================================
Same result as the current fleet_post_process, computed without recomputing the
full chain objective for every candidate. The insertion inner loop goes from
O(L^2) to O(L) per (node, chain) by using the structure of env.waoi:

  objective(traj) = theta1 * sum_k W_k*(tcd[c_k] + tf(c_k, next_k))  -  theta2 * sum wi
                    with W_k = cumulative weight through position k.

Inserting node u at position p (chain c, length L):
  d_priority = -theta2 * wi[u]                         (position-independent)
  d_waoi     = W_{p-1} * (tf(c_{p-1},u) - tf(c_{p-1},oldnext))     # p-1 leg change
             + (W_{p-1}+wi[u]) * (tcd[u] + tf(u, c_p or home))     # u's own term
             + wi[u] * SuffixLegSum(p)                             # suffix W bump
  gain = base_obj - obj(cand) = theta2*wi[u] - theta1*d_waoi
  d_energy   = Pf*(tf(a,u)+tf(u,b)-tf(a,b)) + Ph*tcd[u]            # a,b = neighbours/home

Precompute per chain (O(L)): cumulative weights Wpre, per-position legs, suffix leg
sums, and base energy. Each candidate is then O(1); all positions O(L).

2-opt is UNCHANGED (kept as the module's _two_opt_chain) -- it dominates only at
K=1/long chains, which isn't a deployment regime. This targets K>=4 large-M.

IDENTITY CAVEAT: this computes the same quantities a cheaper way, so results match
the full-recompute pp to float64 precision (~1e-10), which is ~1e4x below the
1e-6 acceptance threshold. That means trajectory-identical in all but knife-edge
float ties. It is NOT guaranteed literally bit-identical the way the earlier
restructuring was (that reused env.objective directly). RUN THE VERIFY MAIN below
before wiring it in.

USAGE
  1. Verify:   python fleet_post_process_delta.py
     -> compares this vs the current module fleet_post_process on 12 instances.
  2. If it reports identical, wire it in by replacing the body of
     fleet_post_process in multi_uav_solver.py with a call:
         def fleet_post_process(env, fleet):
             from fleet_post_process_delta import fleet_post_process_delta
             return fleet_post_process_delta(env, fleet)
     (or paste the function over it).
"""
from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, MultiUAVPolicy, fleet_rollout,
                              _two_opt_chain, _chain_energy)


def fleet_post_process_delta(env, fleet):
    # 1. per-chain 2-opt (unchanged)
    for k in range(fleet.K):
        fleet.trajs[k] = _two_opt_chain(env, fleet.trajs[k], fleet.Emax_each)

    Ee = fleet.Emax_each
    served = set(j for t in fleet.trajs for j in t)
    unvisited = [j for j in range(env.M) if j not in served]   # ascending

    precomp = [None] * fleet.K   # (Wpre, legs, suf)
    base_E  = [0.0] * fleet.K

    def build(k):
        chain = fleet.trajs[k]; L = len(chain)
        Wpre = [0.0] * (L + 1)
        for p in range(L):
            Wpre[p + 1] = Wpre[p] + float(env.wi[chain[p]])
        legs = [0.0] * L
        for m in range(L):
            nxt = env.pos[chain[m + 1]] if m < L - 1 else P.home
            legs[m] = float(env.tcd[chain[m]]) + env.tf(env.pos[chain[m]], nxt)
        suf = [0.0] * (L + 1)
        for p in range(L - 1, -1, -1):
            suf[p] = suf[p + 1] + legs[p]
        precomp[k] = (Wpre, legs, suf)
        base_E[k] = _chain_energy(env, chain)

    for k in range(fleet.K):
        build(k)

    def eval_jk(u, k):
        """Best improving feasible insertion of node u into chain k, via delta.
        Returns (gain, pos). gain<=0 => none. Mirrors the original tie-break:
        strict '>' over positions, first max-pos wins."""
        chain = fleet.trajs[k]; L = len(chain)
        Wpre, legs, suf = precomp[k]
        posu = env.pos[u]; wu = float(env.wi[u]); tcdu = float(env.tcd[u]); bE = base_E[k]
        best_g = 0.0; best_pos = -1
        for p in range(L + 1):
            a = P.home if p == 0 else env.pos[chain[p - 1]]
            b = P.home if p == L else env.pos[chain[p]]
            dE = P.Pf * (env.tf(a, posu) + env.tf(posu, b) - env.tf(a, b)) + P.Ph * tcdu
            if bE + dE > Ee:                       # matches _chain_energy(cand) > Ee
                continue
            dW = 0.0
            if p > 0:
                old_travel = legs[p - 1] - float(env.tcd[chain[p - 1]])   # tf(c_{p-1}, oldnext)
                new_travel = env.tf(env.pos[chain[p - 1]], posu)          # tf(c_{p-1}, u)
                dW += Wpre[p] * (new_travel - old_travel)
            nextu = env.pos[chain[p]] if p < L else P.home
            dW += (Wpre[p] + wu) * (tcdu + env.tf(posu, nextu))
            dW += wu * suf[p]
            gain = P.theta2 * wu - P.theta1 * dW    # = base_obj - obj(cand)
            if gain > best_g:
                best_g = gain; best_pos = p
        return best_g, best_pos

    cache = {}
    for u in unvisited:
        for k in range(fleet.K):
            cache[(u, k)] = eval_jk(u, k)

    while True:
        best_gain = 1e-6; best_move = None
        for u in unvisited:                         # ascending == original scan order
            for k in range(fleet.K):
                g, p = cache[(u, k)]
                if g > best_gain:
                    best_gain = g; best_move = (k, p, u)
        if best_move is None:
            break
        k, p, u = best_move
        chain = fleet.trajs[k]
        fleet.trajs[k] = chain[:p] + [u] + chain[p:]
        fleet.visited[u] = True
        unvisited.remove(u)
        for kk in range(fleet.K):
            cache.pop((u, kk), None)
        build(k)                                    # only mutated chain
        for uu in unvisited:
            cache[(uu, k)] = eval_jk(uu, k)

    return fleet


# ── verification: delta vs current module pp, same rollout, no confounds ────────
def _verify():
    import numpy as np, torch
    from multi_uav_solver import fleet_post_process as fpp_current

    CKPT = 'models_multi_uav/fleet_M100_K4_split_seed42.pt'
    M, K, N = 100, 4, 12
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(CKPT, map_location=dev)
    pol = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(dev)
    pol.load_state_dict(ck['policy']); pol.eval()
    Ee = P.Emax / K
    rng = np.random.default_rng(2025)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(N)]

    worst = 0.0; mism = 0
    print(f'delta vs current pp   ckpt={CKPT}  M={M} K={K}  n={N}')
    print(f'{"seed":>9} {"DELTA":>10} {"CURRENT":>10} {"|diff|":>9}  trajs')
    for s in seeds:
        e1 = Env(M=M, seed=s); f1 = fleet_rollout(pol, e1, K, dev, Emax_each=Ee, greedy=True)
        fleet_post_process_delta(e1, f1); o1 = f1.fleet_objective()
        e2 = Env(M=M, seed=s); f2 = fleet_rollout(pol, e2, K, dev, Emax_each=Ee, greedy=True)
        fpp_current(e2, f2); o2 = f2.fleet_objective()
        d = abs(o1 - o2); worst = max(worst, d)
        same = [list(a) for a in f1.trajs] == [list(a) for a in f2.trajs]
        mism += (0 if same else 1)
        print(f'{s:>9} {o1:>10.4f} {o2:>10.4f} {d:>9.1e}  '
              f'{"identical" if same else "DIFFER"}')
    print(f'\nworst |obj diff| = {worst:.1e}   traj mismatches: {mism}/{N}')
    if worst < 1e-6 and mism == 0:
        print('OK: identical to float precision. Safe to wire in.')
    elif worst < 1e-6:
        print('Objectives match to <1e-6 but a traj differs -> knife-edge float tie, '
              'not a bug; both solutions equal quality. Widen N to gauge frequency.')
    else:
        print('NOT identical -> real discrepancy. Paste output back for a fix.')


if __name__ == '__main__':
    _verify()
