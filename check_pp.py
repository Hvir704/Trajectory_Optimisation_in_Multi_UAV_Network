"""
compare_pp.py  --  prove the pasted fleet_post_process is (or isn't) identical
to the original, with NO checkpoint/seed confound. Runs NEW pp (the one now in
multi_uav_solver) and a REF copy of the ORIGINAL pp on the SAME deterministic
rollout, per instance, and reports any divergence.

Put this in the repo root and run:  python compare_pp.py
"""
import numpy as np, torch
from uav_aoi_solver import P, Env
from multi_uav_solver import (MP, MultiUAVPolicy, fleet_rollout,
                              fleet_post_process,        # NEW (pasted) version
                              _chain_energy)

CKPT = 'models_multi_uav/fleet_M100_K4_split_seed42.pt'
M, K, N = 100, 4, 12


# ── ORIGINAL functions, verbatim, as reference ──────────────────────────────
def _two_opt_chain_ref(env, traj, budget):
    if len(traj) < 3:
        return traj
    best = list(traj); improved = True
    while improved:
        improved = False
        best_obj = env.objective(best)
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                cand = best[:i+1] + best[i+1:j+1][::-1] + best[j+1:]
                if env.objective(cand) < best_obj - 1e-6 and \
                   _chain_energy(env, cand) <= budget:
                    best = cand; best_obj = env.objective(cand); improved = True; break
            if improved:
                break
    return best


def fleet_post_process_ref(env, fleet):
    for k in range(fleet.K):
        fleet.trajs[k] = _two_opt_chain_ref(env, fleet.trajs[k], fleet.Emax_each)
    served = set(j for t in fleet.trajs for j in t)
    changed = True
    while changed:
        changed = False
        best_gain = 1e-6; best_move = None
        for j in range(env.M):
            if j in served:
                continue
            for k in range(fleet.K):
                base_obj = env.objective(fleet.trajs[k])
                for pos in range(len(fleet.trajs[k]) + 1):
                    cand = fleet.trajs[k][:pos] + [j] + fleet.trajs[k][pos:]
                    if _chain_energy(env, cand) > fleet.Emax_each:
                        continue
                    gain = base_obj - env.objective(cand)
                    if gain > best_gain:
                        best_gain = gain; best_move = (k, cand, j)
        if best_move is not None:
            k, cand, j = best_move
            fleet.trajs[k] = cand; served.add(j); fleet.visited[j] = True; changed = True
    return fleet
# ─────────────────────────────────────────────────────────────────────────────


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(CKPT, map_location=dev)
    pol = MultiUAVPolicy(hidden=256, input_dim=MP.INPUT_DIM).to(dev)
    pol.load_state_dict(ck['policy']); pol.eval()
    Ee = P.Emax / K
    rng = np.random.default_rng(2025)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(N)]

    worst = 0.0; n_traj_mismatch = 0
    print(f'ckpt={CKPT}  M={M} K={K}  n={N}')
    print(f'{"seed":>9} {"NEW obj":>10} {"REF obj":>10} {"|diff|":>9}  trajs')
    for s in seeds:
        e1 = Env(M=M, seed=s); f1 = fleet_rollout(pol, e1, K, dev, Emax_each=Ee, greedy=True)
        fleet_post_process(e1, f1)                       # NEW (in-module)
        o_new = f1.fleet_objective()

        e2 = Env(M=M, seed=s); f2 = fleet_rollout(pol, e2, K, dev, Emax_each=Ee, greedy=True)
        fleet_post_process_ref(e2, f2)                   # REF (original)
        o_ref = f2.fleet_objective()

        d = abs(o_new - o_ref); worst = max(worst, d)
        same = [list(a) for a in f1.trajs] == [list(a) for a in f2.trajs]
        n_traj_mismatch += (0 if same else 1)
        print(f'{s:>9} {o_new:>10.3f} {o_ref:>10.3f} {d:>9.2e}  '
              f'{"identical" if same else "DIFFER"}')
        if not same:
            for k in range(K):
                if list(f1.trajs[k]) != list(f2.trajs[k]):
                    print(f'    chain {k}:')
                    print(f'      NEW: {list(f1.trajs[k])}')
                    print(f'      REF: {list(f2.trajs[k])}')
                    break   # first differing chain only

    print(f'\nworst |obj diff| = {worst:.2e}   trajectory mismatches: {n_traj_mismatch}/{N}')
    if worst < 1e-9 and n_traj_mismatch == 0:
        print('VERDICT: bit-identical. The rewrite is safe; the -198 vs -190 was a '
              'checkpoint/seed confound (bench_speed --seed bug), not pp.')
    else:
        print('VERDICT: NOT identical. Paste this output back and I will localize the '
              'divergence (2-opt vs insertion) and fix it.')


if __name__ == '__main__':
    main()