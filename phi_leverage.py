"""
phi_leverage.py  --  is Phi CONTROLLABLE, or is it fixed by geometry?
======================================================================
measure_phi.py established that Phi is large (8-20% at delta=50m, Emax=100k).
But Phi is a CEILING, not a prize: co-optimisation can only recover the part
that responds to decisions. This probe measures how much that is, WITHOUT
building a solver -- so you learn whether the conflict-aware direction has
leverage before spending weeks on it.

Three levers, cheapest first:

  L1  SCHEDULING SLACK (free, already implemented)
      Best vs worst launch ORDER on the SAME routes. deconfliction_schedule
      already enumerates all K! orders for K<=6; we take min and max weighted
      delay. This is the gain already captured by the existing layer -- it is
      NOT available to a router (it's post-hoc). Reported as context: if the
      order spread is huge, scheduling matters more than routing.

  L2  ASSIGNMENT SENSITIVITY (the key number)
      Randomly swap nodes BETWEEN UAVs (keeping feasibility), re-optimise the
      schedule, and record the Phi spread across perturbations. If small routing
      changes move Phi a lot, a router that *aims* at low Phi has real leverage.
      If Phi barely moves, it is geometry-bound and co-optimisation is futile.
      Reported as (Phi_min_found - Phi_base) = recoverable-by-random-search.

  L3  ROUTING COST OF THAT GAIN (the trade-off)
      For the best-Phi perturbation found, how much did J_route worsen? The net
      dJ_FINAL = dJ_route + dPhi tells you whether the trade is actually
      profitable. A Phi gain paid for by a larger routing loss is worthless.

READ:
  net_gain_pct >= ~3%  -> co-optimisation has real leverage; build the solver.
      (random perturbation is a WEAK searcher; a real LNS should beat it
       comfortably, so this is a conservative lower bound on the achievable gain)
  net_gain_pct ~ 0-1%  -> Phi is geometry-bound. Do NOT build co-optimisation.
      Report Phi as the K*-determining mechanism (already a solid result) and
      stop. Note the loss column: if dJ_route always cancels dPhi, that is a
      structural finding worth stating explicitly in the paper.

Run:
    python phi_leverage.py --M 100 --K 6 8 --Emax 100000 --delta 50 ^
        --instances 4 --perturb 40 --iters 1500
"""
import os, csv, argparse
import numpy as np

from uav_aoi_solver import P, Env
from multi_uav_solver import deconfliction_schedule
from compare_baseline import gen, INSTANCE_SEED, fleet_obj, chain_energy
from sa_routes import sa_best_with_routes
from measure_phi import fleet_from_trajs


def phi_of(env, trajs, Ee, delta, dt, optimize_order=True):
    fs = fleet_from_trajs(env, trajs, Ee)
    s = deconfliction_schedule(fs, delta=delta, dt=dt,
                               optimize_order=optimize_order, verify=False)
    return float(s['aoi_penalty']), float(s['makespan'])


def feasible_all(trajs, pos, tcd, Ee):
    return all(chain_energy(t, pos, tcd) <= Ee + 1e-6 for t in trajs)


def perturb(trajs, rng, K):
    """Move or swap nodes BETWEEN UAVs (routing-level change, not scheduling)."""
    nt = [list(t) for t in trajs]
    nonempty = [k for k in range(K) if nt[k]]
    if not nonempty:
        return nt
    if rng.random() < 0.5 and len(nonempty) >= 2:          # swap one node each way
        a, b = rng.choice(nonempty, 2, replace=False)
        ia = int(rng.integers(0, len(nt[a]))); ib = int(rng.integers(0, len(nt[b])))
        nt[a][ia], nt[b][ib] = nt[b][ib], nt[a][ia]
    else:                                                   # move one node across
        a = int(rng.choice(nonempty)); b = int(rng.integers(0, K))
        if a == b or not nt[a]:
            return nt
        ia = int(rng.integers(0, len(nt[a])))
        j = nt[a].pop(ia)
        p = int(rng.integers(0, len(nt[b]) + 1))
        nt[b] = nt[b][:p] + [j] + nt[b][p:]
    return nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--K', type=int, nargs='+', default=[6, 8])
    ap.add_argument('--Emax', type=float, default=100000.0)
    ap.add_argument('--delta', type=float, default=50.0)
    ap.add_argument('--dt', type=float, default=0.25)
    ap.add_argument('--instances', type=int, default=4)
    ap.add_argument('--perturb', type=int, default=40)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--out-dir', default='phi_leverage')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng0 = np.random.default_rng(a.seed)
    seeds = [int(rng0.integers(0, 1e7)) for _ in range(a.instances)]
    M = a.M

    print('=' * 92)
    print(f'  Phi leverage | M={M} Emax={a.Emax:.0f} delta={a.delta}m '
          f'| {a.perturb} perturbations x {a.instances} instances')
    print('=' * 92)

    rows = []
    for K in a.K:
        Ee = a.Emax / K
        ordspread, dphis, djr, dnets = [], [], [], []
        base_J, base_P = [], []
        for s in seeds:
            pos, wi, tcd = gen(M, s)
            J0, trajs = sa_best_with_routes(pos, wi, tcd, K, Ee, M, a.iters, a.restarts)
            env = Env(M=M, seed=s)

            # L1: scheduling slack = best-order vs greedy-order Phi
            p_opt, _ = phi_of(env, trajs, Ee, a.delta, a.dt, optimize_order=True)
            p_gre, _ = phi_of(env, trajs, Ee, a.delta, a.dt, optimize_order=False)
            ordspread.append(p_gre - p_opt)

            # L2/L3: routing perturbations, scored on J_FINAL
            best_net = 0.0; best_dphi = 0.0; best_djr = 0.0
            rng = np.random.default_rng(s)
            for _ in range(a.perturb):
                nt = perturb(trajs, rng, K)
                if not feasible_all(nt, pos, tcd, Ee):
                    continue
                Jn = fleet_obj(nt, pos, wi, tcd)
                pn, _ = phi_of(env, nt, Ee, a.delta, a.dt, optimize_order=True)
                d_phi = pn - p_opt          # negative = Phi improved
                d_jr = Jn - J0              # negative = routing improved
                net = d_jr + d_phi          # negative = J_FINAL improved
                if net < best_net:
                    best_net, best_dphi, best_djr = net, d_phi, d_jr

            base_J.append(J0); base_P.append(p_opt)
            dphis.append(best_dphi); djr.append(best_djr); dnets.append(best_net)

        J = float(np.mean(base_J)); Pb = float(np.mean(base_P))
        JF = J + Pb
        net = float(np.mean(dnets))
        row = dict(M=M, K=K, Emax=a.Emax, delta=a.delta, instances=a.instances,
                   perturb=a.perturb,
                   J_route=round(J, 3), Phi=round(Pb, 3),
                   J_FINAL=round(JF, 3),
                   Phi_pct=round(100 * Pb / abs(J), 2),
                   order_spread=round(float(np.mean(ordspread)), 3),
                   best_dPhi=round(float(np.mean(dphis)), 3),
                   best_dJroute=round(float(np.mean(djr)), 3),
                   best_dNet=round(net, 3),
                   net_gain_pct=round(100 * abs(net) / abs(JF), 2))
        rows.append(row)

        print(f'\nK={K}  Emax_each={Ee:.0f}')
        print(f'  base           J_route={J:>9.2f}  Phi={Pb:>7.2f} ({row["Phi_pct"]:.1f}%)'
              f'  J_FINAL={JF:>9.2f}')
        print(f'  L1 order slack  {row["order_spread"]:>7.2f}   '
              f'(Phi already saved by best-of-orderings vs greedy order)')
        print(f'  L2 best dPhi    {row["best_dPhi"]:>7.2f}   (from random routing perturbation)')
        print(f'  L3 dJ_route     {row["best_dJroute"]:>7.2f}   (cost paid for it)')
        print(f'  => net dJ_FINAL {row["best_dNet"]:>7.2f}   = {row["net_gain_pct"]:.2f}% of J_FINAL')

    out = os.path.join(a.out_dir, 'phi_leverage.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    g = max(r['net_gain_pct'] for r in rows)
    print('\n' + '=' * 92)
    print(f'  best net gain from RANDOM perturbation: {g:.2f}% of J_FINAL')
    if g >= 3.0:
        print('  => LEVERAGE EXISTS. Random search is weak; a directed LNS should beat')
        print('     this comfortably. Build the conflict-aware solver.')
    elif g >= 1.0:
        print('  => MARGINAL. A directed search may do 2-3x better than random, so a')
        print('     real gain is plausible but not assured. Consider a bigger --perturb')
        print('     before committing.')
    else:
        print('  => NO LEVERAGE. Phi is geometry-bound: routing changes do not move it,')
        print('     or the routing cost cancels the Phi gain. Do NOT build the solver.')
        print('     Report Phi as the K*-determining mechanism -- that result stands.')
    print(f'  wrote {out}')
    print('=' * 92)


if __name__ == '__main__':
    main()
