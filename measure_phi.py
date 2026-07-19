"""
measure_phi.py  --  how big is the deconfliction penalty, really?
==================================================================
THE GATING MEASUREMENT. The conflict-aware direction (co-optimising routing with
corridor separation) is only worth building if Phi is a meaningful share of the
objective. Phi = theta1 * sum_k o_k * W_tot^k  (Eq. 26) is already implemented
exactly as `aoi_penalty` in deconfliction_schedule.

This runs SA (near-optimal, now route-returning) on shared-seed instances, wraps
each solution in a FleetState, schedules launches, and reports:

    J_route            SA routing objective          (Phi=0, intra-mission)
    Phi                theta1 * weighted_delay       (common t=0 penalty)
    Phi_pct            100 * Phi / |J_route|         <-- THE NUMBER
    J_FINAL            J_route + Phi
    makespan           max launch offset (s)
    total_delay        sum of offsets (s)
    conflicts_left     residual conflicts (must be 0)

READ:
  Phi_pct >= 5% and growing with K -> real headroom; conflict-aware search is
      worth building, and the K* curve genuinely shifts under t=0 (Corollary 1).
  Phi_pct ~ 1-2%             -> the margin over route-then-deconflict is inside
      solver noise; do NOT spend weeks on co-optimisation. Report deconfliction
      as a separate scheduling cost (makespan/delay) per Prop 5 and move on.

Note this measures Phi on routes chosen WITHOUT any conflict awareness, which is
exactly right: it is the penalty the decoupled pipeline pays, i.e. the ceiling on
what co-optimisation could ever recover.

Run (small slice first -- the anchor sweep may be using the CPU):
    python measure_phi.py --M 100 --K 2 3 4 5 6 --instances 6 --iters 1500

Then, if promising, widen:
    python measure_phi.py --M 50 100 200 --K 2 3 4 5 6 8 --instances 12 --iters 2000
"""
import os, csv, argparse
import numpy as np

from uav_aoi_solver import P, Env
from multi_uav_solver import FleetState, deconfliction_schedule
from compare_baseline import gen, INSTANCE_SEED
from sa_routes import sa_best_with_routes


def fleet_from_trajs(env, trajs, Emax_each):
    """Wrap SA's plain trajectory lists in a FleetState so the deconfliction
    layer (which needs env, W_cum, trajs) can consume them."""
    K = len(trajs)
    fs = FleetState(env, K, Emax_each)
    for k in range(K):
        fs.trajs[k] = list(trajs[k])
        for j in trajs[k]:
            fs.visited[j] = True
    # W_cum[k] = total priority of chain k (what Phi weights the offset by)
    for k in range(K):
        fs.W_cum[k] = float(sum(env.wi[j] for j in trajs[k]))
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[100])
    ap.add_argument('--K', type=int, nargs='+', default=[2, 3, 4, 5, 6])
    ap.add_argument('--Emax', type=float, default=P.Emax)
    ap.add_argument('--instances', type=int, default=6)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--restarts', type=int, default=1)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--delta', type=float, default=25.0, help='safety separation (m)')
    ap.add_argument('--dt', type=float, default=0.25)
    ap.add_argument('--out-dir', default='phi_measure')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 1e7)) for _ in range(a.instances)]

    print('=' * 86)
    print(f'  Phi magnitude on SA solutions | Emax={a.Emax:.0f} (split /K) '
          f'| delta={a.delta}m | n={a.instances}')
    print(f'  SA iters={a.iters} restarts={a.restarts} | seed={a.seed}')
    print('=' * 86)
    print(f'{"M":>4} {"K":>2} {"J_route":>10} {"Phi":>8} {"Phi%":>7} {"J_FINAL":>10} '
          f'{"makespan":>9} {"conf":>5}')

    rows = []
    for M in a.M:
        for K in a.K:
            Ee = a.Emax / K
            Js, Phis, mks, tds, confs = [], [], [], [], []
            for s in seeds:
                pos, wi, tcd = gen(M, s)
                Jr, trajs = sa_best_with_routes(pos, wi, tcd, K, Ee, M,
                                                a.iters, a.restarts)
                env = Env(M=M, seed=s)
                fs = fleet_from_trajs(env, trajs, Ee)
                sched = deconfliction_schedule(fs, delta=a.delta, dt=a.dt,
                                               optimize_order=True, verify=True)
                Js.append(Jr)
                Phis.append(float(sched['aoi_penalty']))
                mks.append(float(sched['makespan']))
                tds.append(float(sched['total_delay']))
                confs.append(int(sched['conflicts_left']))

            J = float(np.mean(Js)); Ph = float(np.mean(Phis))
            pct = 100 * Ph / abs(J) if abs(J) > 1e-9 else float('nan')
            row = dict(M=M, K=K, Emax=a.Emax, Ee=round(Ee, 1),
                       instances=a.instances, delta=a.delta,
                       J_route=round(J, 3), Phi=round(Ph, 4),
                       Phi_pct=round(pct, 2), J_FINAL=round(J + Ph, 3),
                       Phi_std=round(float(np.std(Phis)), 4),
                       makespan_s=round(float(np.mean(mks)), 2),
                       total_delay_s=round(float(np.mean(tds)), 2),
                       conflicts_left=int(sum(confs)))
            rows.append(row)
            print(f'{M:>4} {K:>2} {J:>10.2f} {Ph:>8.3f} {pct:>6.2f}% {J+Ph:>10.2f} '
                  f'{np.mean(mks):>9.2f} {sum(confs):>5}')

    out = os.path.join(a.out_dir, 'phi_measure.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    pcts = [r['Phi_pct'] for r in rows if r['Phi_pct'] == r['Phi_pct']]
    print('\n' + '=' * 86)
    print(f'  Phi ranges {min(pcts):.2f}% -- {max(pcts):.2f}% of |J_route|')
    if max(pcts) >= 5.0:
        print('  => REAL HEADROOM. Conflict-aware co-optimisation is worth building,')
        print('     and K* genuinely shifts under t=0 (Corollary 1).')
    elif max(pcts) >= 2.0:
        print('  => MARGINAL. Co-optimisation could recover only a few % of a few %.')
        print('     Check whether Phi grows with K / smaller delta before committing.')
    else:
        print('  => TOO SMALL. Do not build co-optimisation on this. Report')
        print('     deconfliction as separate scheduling cost (Prop 5) and move on.')
    print('  (conflicts_left must be 0 everywhere; nonzero = scheduling bug.)')
    print(f'  wrote {out}')
    print('=' * 86)


if __name__ == '__main__':
    main()
