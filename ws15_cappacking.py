"""
ws15_cappacking.py  --  is K* an artifact of greedy's cap-packing?
===================================================================
THE QUESTION. `greedy_init` fills chains to 98-100% of the per-UAV budget Ee, and
at M>=100 SA returns greedy's solution UNCHANGED (30/30 at M=100,K=4). The energy
budget is exactly the variable K* depends on. So a reviewer will ask: is the
fleet-sizing law a real property of the problem, or an artifact of a greedy
heuristic packing routes to capacity?

THE TEST. Existing exact-DFS validation used budget=9000 J at M=7-9, where the
budget is SLACK and cap-packing never happens. That validates the solvers in the
wrong regime. This script instead:

  1. Sweeps the budget DOWN at small M until greedy's chain fill ratio
     (chain_energy / Ee) enters the cap-packed band (>= --pack-lo, default 0.98).
  2. At those cap-packed budgets, compares greedy and SA against the EXACT
     optimum (exact_single, DFS with energy pruning).
  3. Reports the gap. Small gap => cap-packing does not produce bad solutions =>
     K* is safe. Large gap => the law needs re-examination.

K=1 ONLY. `exact_single` is single-UAV. That is the honest scope of this test:
it shows cap-packing per se does not break near-optimality. It does not certify
multi-UAV partitioning (that needs the MILP in WS80).

OUTPUT. `ws15_cappacking/cappacking.csv` plus a verdict line.

Run:
    python ws15_cappacking.py --M 8 9 --instances 12
    python ws15_cappacking.py --M 7 8 9 --instances 20 --iters 4000   # tighter
"""
import os, csv, argparse
import numpy as np

from compare_baseline import gen, greedy_init, chain_energy, fleet_obj
from fleet_optimality_gap import exact_single, sa_best


def fill_ratio(traj, pos, tcd, Ee):
    """Fraction of the budget the chain actually consumes."""
    if not traj:
        return 0.0
    return chain_energy(traj, pos, tcd) / Ee


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, nargs='+', default=[8, 9])
    ap.add_argument('--instances', type=int, default=12)
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--restarts', type=int, default=3)
    ap.add_argument('--seed', type=int, default=2025)
    ap.add_argument('--budgets', type=float, nargs='+', default=None,
                    help='explicit budgets; default sweeps a log grid')
    ap.add_argument('--n-budgets', type=int, default=10)
    ap.add_argument('--b-lo', type=float, default=2000.0)
    ap.add_argument('--b-hi', type=float, default=14000.0)
    ap.add_argument('--pack-lo', type=float, default=0.98,
                    help='fill ratio at/above which a cell counts as cap-packed')
    ap.add_argument('--out-dir', default='ws15_cappacking')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    budgets = (a.budgets if a.budgets is not None
               else list(np.unique(np.round(
                   np.logspace(np.log10(a.b_lo), np.log10(a.b_hi), a.n_budgets)))))

    # meta-seed semantics, matching compare_baseline.py:111-113
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]

    print('=' * 92)
    print(f'  WS15 cap-packing check | M={a.M} | budgets={[int(b) for b in budgets]}')
    print(f'  instances={a.instances} (meta-seed {a.seed}) | SA iters={a.iters} '
          f'restarts={a.restarts}')
    print(f'  cap-packed threshold: fill >= {a.pack_lo}')
    print('=' * 92)
    print(f'{"M":>3} {"Ee":>7} {"fill":>6} {"packed":>7} {"J_exact":>10} {"J_greedy":>10} '
          f'{"J_SA":>10} {"g_gap%":>7} {"sa_gap%":>7}')

    rows = []
    for M in a.M:
        for Ee in budgets:
            fills, Jex, Jgr, Jsa = [], [], [], []
            for s in seeds:
                pos, wi, tcd = gen(M, s)
                # greedy (K=1) — the constructor whose packing is under suspicion
                gtr, _ = greedy_init(pos, wi, tcd, 1, Ee, M)
                fills.append(fill_ratio(gtr[0], pos, tcd, Ee))
                Jgr.append(fleet_obj(gtr, pos, wi, tcd))
                # exact optimum
                Jex.append(exact_single(pos, wi, tcd, Ee, M))
                # SA
                Jsa.append(sa_best(pos, wi, tcd, 1, Ee, M, a.iters, a.restarts))

            f = float(np.mean(fills))
            je, jg, js = float(np.mean(Jex)), float(np.mean(Jgr)), float(np.mean(Jsa))
            packed = f >= a.pack_lo
            ggap = 100 * (jg - je) / abs(je) if abs(je) > 1e-9 else float('nan')
            sgap = 100 * (js - je) / abs(je) if abs(je) > 1e-9 else float('nan')
            rows.append(dict(M=M, Ee=round(float(Ee), 1), fill=round(f, 4),
                             cap_packed=int(packed), J_exact=round(je, 4),
                             J_greedy=round(jg, 4), J_SA=round(js, 4),
                             greedy_gap_pct=round(ggap, 3),
                             sa_gap_pct=round(sgap, 3),
                             instances=a.instances))
            print(f'{M:>3} {Ee:>7.0f} {f:>6.3f} {"YES" if packed else "no":>7} '
                  f'{je:>10.3f} {jg:>10.3f} {js:>10.3f} {ggap:>7.2f} {sgap:>7.2f}')

    out = os.path.join(a.out_dir, 'cappacking.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    packed_rows = [r for r in rows if r['cap_packed']]
    slack_rows = [r for r in rows if not r['cap_packed']]

    print('\n' + '=' * 92)
    if not packed_rows:
        print('  NO CAP-PACKED CELLS REACHED. Lower --b-lo (or raise --pack-lo) and')
        print('  re-run; the test has not been performed.')
    else:
        gp = float(np.mean([r['greedy_gap_pct'] for r in packed_rows]))
        sp = float(np.mean([r['sa_gap_pct'] for r in packed_rows]))
        gwp = float(np.max([r['greedy_gap_pct'] for r in packed_rows]))
        print(f'  CAP-PACKED cells ({len(packed_rows)}): greedy mean gap {gp:.2f}% '
              f'(worst {gwp:.2f}%), SA mean gap {sp:.2f}%')
        if slack_rows:
            gs = float(np.mean([r['greedy_gap_pct'] for r in slack_rows]))
            print(f'  SLACK cells ({len(slack_rows)}):      greedy mean gap {gs:.2f}%')
            print(f'  -> difference packed vs slack: {gp - gs:+.2f} pp')
        print()
        if gwp < 3.0:
            print('  VERDICT: greedy stays near-optimal WHEN CAP-PACKED.')
            print('  => cap-packing does not produce bad solutions; K* is not an')
            print('     artifact of it. Report this as the WS15 answer.')
        elif gwp < 8.0:
            print('  VERDICT: MODERATE degradation under cap-packing. Report the gap')
            print('     honestly and check whether it grows with M before claiming K*.')
        else:
            print('  VERDICT: LARGE degradation under cap-packing. K* may be biased by')
            print('     the constructor. Escalate — this needs the MILP (WS80) at')
            print('     larger M before the law can be claimed.')
    print(f'\n  SCOPE: K=1 only (exact_single is single-UAV). Multi-UAV partitioning')
    print( '  is NOT certified here — that requires WS80.')
    print(f'  wrote {out}')
    print('=' * 92)


if __name__ == '__main__':
    main()