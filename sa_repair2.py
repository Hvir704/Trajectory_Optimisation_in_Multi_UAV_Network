
"""
sa_repair2.py  --  COST-AWARE energy repair, with move-level diagnostics.
==========================================================================
WHY v2. `sa_repair.py` (v1) paired a random removal with a random insertion. At a
cap-packed solution the inserted node must fit in the exact energy slot freed by
the removed one, and node costs vary widely (payloads 0.5-5 Mbit, arbitrary
distances), so random pairing essentially never lands inside the budget. Measured
outcome: freeze 21/24 both arms (identical to baseline), gains -0.01 / +0.03 /
-0.66 % -- two NEGATIVE, i.e. the repair budget was pure waste.

v2 changes the operator from random to TARGETED:

  op5' cost-aware eject-insert
       1. pick the insertion candidate j FIRST (biased to high w_j)
       2. compute its true marginal insertion cost at the best position
       3. eject the CHEAPEST set of served nodes that frees >= that much energy
          (greedy by energy-freed-per-priority-lost, so we shed low-value nodes)
       4. accept only if the net swap is energy-feasible
     This is a k-for-1 exchange: it can remove SEVERAL cheap nodes to admit one
     valuable one, which the 1-for-1 version structurally could not do.

  op6' value-directed ejection chain
       swap node A (low w, high energy) in chain k for node B (high w, low
       energy) in chain k2, choosing A and B by the value/energy ratio rather
       than uniformly.

DIAGNOSTICS. v1's failure was invisible until the freeze counts were compared.
v2 counts, per cell: repair moves ATTEMPTED, moves that produced a FEASIBLE
candidate, and moves ACCEPTED. If feasible ~ 0 the operator is still not firing
and no conclusion about K* should be drawn from the run.

READ THE OUTPUT IN THIS ORDER:
  1. rep_feas -- if ~0, the operator is dead; nothing else in the row means
     anything. Try --repair-p 0.5 --iters 8000, else conclude the neighbourhood
     is genuinely inescapable.
  2. rep-frz vs base-frz -- did the freeze actually break?
  3. argmin -- only meaningful once (1) and (2) show real search happened.

Run:
    python sa_repair2.py --M 100 --K 3 4 5 --Emax 50000 --instances 8 --iters 4000
"""
import os, csv, time, argparse
import numpy as np

from compare_baseline import (gen, greedy_init, fleet_obj, feasible,
                              chain_energy, tf, PF, PH, HOME, EMAX, INSTANCE_SEED)
from compare_baseline import sa as sa_baseline


def _ins_cost(chain, j, pos, tcd, p):
    """Energy delta of inserting node j at position p of chain (list of ids)."""
    a = HOME if p == 0 else pos[chain[p - 1]]
    b = HOME if p == len(chain) else pos[chain[p]]
    return PF * (tf(a, pos[j]) + tf(pos[j], b) - tf(a, b)) + PH * tcd[j]


def _best_ins(chain, j, pos, tcd):
    """(cheapest energy delta, position) for inserting j into chain."""
    best, bp = None, 0
    for p in range(len(chain) + 1):
        d = _ins_cost(chain, j, pos, tcd, p)
        if best is None or d < best:
            best, bp = d, p
    return best, bp


def _rem_gain(chain, i, pos, tcd):
    """Energy freed by removing the node at index i of chain."""
    a = HOME if i == 0 else pos[chain[i - 1]]
    b = HOME if i == len(chain) - 1 else pos[chain[i + 1]]
    j = chain[i]
    return PF * (tf(a, pos[j]) + tf(pos[j], b) - tf(a, b)) + PH * tcd[j]


def sa_repaired2(pos, wi, tcd, K, Ee, M, iters, seed, repair_p=0.35, stats=None):
    rng = np.random.default_rng(seed)
    trajs, served = greedy_init(pos, wi, tcd, K, Ee, M)
    cur = fleet_obj(trajs, pos, wi, tcd)
    best = cur
    T0, T1 = abs(cur) * 0.05 + 1e-3, 1e-4
    att = feas = acc = 0

    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(iters - 1, 1))
        nt = [t[:] for t in trajs]
        uns = [j for j in range(M) if j not in served]
        is_repair = (rng.random() < repair_p) and uns and any(nt)

        if is_repair:
            att += 1
            if rng.random() < 0.6:
                # ---- op5' cost-aware eject-insert (k-for-1) ----
                # 1. choose insertion candidate, biased toward high priority
                w_u = np.array([wi[j] for j in uns], float)
                pr = w_u / w_u.sum()
                j = int(rng.choice(uns, p=pr))
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                need, p = _best_ins(nt[k], j, pos, tcd)
                slack = Ee - chain_energy(nt[k], pos, tcd)
                # 2. eject cheapest-value nodes until enough energy is freed
                if need > slack:
                    cand = []
                    for i in range(len(nt[k])):
                        gfree = _rem_gain(nt[k], i, pos, tcd)
                        if gfree > 1e-9:
                            cand.append((wi[nt[k][i]] / gfree, i, gfree))
                    cand.sort()                    # lowest value-per-joule first
                    freed, drop = 0.0, []
                    for _, i, gfree in cand:
                        drop.append(i); freed += gfree
                        if slack + freed >= need:
                            break
                    if slack + freed < need:
                        continue                   # cannot free enough; skip
                    for i in sorted(drop, reverse=True):
                        del nt[k][i]
                    need, p = _best_ins(nt[k], j, pos, tcd)
                    if chain_energy(nt[k], pos, tcd) + need > Ee + 1e-9:
                        continue
                nt[k] = nt[k][:p] + [j] + nt[k][p:]
            else:
                # ---- op6' value-directed ejection chain ----
                ne = [i for i in range(K) if nt[i]]
                if len(ne) < 2:
                    continue
                k, k2 = (int(x) for x in rng.choice(ne, 2, replace=False))
                # A = worst value-per-joule in k ; B = best in k2
                def ratios(kk):
                    out = []
                    for i in range(len(nt[kk])):
                        g = _rem_gain(nt[kk], i, pos, tcd)
                        if g > 1e-9:
                            out.append((wi[nt[kk][i]] / g, i))
                    return out
                ra, rb = ratios(k), ratios(k2)
                if not ra or not rb:
                    continue
                ia = min(ra)[1]; ib = max(rb)[1]
                A, B = nt[k][ia], nt[k2][ib]
                del nt[k][ia]; del nt[k2][ib]
                dB, pB = _best_ins(nt[k], B, pos, tcd)
                dA, pA = _best_ins(nt[k2], A, pos, tcd)
                if (chain_energy(nt[k], pos, tcd) + dB > Ee + 1e-9 or
                        chain_energy(nt[k2], pos, tcd) + dA > Ee + 1e-9):
                    continue
                nt[k] = nt[k][:pB] + [B] + nt[k][pB:]
                nt[k2] = nt[k2][:pA] + [A] + nt[k2][pA:]
        else:
            op = rng.integers(0, 5)
            if op == 0 and uns:
                j = int(rng.choice(uns)); k = int(rng.integers(0, K))
                p = int(rng.integers(0, len(nt[k]) + 1))
                nt[k] = nt[k][:p] + [j] + nt[k][p:]
            elif op == 1 and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k]))); del nt[k][i]
            elif op == 2 and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k]))); j = nt[k][i]; del nt[k][i]
                k2 = int(rng.integers(0, K)); p = int(rng.integers(0, len(nt[k2]) + 1))
                nt[k2] = nt[k2][:p] + [j] + nt[k2][p:]
            elif op == 3 and any(len(t) >= 2 for t in nt):
                k = int(rng.choice([i for i in range(K) if len(nt[i]) >= 2]))
                a2, b2 = sorted(rng.choice(len(nt[k]), 2, replace=False))
                nt[k][a2:b2 + 1] = nt[k][a2:b2 + 1][::-1]
            elif op == 4 and uns and any(nt):
                k = int(rng.choice([i for i in range(K) if nt[i]]))
                i = int(rng.integers(0, len(nt[k])))
                nt[k][i] = int(rng.choice(uns))

        if not feasible(nt, pos, tcd, Ee):
            continue
        if is_repair:
            feas += 1
        o = fleet_obj(nt, pos, wi, tcd); d = o - cur
        if d < 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
            if is_repair:
                acc += 1
            trajs = nt; cur = o; served = set(x for t in nt for x in t)
            if o < best:
                best = o
    if stats is not None:
        stats['att'] += att; stats['feas'] += feas; stats['acc'] += acc
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--K', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--Emax', type=float, default=EMAX)
    ap.add_argument('--instances', type=int, default=8)
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--restarts', type=int, default=2)
    ap.add_argument('--repair-p', type=float, default=0.35)
    ap.add_argument('--seed', type=int, default=INSTANCE_SEED)
    ap.add_argument('--out-dir', default='sa_repair2')
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    seeds = [int(rng.integers(0, 10_000_000)) for _ in range(a.instances)]
    M = a.M

    print('=' * 104)
    print(f'  COST-AWARE SA repair | M={M} Emax={a.Emax:.0f} K={a.K} '
          f'| inst={a.instances} iters={a.iters} rp={a.repair_p}')
    print('=' * 104)
    print(f'{"K":>2} {"Ee":>7} {"greedy":>10} {"SA_base":>10} {"SA_rep2":>10} '
          f'{"base-frz":>9} {"rep-frz":>8} {"gain%":>7} '
          f'{"rep_att":>8} {"rep_feas":>9} {"rep_acc":>8} {"sec":>6}')

    rows = []
    for K in a.K:
        Ee = a.Emax / K
        g, sb, sr, fz_b, fz_r = [], [], [], 0, 0
        st = dict(att=0, feas=0, acc=0)
        t0 = time.time()
        for s in seeds:
            pos, wi, tcd = gen(M, s)
            gtr, _ = greedy_init(pos, wi, tcd, K, Ee, M)
            Jg = fleet_obj(gtr, pos, wi, tcd)
            Jb = min(sa_baseline(pos, wi, tcd, K, Ee, M, a.iters, r)
                     for r in range(a.restarts))
            Jr = min(sa_repaired2(pos, wi, tcd, K, Ee, M, a.iters, r,
                                  a.repair_p, st) for r in range(a.restarts))
            g.append(Jg); sb.append(Jb); sr.append(Jr)
            fz_b += int(abs(Jb - Jg) < 1e-9)
            fz_r += int(abs(Jr - Jg) < 1e-9)
        dt = time.time() - t0
        G, B, R = float(np.mean(g)), float(np.mean(sb)), float(np.mean(sr))
        gain = 100 * (B - R) / abs(B) if abs(B) > 1e-9 else 0.0
        rows.append(dict(M=M, K=K, Ee=round(Ee, 1), greedy=round(G, 4),
                         SA_base=round(B, 4), SA_repair2=round(R, 4),
                         base_frozen=fz_b, repair_frozen=fz_r,
                         repair_gain_pct=round(gain, 3),
                         rep_attempted=st['att'], rep_feasible=st['feas'],
                         rep_accepted=st['acc'], instances=a.instances,
                         iters=a.iters, sec=round(dt, 1)))
        print(f'{K:>2} {Ee:>7.0f} {G:>10.3f} {B:>10.3f} {R:>10.3f} '
              f'{fz_b:>4}/{a.instances:<4} {fz_r:>3}/{a.instances:<4} {gain:>7.2f} '
              f'{st["att"]:>8} {st["feas"]:>9} {st["acc"]:>8} {dt:>6.1f}')

    out = os.path.join(a.out_dir, f'sa_repair2_M{M}_E{int(a.Emax)}.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    tot_att = sum(r['rep_attempted'] for r in rows)
    tot_feas = sum(r['rep_feasible'] for r in rows)
    tot_acc = sum(r['rep_accepted'] for r in rows)
    kb = min(rows, key=lambda r: r['SA_base'])['K']
    kr = min(rows, key=lambda r: r['SA_repair2'])['K']

    print('\n' + '=' * 104)
    print(f'  repair moves: attempted {tot_att}, feasible {tot_feas} '
          f'({100*tot_feas/max(tot_att,1):.1f}%), accepted {tot_acc} '
          f'({100*tot_acc/max(tot_att,1):.1f}%)')
    if tot_feas < 0.02 * max(tot_att, 1):
        print('\n  OPERATOR STILL DEAD (<2% feasible). Do NOT read the argmin.')
        print('  Either retry with --repair-p 0.5 --iters 8000, or conclude that')
        print('  cap-packed optima are genuinely inescapable under exchange moves')
        print('  and report the freeze as a STRUCTURAL finding with this evidence.')
    else:
        print(f'\n  argmin K  baseline = {kb}   repaired = {kr}')
        if kb == kr:
            print('  => K* DOES NOT MOVE under a working repair operator.')
            print('     This is now a REAL test. Report the freeze as a documented')
            print('     limitation with evidence it does not affect the argmin.')
        else:
            print('  => K* MOVES. Re-run the K* grid with repaired SA before')
            print('     claiming the law. Do it now, not near the deadline.')
    print(f'  wrote {out}')
    print('=' * 104)


if __name__ == '__main__':
    main()