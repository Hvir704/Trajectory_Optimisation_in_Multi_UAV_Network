"""
compare_final.py
================================================================================
Reproducible FINAL-vs-FINAL comparison of the greedy and beam deconfliction grids.
Reads both resume stores directly (no hardcoded numbers) and reports, per (M,K):

    routing  : greedy+pp vs beam+pp fleet_objective        (dRoute = beam - greedy)
    penalty  : greedy vs beam deconfliction AoI penalty    (dPen  = beam - greedy)
    FINAL    : greedy vs beam collision-avoided objective  (dFINAL = beam - greedy)
    augmented: per-cell better FINAL, and which method wins

dFINAL < 0 means beam's routing gain survives deconfliction at that cell. The
'augmented' column is the deployable per-cell choice (keep whichever method gives
the better collision-avoided objective) and never regresses vs greedy.

Both grids must be on the SAME instance set (same INSTANCE_SEED and --instances);
the script checks instances / n_seeds / residual conflicts and warns on mismatch,
since that silently invalidates the comparison.

RUN:
    python compare_final.py
    python compare_final.py --greedy-dir deconflict_out --beam-dir deconflict_beam_out
    python compare_final.py --out compare_out       # also write table + csv there
"""
from __future__ import annotations
import os, json, argparse
from collections import defaultdict


def load_store(d):
    """Return {(M,K): row} from a deconfliction _deconflict_store.jsonl under dir d."""
    path = os.path.join(d, '_deconflict_store.jsonl')
    if not os.path.exists(path):
        raise SystemExit(f"no store at {path}")
    out = {}
    for line in open(path):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[(r['M'], r['K'])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--greedy-dir', default='deconflict_out')
    ap.add_argument('--beam-dir',   default='deconflict_beam_out')
    ap.add_argument('--out', default=None, help='dir to also write table + csv into')
    args = ap.parse_args()

    G = load_store(args.greedy_dir)
    B = load_store(args.beam_dir)
    keys = sorted(set(G) & set(B))
    only_g = sorted(set(G) - set(B)); only_b = sorted(set(B) - set(G))
    if only_g: print(f"[warn] cells only in greedy store: {only_g}")
    if only_b: print(f"[warn] cells only in beam store:   {only_b}")

    # ---- validity checks: same instances / seeds, no residual conflicts ----
    warns = []
    for k in keys:
        g, b = G[k], B[k]
        if g.get('instances') != b.get('instances'):
            warns.append(f"  {k}: instances greedy={g.get('instances')} beam={b.get('instances')} (NOT comparable)")
        if g.get('n_seeds') != b.get('n_seeds'):
            warns.append(f"  {k}: n_seeds greedy={g.get('n_seeds')} beam={b.get('n_seeds')}")
        if g.get('conflicts_left', 0) or b.get('conflicts_left', 0):
            warns.append(f"  {k}: residual conflicts greedy={g.get('conflicts_left')} beam={b.get('conflicts_left')}")
    if warns:
        print("[warn] comparison-validity issues:"); print("\n".join(warns))
    else:
        n = G[keys[0]].get('instances'); print(f"[ok] both grids: {len(keys)} cells, {n} instances, conf=0, seeds matched")

    # ---- per-cell comparison ----
    lines, byK, worse = [], defaultdict(list), []
    header = (f"{'M':>4}{'K':>3} | {'gRoute':>9}{'bRoute':>9}{'dRoute':>8} |"
              f"{'gPen':>7}{'bPen':>7}{'dPen':>7} |{'gFINAL':>10}{'bFINAL':>10}{'dFINAL':>8} | aug")
    sep = "-" * len(header)
    lines += [header, sep]
    for k in keys:
        M, K = k; g, b = G[k], B[k]
        gr, br = g['routing_mean'], b['routing_mean']
        gp, bp = g['penalty_mean'], b['penalty_mean']
        gf, bf = g['final_mean'],   b['final_mean']
        dR, dP, dF = br - gr, bp - gp, bf - gf
        byK[K].append(dF)
        win = 'beam' if bf < gf else 'greedy'
        if bf >= gf:
            worse.append((M, K, round(dF, 2)))
        lines.append(f"{M:>4}{K:>3} | {gr:>9.2f}{br:>9.2f}{dR:>8.2f} |"
                     f"{gp:>7.2f}{bp:>7.2f}{dP:>7.2f} |{gf:>10.2f}{bf:>10.2f}{dF:>8.2f} | {win}")
    lines.append(sep)
    lines.append("cells where beam FINAL does NOT beat greedy FINAL: "
                 + (str(worse) if worse else "NONE"))
    lines.append("mean dFINAL by K: " +
                 "  ".join(f"K{K}:{sum(v)/len(v):+.2f}" for K, v in sorted(byK.items())))
    alld = [x for v in byK.values() for x in v]
    lines.append(f"overall mean dFINAL: {sum(alld)/len(alld):+.2f}   "
                 f"(beam better in {sum(1 for x in alld if x < 0)}/{len(alld)} cells)")

    text = "\n".join(lines)
    print("\n" + text)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, 'compare_final_table.txt'), 'w') as f:
            f.write(text + "\n")
        csvp = os.path.join(args.out, 'compare_final.csv')
        with open(csvp, 'w') as f:
            f.write("M,K,instances,greedy_routing,beam_routing,d_routing,"
                    "greedy_penalty,beam_penalty,d_penalty,"
                    "greedy_final,beam_final,d_final,augmented_final,augmented_method\n")
            for k in keys:
                M, K = k; g, b = G[k], B[k]
                gf, bf = g['final_mean'], b['final_mean']
                aug = min(gf, bf); meth = 'beam' if bf < gf else 'greedy'
                f.write(f"{M},{K},{g.get('instances')},{g['routing_mean']:.3f},{b['routing_mean']:.3f},"
                        f"{b['routing_mean']-g['routing_mean']:.3f},"
                        f"{g['penalty_mean']:.3f},{b['penalty_mean']:.3f},"
                        f"{b['penalty_mean']-g['penalty_mean']:.3f},"
                        f"{gf:.3f},{bf:.3f},{bf-gf:.3f},{aug:.3f},{meth}\n")
        print(f"\nwrote {args.out}/compare_final_table.txt and compare_final.csv")


if __name__ == '__main__':
    main()
