"""fill_tail.py -- find the far (stranding) minimum that the original anti-censoring rule missed.

The original extension fired only when a sweep's optimum sat at its TOP edge. In the core family J(K)
has two minima: one near the reach cap and a far one where the outer sensors are abandoned and the
fleet serves the dense core (core M=200: 13.7 and 28.3). When the reach-cap minimum is interior, the
sweep was never extended, even if it ended on a descending slope toward a lower far minimum
(36 SA instances in grid_final.csv; confirmed lower in a pilot: core M=50 1.5 MJ s5, K=5 recorded,
K=10-11 lower).

Rule applied here, per deployment, until nothing changes:
    extend by +3 K whenever  J(K_last) < J(K_last - 1)   (sweep ends descending)
                         or  argmin_K J == K_last        (optimum at the top edge)
so every finished sweep ends on a RISING slope with its optimum interior. With --union, K values that
another planner has for the same deployment are also added (own optima on a common grid). A hard cap
(--kmax, default 60) and non-finite J stop an extension.

Rows are appended to OUT (run_grid schema) with the planner given; inputs are never modified. Run it
on the same machine as the input file (SA is not bit-reproducible across platforms).

usage:
  python fill_tail.py fill_sa.csv      --inputs sa_rerun.csv      --planner sa             --procs 22
  python fill_tail.py fill_tour.csv    --inputs rr_tour_win.csv   --planner rr_tour        --procs 22
  python fill_tail.py fill_cluster.csv --inputs cluster_win.csv   --planner cluster_patrol --procs 22
  add --union sa_rerun.csv fill_sa.csv to the patrol runs to put them on SA's K grid as well
  python fill_tail.py fill_depot_q.csv --inputs depot_q.csv --planner sa --depot 0.25 0.25 --procs 22
  python fill_tail.py fill_sa_rule15.csv --inputs sa_rerun.csv --planner sa --rule15 --procs 22
  add --dry to list the runs first
"""
import argparse, csv, math, os, sys
from collections import defaultdict
# An off-centre sweep must be filled with the SAME depot: pass --depot fx fy. The override is set in the
# environment before any worker starts (spawn workers re-import run_depot, which applies it).
if os.environ.get("DEPOT_FRAC"):
    import run_depot  # noqa: F401  (applies the override at import)
from run_grid import one, FIELDS


def std(r):
    return (float(r["Th"]) == 43200 and r["coord"] == "exclude" and r.get("replan", "launch") == "launch"
            and str(r.get("divert", "0")) in ("0", "0.0") and float(r.get("q", 0)) == 0
            and r.get("belief", "prior") == "prior")


def load(paths, planner=None):
    d = defaultdict(dict)
    for p in paths:
        if not os.path.exists(p) or not os.path.getsize(p): continue
        for r in csv.DictReader(open(p)):
            if not std(r): continue
            if planner and r.get("planner", "sa") != planner: continue
            key = (r["layout"], int(r["M"]), float(r["Emax"]), float(r.get("L", 12600)), int(r["seed"]))
            try: d[key][int(r["K"])] = float(r["J"])
            except ValueError: d[key][int(r["K"])] = math.inf
    return d


def needed(d, union, kmax, rule15=False):
    jobs = []
    for key, b in d.items():
        Ks = sorted(b); fin = {K: J for K, J in b.items() if math.isfinite(J)}
        if len(fin) < 2: continue
        last = Ks[-1]; am = min(fin, key=fin.get)
        want = set()
        if math.isfinite(b[last]) and last - 1 in b and b[last] < b[last - 1]: want |= {last + 1, last + 2, last + 3}
        if am == last: want |= {last + 1, last + 2, last + 3}
        # registered C4 rule: the three largest K values must each exceed 1.5 x the minimum J
        if rule15 and not all(b[K] > 1.5 * fin[am] for K in Ks[-3:] if math.isfinite(b[K])):
            want |= {last + 1, last + 2, last + 3}
        if union and key in union: want |= {K for K in union[key] if K not in b}
        for K in sorted(want):
            if K <= kmax and K not in b: jobs.append((key, K))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--planner", default="sa"); ap.add_argument("--union", nargs="*", default=[])
    ap.add_argument("--procs", type=int, default=os.cpu_count()); ap.add_argument("--kmax", type=int, default=60)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", nargs="*", default=[], help='restrict to cells, e.g. "paper,100,6e6"')
    ap.add_argument("--depot", nargs=2, type=float, default=None, help="depot fractions, e.g. 0.25 0.25")
    ap.add_argument("--rule15", action="store_true",
                    help="also extend until the three largest K each exceed 1.5x the minimum (C4's registered rule)")
    a = ap.parse_args()
    if a.depot:
        os.environ["DEPOT_FRAC"] = f"{a.depot[0]},{a.depot[1]}"
        import run_depot
        run_depot.apply_depot(os.environ["DEPOT_FRAC"])
    only = {(c.split(",")[0], int(c.split(",")[1]), float(c.split(",")[2])) for c in a.only}
    union = load(a.union) if a.union else None
    new = not (os.path.exists(a.out) and os.path.getsize(a.out))
    for rnd in range(30):
        d = load(a.inputs + [a.out], planner=None if a.planner == "sa" else a.planner)
        if only: d = {k: v for k, v in d.items() if k[:3] in only}
        jobs = needed(d, union, a.kmax, a.rule15)
        print(f"round {rnd}: {len(jobs)} runs over {len({k for k, _ in jobs})} deployments", flush=True)
        for (key, K) in jobs[:20]: print(f"   {key[0]} M={key[1]} E={key[2]:.1e} L={key[3]:.0f} s={key[4]} K={K}")
        if not jobs or a.dry: break
        args = [(k[0], k[1], k[2], K, k[4], "exclude", "launch", 0.0, 0.0, "prior", 0.0, a.planner, k[3], 43200.0, 1200)
                for (k, K) in jobs]
        from multiprocessing import Pool
        with open(a.out, "a", newline="") as f, Pool(min(a.procs, len(args))) as pool:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new: w.writeheader(); new = False
            for r in pool.imap_unordered(one, args):
                w.writerow(r); f.flush()
                print(f"  {r['layout']} M={r['M']} E={float(r['Emax']):.1e} s={r['seed']} K={r['K']} J={float(r['J']):.3e}", flush=True)
    print("done. Merge OUT into the planner's file (or pass both to reproduce/build_grid_final) and re-run reproduce.py")


if __name__ == "__main__":
    main()
