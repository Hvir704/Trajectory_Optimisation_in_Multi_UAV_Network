"""selector.py -- option A: regime-aware choice between field-wide SA and a territorial planner.
Protocol: PLANNER_RESEARCH.md (candidate set pre-declared before any pilot objective values).

  python selector.py fit  pilot.csv --terr cluster_patrol          -> prints a CHOICE string (pilot seeds only)
  python selector.py eval SA.csv TERR.csv --terr cluster_patrol --choice "<string from fit>"
fit refuses seeds <= 12. eval needs the choice explicitly; if the files carry depot_x/depot_y (off-centre
depot sweeps) the depot override is applied before any statistic is computed (the OOD test)."""
import sys, csv, math, argparse, json
from collections import defaultdict
import numpy as np

STATS = ("rho1", "rho2", "rho3", "rho4")

def rule_K(r0):
    kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
    return int(law) if law == kr else int(round(law))

def stats(lay, M, E, seed, K):
    from dyn_env import DynParams, SensorField
    from rr_planner import _kmeans
    import exante_rc as X
    K = max(K, 1); p = DynParams(M=M, Emax=E, layout=lay)
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); lab = _kmeans(F.pos, K, seed=0)
    ks = [j for j in range(K) if np.any(lab == j)]
    near = np.mean([r[lab == j].min() for j in ks])
    cen = np.mean([np.linalg.norm(F.pos[lab == j].mean(0) - p.home) for j in ks])
    rc, _, _ = X.exante(p, r, F.lam_bits, F.pos, K)
    return dict(rho1=near / rc, rho2=cen / rc, rho3=cen / r.mean(), rho4=r.mean() / r.max())

def load(path, planner):
    d = defaultdict(dict); depot = None
    for r in csv.DictReader(open(path)):
        if r.get("planner", "sa") != planner or float(r["Th"]) != 43200 or r["coord"] != "exclude": continue
        d[(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
        if r.get("depot_x"): depot = f"{r['depot_x']},{r['depot_y']}"
    return d, depot

def pairs(S, T):
    out = []
    for k in sorted(set(S) & set(T)):
        Ks = sorted(set(S[k]) & set(T[k]))
        if len(Ks) < 3: continue
        out.append(dict(k=k, st=stats(*k, rule_K(S[k][min(S[k])])),
                        J_sa=min(float(S[k][K]["J"]) for K in Ks), J_terr=min(float(T[k][K]["J"]) for K in Ks)))
    return out

def pick(x, c):
    v = x["st"][c["stat"]]
    if c["rule"] == "le": terr = v <= c["t1"]
    elif c["rule"] == "ge": terr = v >= c["t1"]
    else: terr = v <= c["t1"] or v >= c["t2"]
    return terr

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["fit", "eval"]); ap.add_argument("files", nargs="+")
    ap.add_argument("--terr", default="cluster_patrol"); ap.add_argument("--choice")
    a = ap.parse_args()
    if a.mode == "fit":
        S, _ = load(a.files[0], "sa"); T, _ = load(a.files[0], a.terr)
        if any(k[3] <= 12 for k in list(S) + list(T)): sys.exit("fit must use pilot seeds only (> 12)")
        P = pairs(S, T)
        if not P: sys.exit(f"no pilot instances with both 'sa' and '{a.terr}' rows -- run run_pilot.py with both planners first")
        best = None
        for stat in STATS:
            vals = sorted({x["st"][stat] for x in P}); cut = [vals[0] - 1] + [(a_ + b) / 2 for a_, b in zip(vals, vals[1:])] + [vals[-1] + 1]
            cands = [dict(stat=stat, rule=r, t1=t) for r in ("le", "ge") for t in cut]
            cands += [dict(stat=stat, rule="band", t1=t1, t2=t2) for i, t1 in enumerate(cut) for t2 in cut[i + 1:]]
            for c in cands:
                correct = sum((x["J_terr"] < x["J_sa"]) == pick(x, c) for x in P)
                key = (correct, c["rule"] != "band")              # prefer single thresholds on ties
                if best is None or key > best[0]: best = (key, c)
        (correct, _), c = best
        print(f"pilot n={len(P)} | correct choices {correct}/{len(P)} | CHOICE {json.dumps(c)}")
        for fam in ("paper", "ring", "core"):
            Q = [x for x in P if x["k"][0] == fam]
            if Q: print(f"  {fam:5s} territorial better {sum(x['J_terr'] < x['J_sa'] for x in Q)}/{len(Q)} | "
                        f"{c['stat']} range {min(x['st'][c['stat']] for x in Q):.2f}-{max(x['st'][c['stat']] for x in Q):.2f} | "
                        f"picks territories {sum(pick(x, c) for x in Q)}/{len(Q)}")
    else:
        if not a.choice: sys.exit("eval needs --choice from the pilot fit")
        c = json.loads(a.choice)
        S, dep = load(a.files[0], "sa"); T, dep2 = load(a.files[1], a.terr)
        if dep or dep2:
            from run_depot import apply_depot; apply_depot(dep or dep2); print(f"depot override applied: {dep or dep2}")
        P = pairs(S, T); res = {}
        print(f"evaluation n={len(P)} | choice {json.dumps(c)} | territorial component {a.terr}")
        for fam in ("all", "paper", "ring", "core"):
            Q = [x for x in P if fam == "all" or x["k"][0] == fam]
            if not Q: continue
            rel = np.array([(x["J_terr"] if pick(x, c) else x["J_sa"]) / x["J_terr"] - 1 for x in Q]); res[fam] = np.median(rel)
            print(f"  {fam:5s} n={len(Q):3d} | selected vs {a.terr}: median {np.median(rel):+.1%} | better {int(np.sum(rel < -1e-12))} "
                  f"tied {int(np.sum(abs(rel) <= 1e-12))} worse {int(np.sum(rel > 1e-12))} | territories {np.mean([pick(x, c) for x in Q]):.0%}")
        ok = res["all"] < 0 and all(v <= 0.01 for f, v in res.items() if f != "all")
        print(f"pre-registered criterion (overall median < 0, no family > +1%): {'MET' if ok else 'NOT MET'}")

if __name__ == "__main__":
    main()
