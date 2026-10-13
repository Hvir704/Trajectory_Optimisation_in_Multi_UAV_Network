"""gao_sizing.py -- Gao et al.'s own fleet-size rule (Alg. 2, gzz_planner.gzz_k_choice) on the primary grid, compared
with the measured optima (annealer and Gao et al.'s planner) and with the rule ell.

Gao's rule: the smallest K whose clusters each fit ONE endurance-limited round trip (full-buffer dwell). Under this
paper's shared inventory each UAV's budget is (1 - rho) Emax / K, so the rule returns 'none' when no K fits.

usage: python gao_sizing.py grid_final.csv gzz_eval.csv [--procs P]
"""
import sys, csv, collections, numpy as np
from multiprocessing import Pool
from dyn_env import DynParams, SensorField
from gzz_planner import gzz_k_choice
from kref import kref

def job(a):
    layout, M, Emax, seed = a
    p = DynParams(M=M, Emax=Emax, layout=layout)
    F = SensorField(p, np.random.default_rng(seed))
    return a, gzz_k_choice(F.pos, p.home, p, K_max=60, seed=seed)

def argmins(path, planner=None):
    by = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if planner and r.get("planner", "sa") != planner: continue
        if float(r["L"]) != 12600 or float(r["Th"]) != 43200: continue
        by[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = r
    return by

if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    P = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 4
    sa, gz = argmins(args[0], "sa"), argmins(args[1], "gzz_rev")
    keys = sorted(k for k in sa if k[1] <= 400)
    with Pool(P) as pool: res = dict(pool.map(job, keys))
    print("Gao et al.'s own fleet-size rule (Alg. 2) on the primary grid: K_G = smallest K whose clusters each fit one trip")
    cells = collections.defaultdict(list)
    for k in keys: cells[k[:3]].append(k)
    tot = collections.Counter()
    for c in sorted(cells):
        ks = cells[c]; KG = [res[k][0] for k in ks]; sl = [res[k][1] for k in ks]
        none = sum(x is None for x in KG)
        ksa = [min(sa[k], key=lambda K: float(sa[k][K]["J"])) for k in ks]
        kgz = [min(gz[k], key=lambda K: float(gz[k][K]["J"])) if k in gz else None for k in ks]
        ell = []
        for k in ks:
            r = sa[k][kref(sa[k])]
            ell.append(min(int(np.floor(float(r["K_reach_i"]))), int(round(float(r["K_commute_i"])))))
        ok = [i for i, x in enumerate(KG) if x is not None]
        tot["n"] += len(ks); tot["none"] += none
        line = f"  {c[0]:5s} M={c[1]:3d} {c[2]/1e6:.1f} MJ: K_G none in {none}/{len(ks)}"
        if ok:
            d_sa = [KG[i] - ksa[i] for i in ok]; d_gz = [KG[i] - kgz[i] for i in ok if kgz[i] is not None]
            w1 = sum(abs(x) <= 1 for x in d_sa)
            tot["ok"] += len(ok); tot["w1_sa"] += w1; tot["w1_gz"] += sum(abs(x) <= 1 for x in d_gz); tot["n_gz"] += len(d_gz)
            # regret of Gao's own planner at K_G against its own optimum (where K_G was swept)
            reg = [float(gz[ks[i]][KG[i]]["J"]) / float(gz[ks[i]][kgz[i]]["J"]) - 1 for i in ok
                   if kgz[i] is not None and KG[i] in gz[ks[i]]]
            line += (f"; mean K_G {np.mean([KG[i] for i in ok]):.2f}, annealer optimum {np.mean([ksa[i] for i in ok]):.2f}, "
                     + (f"Gao optimum {np.mean(g):.2f}, " if (g := [kgz[i] for i in ok if kgz[i] is not None]) else "Gao not run here, ")
                     + f"ell {np.mean([ell[i] for i in ok]):.2f}; "
                     f"within one of annealer {w1}/{len(ok)}, of Gao's own optimum {sum(abs(x) <= 1 for x in d_gz)}/{len(d_gz)}")
            if reg: line += f"; Gao regret at K_G median {100*np.median(reg):.1f}% (n={len(reg)})"
        else:
            line += f"; smallest worst-cluster load ratio {np.min(sl):.2f}-{np.max(sl):.2f} x budget"
        print(line)
    vals = collections.Counter(res[k][0] for k in keys)
    print(f"  values of K_G: {dict(sorted(((str(a), b) for a, b in vals.items())))}")
    print(f"  total: K_G none in {tot['none']}/{tot['n']}; where defined, within one of the annealer's optimum "
          f"{tot['w1_sa']}/{tot['ok']}, of Gao's own {tot['w1_gz']}/{tot['n_gz']}")
