"""c4_classes.py -- share of sensors that C4's design moves out of the base frequency class, at C4's own-optimum K,
for every held-out deployment. Rebuilds only the design (deterministic; no simulation).
usage: python c4_classes.py eval_all_v2.csv [--procs N]"""
import sys, csv, os
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
from dyn_env import DynParams, SensorField
from cyclic_sched import optimise, CFG


def one(job):
    lay, M, E, seed, K = job
    p = DynParams(M=M, K=K, Emax=E, layout=lay)
    F = SensorField(DynParams(M=M, K=K, Emax=E, layout=lay), np.random.default_rng(seed))
    S, init, _ = optimise(F.pos, p.home, F.wi_base.copy(), p, K, CFG)
    c = np.asarray(S.c)
    return (lay, M, E, seed, K, init, float(np.mean(c > 0)))


if __name__ == "__main__":
    d = defaultdict(dict)
    for r in csv.DictReader(open(sys.argv[1])):
        if r.get("planner") != "cyc" or r["coord"] != "exclude" or r["replan"] != "launch" or float(r["Th"]) != 43200: continue
        d[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = float(r["J"])
    jobs = [k + (min(b, key=b.get),) for k, b in sorted(d.items())]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else os.cpu_count()
    with Pool(procs) as pool:
        res = pool.map(one, jobs)
    sh = np.array([r[-1] for r in res])
    print(f"C4 designs at the own-optimum K: {len(res)}; share of sensors outside the base class: median {np.median(sh):.1%}, "
          f"mean {sh.mean():.1%}, max {sh.max():.1%}")
    for fam in ("paper", "ring", "core"):
        x = [r[-1] for r in res if r[0] == fam]; ini = [r[5] for r in res if r[0] == fam]
        print(f"  {fam}: median {np.median(x):.1%}; start chosen: kmeans {ini.count('kmeans')}, sector {ini.count('sector')}")
