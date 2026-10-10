"""sortie_regress.py -- per-sortie data for the Appendix B regressions (hop, hover and productive energy against the
number of stops), on the ring 3 MJ cells.

Each run is exactly the run_grid.py SA run at the same (layout, M, Emax, K, seed) on the auto K range; this script
only records the post-burn-in sorties. One output row per sortie:
  n          stops served on the sortie
  E_hop      Pf * in-field travel time (commute legs excluded)
  E_hov      Ph * dwell time
  E_p        E_hop + E_hov (productive energy)
and, per run: n_never (the run is stranding-free iff 0), Gamma (fleet visits / s after burn-in) and
H = sum_i lambda_i / R (seconds of hovering the field generates per second), so that the conservation argument
E_hov / n = Ph * H / Gamma can be checked.

usage: python sortie_regress.py results_v2\\sortie_ring.csv --M 50 100 200 --Emax 3e6 --seeds 1-4 --procs 20
Appends; safe to stop and restart (a run already in the file is skipped). Analysis: alpha_fit.py --sorties FILE.
"""
import argparse, csv, os, sys, itertools, time
from multiprocessing import Pool
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
from run_grid import SIM_VERSION, k_range, parse_seeds

FIELDS = ["layout", "M", "Emax", "K", "seed", "sortie", "n", "E_hop", "E_hov", "E_p", "n_never", "Gamma", "H",
          "sim_version"]


def one(job):
    layout, M, Emax, K, seed = job
    p = DynParams(M=M, K=K, Emax=Emax, layout=layout, T_horizon=43200.0, T_burnin=3 * 3600.0)
    sim = DynSim(p, build_sa_planner(iters=1200, seed_base=seed), seed=seed)
    m = sim.run()
    recs = [r for r in sim.records if r.t_launch >= p.T_burnin]
    span = max(recs[-1].t_land - recs[0].t_launch, 1e-9) if recs else 1.0
    gamma = sum(r.n_visited for r in recs) / span
    H = float(np.sum(sim.field.lam_bits)) / p.R
    out = []
    for i, r in enumerate(recs):
        e_hop, e_hov = p.Pf * r.travel_time, p.Ph * r.dwell_time
        out.append(dict(layout=layout, M=M, Emax=Emax, K=K, seed=seed, sortie=i, n=r.n_visited, E_hop=e_hop,
                        E_hov=e_hov, E_p=e_hop + e_hov, n_never=m["n_never_visited"], Gamma=gamma, H=H,
                        sim_version=SIM_VERSION))
    if not out:    # keep a marker row so the run is not repeated on restart
        out.append(dict(layout=layout, M=M, Emax=Emax, K=K, seed=seed, sortie=-1, n=0, E_hop=0, E_hov=0, E_p=0,
                        n_never=m["n_never_visited"], Gamma=gamma, H=H, sim_version=SIM_VERSION))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layout", default="ring")
    ap.add_argument("--M", nargs="+", type=int, default=[50, 100, 200])
    ap.add_argument("--Emax", type=float, default=3e6)
    ap.add_argument("--seeds", default="1-4"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        for r in csv.DictReader(open(a.out)):
            if r.get("sim_version") != SIM_VERSION:
                sys.exit(f"{a.out} holds rows from another simulator version; write to a new file")
            done.add((r["layout"], int(r["M"]), float(r["Emax"]), int(r["K"]), int(r["seed"])))
    jobs = [(a.layout, M, a.Emax, K, s) for M, s in itertools.product(a.M, parse_seeds(a.seeds))
            for K in k_range(a.layout, M, a.Emax, s) if (a.layout, M, a.Emax, K, s) not in done]
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} runs to do ({len(done)} already in {a.out})", flush=True)
    new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        t0 = time.time()
        for i, rows in enumerate(pool.imap_unordered(one, jobs)):
            w.writerows(rows); f.flush()
            r = rows[0]
            print(f"[{i + 1}/{len(jobs)}] M={r['M']} K={r['K']} s={r['seed']}: {len(rows)} sorties, "
                  f"n_never={r['n_never']} ({time.time() - t0:.0f}s)", flush=True)
