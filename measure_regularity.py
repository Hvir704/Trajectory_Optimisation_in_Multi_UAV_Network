"""measure_regularity.py -- why does an age-blind patrol beat SA on the ring?

Hypothesis: contiguous tour segments give near-regular visits (low inter-visit CV) and shorter
inter-stop hops, which outweighs ignoring the priority weights.

For each (planner, K) this records, post-burn-in only:
  tau_hop   = sum(travel time) / sum(n_visited - 1)   -- travel per stop, as in Prop. 3
  CV        = unweighted mean over nodes of sd/mean of inter-visit intervals, first (left-censored)
              interval dropped, nodes with < 3 usable intervals excluded (their share is reported)
  CV_w      = the same, weighted by w_i * f_i -- the statistic Sec. V-B says is the required one
  Gamma     = realised fleet visit rate, visits / s
  J, mean_n, r_c

usage: python measure_regularity.py out.csv --layout ring --M 100 --Emax 3e6 --K 4 5 6 7 8 9
                                    --seeds 1-3 --planners sa rr_tour
Appends; safe to re-run (skips rows already present). One writer per file.
"""
import argparse, csv, os, time
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner

FIELDS = ["planner", "layout", "M", "Emax", "K", "seed", "J", "J_age", "tau_hop", "CV", "CV_w",
          "share_sparse", "Gamma", "J0_ideal", "J1_alloc", "J2_alloc_cv", "E_alloc", "E_reg",
          "mean_n", "r_c", "n_sorties", "secs"]

# Decomposition of the weighted age (Prop. 4, with the inspection-paradox factor of Sec. V-B):
#   J0 = (sum sqrt(w))^2 / (2 Gamma)      ideal sqrt-allocation at the realised total rate
#   J1 = sum w_i / (2 f_i)                realised allocation, perfectly regular visits
#   J2 = sum w_i (1 + CV_i^2) / (2 f_i)   realised allocation and realised variability
# so J_age ~ J2 = J0 * E_alloc * E_reg with E_alloc = J1/J0 >= 1 and E_reg = J2/J1 >= 1.
# This splits any J difference between two planners into rate, allocation and regularity.


def build(planner, seed):
    if planner == "sa":
        return build_sa_planner(iters=1200, seed_base=seed)
    if planner.startswith("rr_"):
        from rr_planner import build_rr_planner
        return build_rr_planner("tour" if planner == "rr_tour" else "sweep")
    if planner == "greedy":
        from dyn_env import greedy_ratio_planner
        return greedy_ratio_planner
    raise ValueError(planner)


def run_one(planner, layout, M, Emax, K, seed, Th=43200.0):
    p = DynParams(M=M, K=K, Emax=Emax, layout=layout, T_horizon=Th, T_burnin=3 * 3600.0)
    sim = DynSim(p, build(planner, seed), seed=seed)
    log = [[] for _ in range(M)]
    orig = sim.field.visit
    def visit(t, i):                      # record, then do exactly what the simulator did
        log[i].append(t)
        return orig(t, i)
    sim.field.visit = visit
    t0 = time.time()
    m = sim.run()

    recs = [r for r in sim.records if r.t_launch >= p.T_burnin]
    hops = sum(max(r.n_visited - 1, 0) for r in recs)
    tau_hop = sum(r.travel_time for r in recs) / max(hops, 1)

    cvs, fs, ws = [], [], []
    n_nodes_ok = 0
    for i, ts in enumerate(log):
        ts = [t for t in ts if t >= p.T_burnin]
        if len(ts) < 4:                   # need >= 3 intervals after dropping the first
            continue
        d = np.diff(ts)[1:]               # drop the left-censored first interval
        if len(d) < 3 or d.mean() <= 0:
            continue
        n_nodes_ok += 1
        cvs.append(d.std(ddof=1) / d.mean())
        fs.append(1.0 / d.mean()); ws.append(sim.field.wi_base[i])
    cvs = np.array(cvs); fs = np.array(fs); ws = np.array(ws)
    wt = ws * fs
    G_est = fs.sum()
    J0 = (np.sqrt(ws).sum() ** 2) / (2 * G_est) if len(fs) else float("nan")
    J1 = float((ws / (2 * fs)).sum()) if len(fs) else float("nan")
    J2 = float((ws * (1 + cvs ** 2) / (2 * fs)).sum()) if len(fs) else float("nan")
    span = max(recs[-1].t_land - recs[0].t_launch, 1e-9)
    return dict(planner=planner, layout=layout, M=M, Emax=Emax, K=K, seed=seed,
                J=m["J_timeavg"], J_age=m["J_age"], tau_hop=tau_hop,
                J0_ideal=J0, J1_alloc=J1, J2_alloc_cv=J2,
                E_alloc=J1 / J0 if J0 == J0 else float("nan"),
                E_reg=J2 / J1 if J1 == J1 else float("nan"),
                CV=float(cvs.mean()) if len(cvs) else float("nan"),
                CV_w=float((cvs * wt).sum() / wt.sum()) if len(cvs) else float("nan"),
                share_sparse=1.0 - n_nodes_ok / M,
                Gamma=sum(r.n_visited for r in recs) / span,
                mean_n=m["mean_n_visited"], r_c=m["r_c_measured"],
                n_sorties=len(recs), secs=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layout", default="ring")
    ap.add_argument("--M", type=int, default=100); ap.add_argument("--Emax", type=float, default=3e6)
    ap.add_argument("--K", nargs="+", type=int, required=True)
    ap.add_argument("--seeds", default="1-3"); ap.add_argument("--planners", nargs="+", default=["sa", "rr_tour"])
    a = ap.parse_args()
    lo, hi = (a.seeds.split("-") + [a.seeds])[:2]; seeds = range(int(lo), int(hi) + 1)

    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        for r in csv.DictReader(open(a.out)):
            done.add((r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["K"]), int(r["seed"])))
    new = not done
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for pl in a.planners:
            for K in a.K:
                for s in seeds:
                    if (pl, a.layout, a.M, a.Emax, K, s) in done: continue
                    row = run_one(pl, a.layout, a.M, a.Emax, K, s)
                    w.writerow(row); f.flush()
                    print(f"{pl:8s} K={K:2d} s={s}: J={row['J']:.3e} tau_hop={row['tau_hop']:5.1f}s "
                          f"CV={row['CV']:.3f} CV_w={row['CV_w']:.3f} sparse={row['share_sparse']:.2f} "
                          f"G={row['Gamma']*3600:6.1f}/h E_alloc={row['E_alloc']:.3f} E_reg={row['E_reg']:.3f} "
                          f"J2/J_age={row['J2_alloc_cv']/max(row['J_age'],1e-9):.3f} ({row['secs']:.0f}s)", flush=True)