"""run_grid.py -- resumable, parallel M x K x Emax x layout x seed sweep -> CSV.

usage: python run_grid.py out.csv --layouts paper ring core --M 50 100 200 400 \
         --Emax 1.5e6 3e6 6e6 --seeds 1-12 --coord exclude --iters 1200 --procs 8
K range per cell is auto: [max(1, floor(0.5*Kreach)) .. ceil(1.5*Kreach)] from the
instance geometry, so higher budgets get wider K ranges without hand-editing.
"""
import argparse, csv, os, sys, itertools, math, time
import numpy as np
from multiprocessing import Pool
from dyn_env import DynParams, DynSim, SensorField
from sa_sortie import build_sa_planner

FIELDS = ["layout","M","Emax","K","seed","coord","replan","q","belief","tau","Th","J","J_age","J_event","n_replans","n_never","share_never","n_reach_inf",
          "K_reach_i","K_commute_i","r_max","r_c","rmax_over_rc","regime_bnd","P_bar","dup_frac",
          "trunc_frac","empty_frac","T_s_over_t_c","mean_n","catch_rate","T_rev","n_sorties","secs"]

def k_range(layout, M, Emax, seed):
    p = DynParams(M=M, Emax=Emax, layout=layout)
    F = SensorField(p, np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1).max()
    Kr = p.v*(1-p.rho)*Emax/(2*p.Pf*r + p.v*p.Ph*p.B_bits/p.R)
    # never schedule K where NO sensor is reachable (ring layouts): J is infinite there
    r_min = np.linalg.norm(F.pos - p.home, axis=1).min()
    K_any = p.v*(1-p.rho)*Emax/(2*p.Pf*r_min + p.v*p.Ph*p.B_bits/p.R)
    hi = min(int(math.ceil(1.5*Kr)), int(math.floor(K_any)))
    return list(range(max(1, int(0.5*Kr)), max(hi, max(1, int(0.5*Kr))) + 1))

def one(args):
    layout, M, Emax, K, seed, coord, replan, q, belief, tau, Th, iters = args
    p = DynParams(M=M, K=K, Emax=Emax, layout=layout, coord_mode=coord,
                  event_corr_q=q, belief_mode=belief,
                  tau_e_lo=(tau*60*2/3 if tau > 0 else DynParams.tau_e_lo),   # tau = MEAN lifetime in minutes;
                  tau_e_hi=(tau*60*4/3 if tau > 0 else DynParams.tau_e_hi),   # U(2/3, 4/3)*tau keeps the default spread
                  replan_mode="per_leg" if replan != "launch" else "launch",
                  replan_planner="greedy" if replan == "per_leg_greedy" else "same",
                  replan_iters=60, T_horizon=Th, T_burnin=3*3600.0)
    t = time.time()
    m = DynSim(p, build_sa_planner(iters=iters, seed_base=seed), seed=seed,
               coordinate=(coord != "none")).run()
    return dict(layout=layout, M=M, Emax=Emax, K=K, seed=seed, coord=coord, replan=replan, q=q, belief=belief, tau=tau, Th=Th,
        J=m["J_timeavg"], J_age=m["J_age"], J_event=m["J_event"], n_replans=m["n_replans"], n_never=m["n_never_visited"], share_never=m["share_J_never_visited"],
        n_reach_inf=m["n_reach_infeasible"], K_reach_i=m["K_cov_instance"],
        K_commute_i=m["K_commute_instance"], r_max=m["r_max_instance"], r_c=m["r_c_measured"],
        rmax_over_rc=m["rmax_over_rc"], regime_bnd=m["regime_boundary"], P_bar=m["P_bar"],
        dup_frac=m["dup_frac"], trunc_frac=m["trunc_frac"], empty_frac=m["empty_frac"],
        T_s_over_t_c=m["T_s_over_t_c"], mean_n=m["mean_n_visited"], catch_rate=m["catch_rate"],
        T_rev=m["T_rev"], n_sorties=m["n_sorties"], secs=time.time()-t)

def parse_seeds(s):
    if "-" in s: a, b = s.split("-"); return list(range(int(a), int(b)+1))
    return [int(x) for x in s.split(",")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--layouts", nargs="+", default=["paper"])
    ap.add_argument("--M", nargs="+", type=int, default=[100])
    ap.add_argument("--Emax", nargs="+", type=float, default=[1.5e6])
    ap.add_argument("--seeds", default="1-12"); ap.add_argument("--coord", default="exclude")
    ap.add_argument("--replan", default="launch", choices=["launch","per_leg_greedy","per_leg_sa"])
    ap.add_argument("--q", type=float, default=0.0); ap.add_argument("--belief", default="prior", choices=["none","prior","kernel"])
    ap.add_argument("--K", nargs="+", type=int, default=None, help="fixed K list (skips auto range)")
    ap.add_argument("--tau", type=float, default=0.0, help="mean event lifetime in MINUTES (0 = default 45-90 min)")
    ap.add_argument("--Th", type=float, default=12*3600); ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out) as f: hdr = f.readline().strip().split(",")
        if hdr != FIELDS:
            import subprocess; subprocess.run([sys.executable, "fix_schema.py", a.out], check=True)
        with open(a.out) as f:
            for row in csv.DictReader(f):
                if "layout" not in row: continue   # tolerate a headerless/partial file
                done.add((row["layout"], int(row["M"]), float(row["Emax"]), int(row["K"]), int(row["seed"]), row["coord"], row.get("replan","launch"), float(row.get("q",0)), row.get("belief","prior"), float(row.get("tau",0)), float(row["Th"])))
    jobs = []
    for layout, M, Emax, seed in itertools.product(a.layouts, a.M, a.Emax, parse_seeds(a.seeds)):
        for K in (a.K if a.K else k_range(layout, M, Emax, seed)):
            key = (layout, M, Emax, K, seed, a.coord, a.replan, a.q, a.belief, a.tau, a.Th)
            if key not in done: jobs.append(key + (a.iters,))
    # biggest first so the pool tail is short
    jobs.sort(key=lambda j: -j[1])
    print(f"{len(jobs)} runs to do ({len(done)} already in {a.out})", flush=True)
    new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, jobs)):
            w.writerow(row); f.flush()
            if i % 10 == 0: print(f"[{i+1}/{len(jobs)}] {row['layout']} M={row['M']} E={row['Emax']:.1e} K={row['K']} s={row['seed']} J={row['J']:.3e} {row['secs']:.0f}s", flush=True)