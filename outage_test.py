"""outage_test.py -- robust (K_reach) vs optimistic (K_flight) reach choice under a fleet grounding.

Inside the reach bracket (K_reach < K <= K_flight) the farthest sensors are serviceable only while
their buffers never fill: a full buffer needs a full-buffer dwell B/R_u, and at such K a single-sensor
sortie with that dwell exceeds U/K, so the sensor becomes permanently unservable (the planner never
drains partially). This script grounds the whole fleet for G hours mid-run -- the planner returns an
empty route, which the simulator turns into a pad wait of one commute time -- and records J over the
standard 3-12 h window and whether the gray-set sensors are ever served after the outage.

usage: python outage_test.py out.csv
Appends; skips rows already present. J window is 3-24 h. Comparisons are paired on one machine: SA
diverges across platforms, so G=0 rows need not match the Windows grid.
"""
import sys, csv, os, time
import numpy as np
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner

CASES = [("paper", 50, 3e6, 3, 10, 11), ("paper", 50, 3e6, 6, 10, 11), ("paper", 100, 3e6, 12, 8, 9),
         ("core", 50, 1.5e6, 6, 4, 5), ("core", 50, 1.5e6, 3, 5, 6), ("core", 50, 3e6, 12, 10, 11)]
OUT_START = 5 * 3600.0
GAPS = (0.0, 3 * 3600.0)
HORIZON = 24 * 3600.0          # long enough for a permanent loss to register after the outage
FIELDS = ["layout", "M", "Emax", "seed", "K", "which", "G_h", "J", "n_gray", "gray_served_after",
          "gray_lost", "n_never", "secs"]


def run(lay, M, E, seed, K, G):
    p = DynParams(M=M, K=K, Emax=E, layout=lay, T_horizon=HORIZON, T_burnin=3 * 3600.0)
    sa = build_sa_planner(iters=1200, seed_base=seed)
    holder = {}
    def pl(req):
        t = holder["sim"]._clock
        if G > 0 and OUT_START <= t < OUT_START + G:
            return []
        return sa(req)
    sim = DynSim(p, pl, seed=seed); holder["sim"] = sim
    F = sim.field
    r = np.linalg.norm(F.pos - p.home, axis=1)
    U = (1 - p.rho) * E
    r_reach = p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
    gray = np.where(r > r_reach)[0]                    # outside the full-buffer radius at this K
    visits = {int(i): [] for i in gray}
    orig = F.visit
    def visit(t, i):
        if i in visits: visits[i].append(t)
        return orig(t, i)
    F.visit = visit
    t0 = time.time(); m = sim.run()
    after = OUT_START + G
    served_after = sum(1 for i in visits if any(t >= after for t in visits[i])) if G > 0 else len(visits)
    return dict(J=m["J_timeavg"], n_gray=len(gray), gray_served_after=served_after,
                gray_lost=len(gray) - served_after, n_never=m["n_never_visited"], secs=time.time() - t0)


if __name__ == "__main__":
    out = sys.argv[1]
    done = set()
    if os.path.exists(out) and os.path.getsize(out):
        done = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"]), float(r["G_h"]))
                for r in csv.DictReader(open(out))}
    new = not done
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 250.0
    t_start = time.time()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader()
        for lay, M, E, seed, kr, kf in CASES:
            for which, K in (("K_reach", kr), ("K_flight", kf)):
                for G in GAPS:
                    key = (lay, M, E, seed, K, G / 3600)
                    if key in done: continue
                    if time.time() - t_start > budget: sys.exit(0)
                    res = run(lay, M, E, seed, K, G)
                    row = dict(layout=lay, M=M, Emax=E, seed=seed, K=K, which=which, G_h=G / 3600, **res)
                    w.writerow(row); f.flush()
                    print(f"{lay} M={M} {E/1e6:g}MJ s{seed} {which:8s} K={K:2d} G={G/3600:.0f}h: J={res['J']:.4e} "
                          f"gray {res['n_gray']} served-after {res['gray_served_after']} lost {res['gray_lost']} "
                          f"({res['secs']:.0f}s)", flush=True)
