"""kappa_diag.py -- where does the surrogate's level error kappa come from? (diagnostic, design seeds only)

kappa = J_age(measured) / served term of (S1), at a stranding-free K or at v3's stranding candidate. Exact identity:
  kappa = k_rate * k_alloc * k_reg * k_res
  k_rate  = Gamma_S / Gamma_meas                    a-priori visit rate vs realised (Gamma_meas = sum_i f_i)
  k_alloc = J1 / J0r,  J0r = A_R B_R / (2 Gamma_meas), J1 = sum_{served} w_i / (2 f_i)
  k_reg   = J2 / J1,   J2 = sum w_i (1 + CV_i^2) / (2 f_i)      (inspection paradox)
  k_res   = J_age / J2                              window edges, sparsely visited sensors, stranded age
f_i, CV_i from the complete post-burn-in inter-visit intervals of each served sensor (>= 2 intervals).
usage: python kappa_diag.py out.csv [--procs 2] [--seeds 1-3] [--planners sa]
"""
import sys, os, csv, math, time, itertools, argparse
import numpy as np
from multiprocessing import Pool
from dyn_env import DynParams, DynSim
from theory_v3 import field, old_rule, surrogate_rule
from exante_rc import exante
from run_grid import k_range, SIM_VERSION

FIELDS = ["planner", "layout", "M", "Emax", "seed", "K", "which", "j", "J_age", "J_age_all_log", "J_age_sim", "served_sur", "kappa", "k_rate", "k_alloc",
          "k_reg", "k_res", "Gamma_S", "Gamma_meas", "tau_hop_sur", "tau_hop_meas", "cv2_wf", "n_sparse", "secs", "sim_version"]

def build(planner, seed):
    if planner == "sa":
        from sa_sortie import build_sa_planner; return build_sa_planner(iters=1200, seed_base=seed)
    if planner == "cyc":
        from cyclic_sched import build_cyclic_planner; return build_cyclic_planner()
    if planner == "rr_tour":
        from rr_planner import build_rr_planner; return build_rr_planner("tour")
    raise ValueError(planner)

def job(a):
    planner, lay, M, E, s, K, which = a
    p0, F0 = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F0.pos - p0.home, axis=1)
    K0 = min(k_range(lay, M, E, s, 12600.0)); rc, Pb, _ = exante(p0, r, np.full(M, p0.lam_bits), F0.pos, K0)
    _, _, curve = surrogate_rule(p0, F0, rc, Pb, return_curve=True, key=(lay, M, 12600.0, s))
    if K not in curve: return None
    Jk, j, served = curve[K]
    U = (1 - p0.rho) * E; tc = 2 * rc / p0.v
    Phi = K * (U - K * p0.Pf * tc) / (U + K * tc * (Pb - p0.Pf))
    keep = r <= p0.v * (U / K - p0.Ph * p0.B_bits / p0.R) / (2 * p0.Pf)
    from theory_v3 import tour_len
    d_S = tour_len(F0.pos[keep], p0.home) / keep.sum(); H = keep.sum() * p0.lam_bits / p0.R
    G_S = p0.v * (Phi - H) / d_S
    pi = r[keep] / r[keep].mean(); w = F0.wi_base
    AB = np.sqrt(w[keep] * pi).sum() * np.sqrt(w[keep] / pi).sum()
    # simulate
    p = DynParams(M=M, K=K, Emax=E, layout=lay, T_horizon=43200.0, T_burnin=3 * 3600.0)
    pl = build(planner, s); sim = DynSim(p, pl, seed=s)
    if hasattr(pl, "bind_sim"): pl.bind_sim(sim)
    log = [[] for _ in range(M)]; orig = sim.field.visit
    def visit(t, i): log[i].append(t); return orig(t, i)
    sim.field.visit = visit; t0 = time.time(); m = sim.run()
    recs = [x for x in sim.records if x.t_launch >= p.T_burnin]
    stops = sum(x.n_visited for x in recs); tau_meas = sum(x.travel_time for x in recs) / max(stops, 1)
    fs, ws, cvs = [], [], []; sparse = 0
    for i in np.where(keep)[0]:
        ts = [t for t in log[i] if t >= p.T_burnin]; d = np.diff(ts)
        if len(d) < 2 or d.mean() <= 0: sparse += 1; continue
        fs.append(1 / d.mean()); ws.append(w[i]); cvs.append(d.std(ddof=1) / d.mean())
    fs, ws, cvs = map(np.array, (fs, ws, cvs))
    Gm = fs.sum(); J0r = AB / (2 * Gm); J1 = (ws / (2 * fs)).sum(); J2 = (ws * (1 + cvs ** 2) / (2 * fs)).sum()
    # time-average age over the scored window, per sensor, from the visit log (age 0 at t = 0, reset at each visit)
    Tb, Te = p.T_burnin, p.T_horizon
    def avg_age(ts):
        pts = [0.0] + [t for t in ts if t < Te]; tot = 0.0
        for a_, b_ in zip(pts, pts[1:] + [Te]):
            lo_, hi_ = max(a_, Tb), min(b_, Te)
            if hi_ > lo_: tot += ((hi_ - a_) ** 2 - (lo_ - a_) ** 2) / 2
        return tot / (Te - Tb)
    ages = np.array([avg_age(log[i]) for i in range(M)])
    Jage_all = float((w * ages).sum()); Jage = float((w[keep] * ages[keep]).sum()); wt = ws / fs
    return dict(planner=planner, layout=lay, M=M, Emax=E, seed=s, K=K, which=which, j=j, J_age=Jage, J_age_all_log=Jage_all,
                J_age_sim=m["J_age"], served_sur=served,
                kappa=Jage / served, k_rate=G_S / Gm, k_alloc=J1 / J0r, k_reg=J2 / J1, k_res=Jage / J2,
                Gamma_S=G_S, Gamma_meas=Gm, tau_hop_sur=d_S / p0.v, tau_hop_meas=tau_meas,
                cv2_wf=float((cvs ** 2 * wt).sum() / wt.sum()), n_sparse=sparse, secs=time.time() - t0, sim_version=SIM_VERSION)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seeds", default="1-3"); ap.add_argument("--planners", nargs="+", default=["sa"])
    a = ap.parse_args(); lo, hi = a.seeds.split("-"); seeds = range(int(lo), int(hi) + 1)
    jobs = []
    for lay, M, E, s in itertools.product(("paper", "ring", "core"), (100, 200), (1.5e6, 3e6), seeds):
        p0, F0 = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F0.pos - p0.home, axis=1)
        K0 = min(k_range(lay, M, E, s, 12600.0)); rc, Pb, _ = exante(p0, r, np.full(M, p0.lam_bits), F0.pos, K0)
        Kold = old_rule(p0, r, rc, Pb); Kv3, j = surrogate_rule(p0, F0, rc, Pb, key=(lay, M, 12600.0, s))
        for pl in a.planners:
            jobs.append((pl, lay, M, E, s, Kold, "old"))
            if lay == "core":   # the stranding candidate: best j > 0 fleet size of the surrogate
                _, _, cur = surrogate_rule(p0, F0, rc, Pb, return_curve=True, key=(lay, M, 12600.0, s))
                cand = [(v[0], K) for K, v in cur.items() if v[1] > 0]
                if cand: jobs.append((pl, lay, M, E, s, min(cand)[1], "strand"))
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)): done.add((r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), int(r["K"])))
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3], j[4], j[5]) not in done]
    jobs.sort(key=lambda j: -j[5]); print(len(jobs), "runs", flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new: wr.writeheader()
        for row in pool.imap_unordered(job, jobs):
            if row: wr.writerow(row); f.flush(); print(row["layout"], row["M"], row["Emax"], row["seed"], row["K"], row["which"], f"kappa {row['kappa']:.2f}", flush=True)
