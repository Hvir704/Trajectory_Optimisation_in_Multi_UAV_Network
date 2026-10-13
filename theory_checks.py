"""theory_checks.py -- the measured quantities behind Sec. IV (Proposition 3 and the r_c(K) statement).

  1) r_c(K): per primary cell, cell-mean r_c at each K, rise from the smallest to the largest swept K, and the
     largest excursion from the smallest-K value (the 'flat within x%' figure).
  2) P_bar spread across stranding-free K, per instance: (max - min)/mean; median and max per family.
  3) tau_hop(K) per stop from the logged aggregates (see tauhop.py): theta = (Ph - Pbar)/(Ph - Pf),
     tau_hop = theta*(T_s - t_c)/n, T_s = T_s_over_t_c * p.t_c (nominal), t_c = 2 r_c / v.
  4) Proposition 3 with measured tau_hop(K): per instance argmax_K (Phi(K) - H)/tau_hop(K) over stranding-free K,
     against argmax Phi and argmin J, for the four ring cells (and any --cells given).

usage: python theory_checks.py SA_GRID.csv [--tour RR_TOUR.csv]
"""
import sys, csv, collections, numpy as np
from dyn_env import DynParams, SensorField
from kref import kref

args = sys.argv[1:]
grid = args[0]
tour = args[args.index("--tour") + 1] if "--tour" in args else None
ALLK = "--rc-stranding-free" not in args   # default: r_c(K) over the whole swept range (as in Sec. VI-?)


def load(path, planner):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in csv.DictReader(open(path)):
        if float(r.get("L", 12600)) != 12600 or float(r.get("Th", 43200)) != 43200: continue
        if r.get("coord", "exclude") != "exclude" or r.get("replan", "launch") != "launch": continue
        if r.get("planner", "sa") != planner or r.get("belief", "prior") != "prior" or float(r.get("q", 0)) != 0: continue
        d[(r["layout"], int(float(r["M"])), float(r["Emax"]))][int(float(r["seed"]))][int(float(r["K"]))] = r
    return d


def tau_hop(p, r):
    n = float(r["mean_n"] or 0); rc = float(r["r_c"] or 0)
    if n <= 0 or rc <= 0 or int(float(r["n_never"])) > 0: return None
    theta = (p.Ph - float(r["P_bar"])) / (p.Ph - p.Pf)
    return theta * (float(r["T_s_over_t_c"]) * p.t_c - 2 * rc / p.v) / n


D = load(grid, "sa")
print("== 1) r_c(K) per cell (cell mean over seeds; K common to all seeds, " + ("all swept K" if ALLK else "stranding-free K only") + ")")
rises = {}
for cell in sorted(D):
    inst = D[cell]
    Kc = sorted(K for K in set.intersection(*[set(b) for b in inst.values()])
                if all(float(inst[s][K]["r_c"] or 0) > 0 and (ALLK or int(float(inst[s][K]["n_never"])) == 0) for s in inst))
    m = {K: np.mean([float(inst[s][K]["r_c"]) for s in inst if float(inst[s][K]["r_c"] or 0) > 0]) for K in Kc}
    if len(Kc) < 2:
        print(f"  {cell[0]:6s} M={cell[1]:3d} E={cell[2]/1e6:.1f}MJ  fewer than two usable K"); continue
    rise = m[Kc[-1]] / m[Kc[0]] - 1; exc = max(abs(m[K] / m[Kc[0]] - 1) for K in Kc)
    rises[cell] = (rise, exc)
    print(f"  {cell[0]:6s} M={cell[1]:3d} E={cell[2]/1e6:.1f}MJ  K {Kc[0]}..{Kc[-1]}  rise {rise:+6.1%}  max |excursion| {exc:5.1%}")
up = [c for c, (r_, e) in rises.items() if r_ > 0.05]
flat = [c for c, (r_, e) in rises.items() if abs(r_) <= 0.05]
print(f"  rises by >5%: {len(up)} of {len(rises)} cells; |rise| <= 5%: {len(flat)} ({sorted(set(c[0] for c in flat))})")
for fam in ("paper", "core", "ring"):
    v = [rises[c] for c in rises if c[0] == fam]
    if v: print(f"  {fam}: rise {min(x[0] for x in v):+.1%} .. {max(x[0] for x in v):+.1%}; max excursion {max(x[1] for x in v):.1%}")

print("\n== 2) P_bar spread across stranding-free K, per instance")
for fam in ("paper", "core", "ring"):
    sp = []
    for cell, inst in D.items():
        if cell[0] != fam: continue
        for s, b in inst.items():
            P = [float(r["P_bar"]) for r in b.values() if int(float(r["n_never"])) == 0 and float(r["P_bar"] or 0) > 0]
            if len(P) >= 2: sp.append((max(P) - min(P)) / np.mean(P))
    print(f"  {fam}: n={len(sp)} median {np.median(sp):.1%} max {max(sp):.1%}")

print("\n== 3) tau_hop(K), seconds per stop (cell mean over seeds with no stranding at that K)")
for path, pl, cells in ((grid, "sa", [("ring", 100, 3e6), ("paper", 100, 1.5e6)]), (tour, "rr_tour", [("ring", 100, 3e6)])):
    if path is None: continue
    DD = D if pl == "sa" else load(path, pl)
    for cell in cells:
        if cell not in DD: continue
        p = DynParams(M=cell[1], Emax=cell[2], layout=cell[0]); by = collections.defaultdict(list)
        for s, b in DD[cell].items():
            for K, r in b.items():
                t = tau_hop(p, r)
                if t is not None: by[K].append(t)
        print(f"  {pl:8s} {cell}: " + "  ".join(f"K{K}:{np.mean(by[K]):.1f}(n{len(by[K])})" for K in sorted(by) if K <= 10))

print("\n== 4) Proposition 3 with measured tau_hop(K)")
for cell in [("ring", 50, 3e6), ("ring", 100, 3e6), ("ring", 200, 3e6), ("ring", 100, 1.5e6)]:
    if cell not in D: continue
    p = DynParams(M=cell[1], Emax=cell[2], layout=cell[0]); U = (1 - p.rho) * cell[2]; out = []
    for s, b in D[cell].items():
        F = SensorField(DynParams(M=cell[1], Emax=cell[2], layout=cell[0]), np.random.default_rng(s)); H = F.lam_bits.sum() / p.R
        r0 = b[kref(b)]; tc = 2 * float(r0["r_c"]) / p.v; Pb = float(r0["P_bar"])
        Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
        G = {}
        for K, r in b.items():
            t = tau_hop(p, r)
            if t is not None and Phi(K) > 0: G[K] = (Phi(K) - H) / t
        J = {K: float(r["J"]) for K, r in b.items()}
        if G: out.append((max(G, key=G.get), max(range(1, 60), key=lambda K: Phi(K) if Phi(K) > 0 else -1), min(J, key=J.get)))
    A = np.array(out, float)
    print(f"  {cell}: argmax (Phi-H)/tau_hop {A[:,0].mean():.2f} | argmax Phi {A[:,1].mean():.2f} | argmin J {A[:,2].mean():.2f}"
          f" | within one of argmin J: {np.mean(abs(A[:,0]-A[:,2])<=1):.2f} vs Phi {np.mean(abs(A[:,1]-A[:,2])<=1):.2f}")

print("\n== 5) Sec. VI-C ring stop count: model n = (U/K - Pf t_c)/(Pbar tau_s) calibrated at K=4 vs realised mean_n")
for cell in [("ring", 100, 3e6), ("ring", 50, 3e6), ("ring", 200, 3e6)]:
    if cell not in D: continue
    p = DynParams(M=cell[1], Emax=cell[2], layout=cell[0]); U = (1 - p.rho) * cell[2]
    out = collections.defaultdict(lambda: [[], []])
    for s, b in D[cell].items():
        if 4 not in b: continue
        r4 = b[4]; tc4 = 2 * float(r4["r_c"]) / p.v
        tau4 = (float(r4["T_s_over_t_c"]) * p.t_c - tc4) / float(r4["mean_n"]); Pb4 = float(r4["P_bar"])
        tc = 2 * float(b[kref(b)]["r_c"]) / p.v            # the rule's t_c (smallest-K row)
        for K in (4, 6, 8, 10):
            if K in b:
                out[K][0].append((U / K - p.Pf * tc) / (Pb4 * tau4)); out[K][1].append(float(b[K]["mean_n"]))
    print(f"  {cell}: " + "  ".join(f"K{K}: realised {np.mean(v[1]):.1f} model {np.mean(v[0]):.1f} ratio {np.mean(v[1]) / np.mean(v[0]):.2f}"
                                    for K, v in sorted(out.items())))
