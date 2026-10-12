"""paper_numbers.py -- the remaining numbers of the paper that no table script prints, each labelled with where it
is used. Reads the assembled files of reproduce_v2.py and a few raw results files; runs no simulation.

usage: python paper_numbers.py RESULTS_DIR ASSEMBLED_DIR
"""
import sys, os, csv, math, json
from collections import defaultdict
import numpy as np
from dyn_env import DynParams, SensorField

RES, ASM = sys.argv[1], sys.argv[2]
R = lambda n: os.path.join(RES, n); A = lambda n: os.path.join(ASM, n)
read = lambda p: list(csv.DictReader(open(p))) if os.path.exists(p) else []
P = DynParams()
H = P.T_burnin + (P.T_horizon - P.T_burnin) / 2


def std(r):
    return (r["coord"] == "exclude" and r["replan"] == "launch" and float(r["Th"]) == 43200
            and str(r.get("divert", "0")) in ("0", "0.0") and float(r.get("q", 0) or 0) == 0
            and r.get("belief", "prior") == "prior" and float(r.get("tau", 0) or 0) == 0)


def inst(rows, planner=None, primary=True):
    d = defaultdict(dict)
    for r in rows:
        if not std(r): continue
        if planner and r.get("planner", "sa") != planner: continue
        L = float(r.get("L") or 12600)
        if primary and L != 12600: continue
        k = (r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"]))) if primary else \
            (r["layout"], int(float(r["M"])), float(r["Emax"]), L, int(float(r["seed"])))
        d[k][int(float(r["K"]))] = r
    return d


def am(b): return min(b, key=lambda K: float(b[K]["J"]))
def J(b, K): return float(b[K]["J"])
def ell(b):
    r0 = b[min(b)]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    return kr if kr <= kc else int(round(kc))
def hdr(s): print(f"\n=== {s}")


G = read(A("grid_final.csv"))
SA = inst(G); SAall = inst(G, primary=False)
EV = read(A("eval_all_v2.csv"))
PL = {"sa": SA, "cyc": inst(EV, "cyc"), "cluster_patrol": inst(EV, "cluster_patrol"),
      "gzz_rev": inst(read(A("gzz_eval.csv")), "gzz_rev"), "asghar_rev": inst(read(A("asghar_eval.csv")), "asghar_rev"),
      "liu_mpga": inst(read(A("liu_all.csv")), "liu_mpga"), "rr_tour": inst(read(A("rr_tour_win.csv")), "rr_tour"),
      "rr_sweep": inst(read(A("rr_sweep_win.csv")), "rr_sweep")}
VER = {(v["layout"], int(float(v["M"])), float(v["Emax"]), int(float(v["seed"]))): v["verdict_current"] for v in read(A("verdicts.csv"))}

# ------------------------------------------------------------------------------------------------ intro / abstract
hdr("abstract/intro: rule's l in the representative cell (paper M=100 1.5 MJ), core M=100 1.5 MJ")
for c in [("paper", 100, 1.5e6), ("core", 100, 1.5e6)]:
    ls = [ell(b) for k, b in SA.items() if k[:3] == c]
    print(f"  {c}: l values {sorted(ls)}  l=4 in {sum(1 for x in ls if x == 4)} of {len(ls)}  mean {np.mean(ls):.2f}  "
          f"floor K_r values {sorted(math.floor(float(b[min(b)]['K_reach_i'])) for k, b in SA.items() if k[:3] == c)}")

# ------------------------------------------------------------------------------------------------ model
hdr("model: empty sorties, P_bar range, commute time t_c")
for fam in ("paper", "core", "ring"):
    rows = [r for k, b in SA.items() if k[0] == fam for r in b.values()]
    e = np.array([float(r["empty_frac"]) for r in rows]); eo = np.array([float(b[am(b)]["empty_frac"]) for k, b in SA.items() if k[0] == fam])
    print(f"  {fam}: empty-sortie share max over all runs {e.max():.1%}, at the optimum median {np.median(eo):.1%} max {eo.max():.1%}")
pb = np.array([float(r["P_bar"]) for b in SAall.values() for r in b.values() if float(r["mean_n"]) > 0])
tc = np.array([2 * float(r["r_c"]) / P.v for b in SA.values() for r in b.values()])
print(f"  P_bar over all SA runs that serve anything: {pb.min():.0f}-{pb.max():.0f} W;  realised commute time 2 r_c / v median {np.median(tc)/60:.1f} min; nominal t_c {P.t_c/60:.1f} min")

# ------------------------------------------------------------------------------------------------ rule section
hdr("Lemma 1: in-reach sensors unserved at floor(K_r), clustered M=400 1.5 MJ")
for k, b in sorted(SA.items()):
    if k[:3] != ("paper", 400, 1.5e6): continue
    K0 = math.floor(float(b[min(b)]["K_reach_i"]))
    print(f"  seed {k[3]}: K0={K0} n_never at K0 = {b[K0]['n_never'] if K0 in b else 'not swept'}")


def criterion(k, b, gray_served=False, rc_of_K=False):
    lay, M, E, seed = k; p = DynParams(M=M, Emax=E, layout=lay); U = (1 - p.rho) * E
    F = SensorField(DynParams(M=M, Emax=E, layout=lay), np.random.default_rng(seed))
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; c = 2 * p.Pf * r / p.v
    r0 = b[min(b)]; rc0 = float(r0["r_c"]); Pb0 = float(r0["P_bar"])
    def inputs(K):
        if rc_of_K and K in b: return float(b[K]["r_c"]), float(b[K]["P_bar"])
        if rc_of_K:      # outside the sweep: nearest swept K
            Kn = min(b, key=lambda x: abs(x - K)); return float(b[Kn]["r_c"]), float(b[Kn]["P_bar"])
        return rc0, Pb0
    def Phi(K, rc, Pb):
        tc = 2 * rc / p.v; return K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    reach = (lambda K: p.v * U / K / (2 * p.Pf)) if gray_served else (lambda K: p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf))
    def Fv(K):
        rc, Pb = inputs(K); S = r <= reach(K); ph = Phi(K, rc, Pb)
        if ph <= 0: return None
        return (np.sqrt(w[S] * c[S]).sum()) ** 2 / (2 * Pb * ph) + H * w[~S].sum()
    K0 = math.floor(float(r0["K_reach_i"])); Kmax = int(math.floor(U / (p.Pf * 2 * rc0 / p.v) - 1e-9))
    F0 = Fv(K0)
    vals = [Fv(K) for K in range(K0 + 1, Kmax + 1)]
    return all(v > F0 for v in vals if v is not None)


hdr("criterion (5): verdict changes when the gray set is treated as served / with K-dependent commute inputs")
base, gs, rk = {}, {}, {}
for k, b in SA.items():
    if not (k[1] <= 400): continue
    r0 = b[min(b)]
    if not (math.floor(float(r0["K_reach_i"])) <= float(r0["K_commute_i"])): continue
    base[k] = criterion(k, b); gs[k] = criterion(k, b, gray_served=True); rk[k] = criterion(k, b, rc_of_K=True)
reg = {k: J(b, ell(b)) / J(b, am(b)) - 1 for k, b in SA.items()}
for lab, alt in (("gray set served", gs), ("r_c(K), P_bar(K)", rk)):
    tp = [k for k in base if not base[k] and alt[k]]; tf = [k for k in base if base[k] and not alt[k]]
    print(f"  {lab}: {len(tp) + len(tf)} of {len(base)} change; pass->flag {len(tf)}, flag->pass {len(tp)}")
    if tf: print(f"     largest regret among pass->flag: {max(reg[k] for k in tf):.1%} {max(tf, key=lambda k: reg[k])}")
passes = [k for k in base if base[k]]
print(f"  base: {len(passes)} pass of {len(base)}; largest passing regret {max(reg[k] for k in passes):.1%} {max(passes, key=lambda k: reg[k])}")

hdr("criterion: flagged core instances, J(K0) above the minimum; sufficient variants on zero-regret clustered instances")
fc = [k for k in base if k[0] == "core" and not base[k]]
x = [J(SA[k], math.floor(float(SA[k][min(SA[k])]['K_reach_i']))) / J(SA[k], am(SA[k])) - 1 for k in fc if math.floor(float(SA[k][min(SA[k])]['K_reach_i'])) in SA[k]]
print(f"  flagged core: {len(fc)}; J(K0)/J* - 1 median {np.median(x):.1%} max {max(x):.1%}")

hdr("regret: reach-bound passing instances (all families)")
xp = np.array([reg[k] for k in passes]); print(f"  n={len(xp)} median {np.median(xp):.1%} p90 {np.quantile(xp, .9, method='higher'):.1%} max {xp.max():.1%}")
xr = np.array([reg[k] for k, b in SA.items() if k[0] == "ring"]); print(f"  ring (commute-bound) n={len(xr)} median {np.median(xr):.1%} max {xr.max():.1%}, instances above 30%: {int((xr > .3).sum())}")
over30 = [k for k in reg if reg[k] > .30]
print(f"  all instances above 30%: {len(over30)}; of which reach-bound and passing: {sum(1 for k in over30 if base.get(k))}")

# ------------------------------------------------------------------------------------------------ commute branch
hdr("Remark 4: r_max / mean r per family")
for fam in ("paper", "ring", "core"):
    q = []
    for k in SA:
        if k[0] != fam: continue
        p = DynParams(M=k[1], Emax=k[2], layout=fam); F = SensorField(DynParams(M=k[1], Emax=k[2], layout=fam), np.random.default_rng(k[3]))
        r = np.linalg.norm(F.pos - p.home, axis=1); q.append(r.max() / r.mean())
    print(f"  {fam}: {min(q):.2f}-{max(q):.2f}  (boundary 2.0: deviation {min(q)/2-1:+.1%} to {max(q)/2-1:+.1%})")

hdr("tau_hop (theory_checks definition) per cell: cell mean at the smallest and largest K with >= 3 stranding-free seeds")
def tau_hop(r):
    n = float(r["mean_n"] or 0); rc = float(r["r_c"] or 0)
    if n <= 0 or rc <= 0 or int(float(r["n_never"])) > 0: return None
    theta = (P.Ph - float(r["P_bar"])) / (P.Ph - P.Pf)
    return theta * (float(r["T_s_over_t_c"]) * P.t_c - 2 * rc / P.v) / n
for c in sorted({k[:3] for k in SA}):
    byK = defaultdict(list)
    for k, b in SA.items():
        if k[:3] != c: continue
        for K, r in b.items():
            t = tau_hop(r)
            if t is not None: byK[K].append(t)
    Ks = [K for K in sorted(byK) if len(byK[K]) >= 3]
    if len(Ks) >= 2:
        a0, a1 = np.mean(byK[Ks[0]]), np.mean(byK[Ks[-1]])
        print(f"  {c}: {a0:.1f} s (K={Ks[0]}) -> {a1:.1f} s (K={Ks[-1]})  change {a1/a0-1:+.0%}")

hdr("sorties at the optimum: duration and stops")
d = [float(b[am(b)]["T_s_over_t_c"]) * P.t_c / 60 for b in SA.values()]
print(f"  sortie duration at the optimum: {min(d):.1f}-{max(d):.1f} min (median {np.median(d):.1f})")
for M in (50, 100, 200, 400):
    n = [float(b[am(b)]["mean_n"]) for k, b in SA.items() if k[1] == M and k[0] == "paper" and k[2] == 1.5e6]
    print(f"  clustered 1.5 MJ M={M}: stops per sortie at the optimum {np.mean(n):.1f}")

hdr("ring 3 MJ M=100: P_bar by K (cell mean, stranding-free)")
byK = defaultdict(list)
for k, b in SA.items():
    if k[:3] == ("ring", 100, 3e6):
        for K, r in b.items():
            if float(r["n_never"]) == 0: byK[K].append(float(r["P_bar"]))
print("  " + "  ".join(f"K{K}:{np.mean(v):.0f}" for K, v in sorted(byK.items())))

# ------------------------------------------------------------------------------------------------ figures 2, 3
hdr("Fig. 2: cell-mean J(K) and its argmin (M=100, 1.5 MJ)")
for fam in ("paper", "ring", "core"):
    byK = defaultdict(list)
    for k, b in SA.items():
        if k[:3] == (fam, 100, 1.5e6):
            for K, r in b.items(): byK[K].append(float(r["J"]))
    Ks = [K for K in sorted(byK) if len(byK[K]) == 12]
    m = {K: np.mean(byK[K]) for K in Ks}
    print(f"  {fam}: argmin of cell mean K={min(m, key=m.get)}; " + " ".join(f"K{K}:{m[K]/1e5:.2f}" for K in Ks))
hdr("Fig. 3: argmin J_age over fleet sizes with every sensor served, relative to floor(K_r)")
cnt = defaultdict(int); n_pl = 0
for k, b in SAall.items():
    full = {K: float(r["J_age"]) for K, r in b.items() if float(r["n_never"]) == 0}
    if not full: cnt["no stranding-free K"] += 1; continue
    n_pl += 1
    r0 = b[min(b)]; kr = math.floor(float(r0["K_reach_i"])); rb = float(r0["rmax_over_rc"]) > float(r0["regime_bnd"])
    a = min(full, key=full.get)
    if rb: cnt["reach side: at floor K_r" if a == kr else ("reach side: above" if a > kr else "reach side: below")] += 1
    else: cnt["commute side"] += 1
print(f"  plotted {n_pl} of {len(SAall)}; {dict(cnt)}")

# ------------------------------------------------------------------------------------------------ evaluation
hdr("Sec. VII-B overall: 13 good cells, worst bias")
cells = defaultdict(list)
for k, b in SA.items(): cells[k[:3]].append((am(b), ell(b)))
for c, v in sorted(cells.items()):
    a = np.array(v); w1 = np.mean(abs(a[:, 0] - a[:, 1]) <= 1)
    if w1 >= 0.9: print(f"  good cell {c}: bias {np.mean(a[:,0]-a[:,1]):+.2f}, within-one {w1:.2f}")

hdr("Sec. VII-B warning indicator: onsets for each planner (cell relative bias vs r_max/r_c, reach-limited cells)")
for pl in ("cyc", "cluster_patrol", "gzz_rev", "asghar_rev", "liu_mpga"):
    out = []
    for c in sorted({k[:3] for k in SA}):
        ks = [k for k in SA if k[:3] == c and k in PL[pl]]
        if not ks or c[0] == "ring": continue
        rb = np.mean([(am(PL[pl][k]) - ell(SA[k])) / ell(SA[k]) for k in ks]); ratio = np.mean([float(SA[k][min(SA[k])]["rmax_over_rc"]) for k in ks])
        out.append((ratio, rb))
    out.sort(); print(f"  {pl}: " + "  ".join(f"{a:.1f}:{b:+.0%}" for a, b in out))

hdr("Sec. VII-C core M=200 1.5 MJ cell-mean J at K=4, 10 and its minimum")
byK = defaultdict(list)
for k, b in SA.items():
    if k[:3] == ("core", 200, 1.5e6):
        for K, r in b.items(): byK[K].append(float(r["J"]))
m = {K: np.mean(v) for K, v in byK.items() if len(v) == 12}
print(f"  K=4 {m.get(4, float('nan')):.3e}  K=10 {m.get(10, float('nan')):.3e}  min {min(m.values()):.3e} at K={min(m, key=m.get)}")
st = [min(K for K, r in b.items() if float(r["n_never"]) >= 1) for k, b in SA.items() if k[:3] == ("core", 200, 1.5e6)]
print(f"  first K with a stranded sensor per seed: {st}")

hdr("Sec. VII-C/VIII: optimum above floor(K_r) per planner, core 1.5 and 3 MJ; instances that strand at the optimum")
for c in [("core", 200, 1.5e6), ("core", 200, 3e6), ("core", 100, 1.5e6), ("core", 100, 3e6)]:
    line = []
    for pl in ("sa", "rr_tour", "cluster_patrol", "cyc", "gzz_rev", "asghar_rev", "liu_mpga"):
        ks = [k for k in SA if k[:3] == c and k in PL[pl]]
        if not ks: continue
        ex = np.mean([am(PL[pl][k]) - math.floor(float(SA[k][min(SA[k])]["K_reach_i"])) for k in ks])
        strand = sum(1 for k in ks if float(PL[pl][k][am(PL[pl][k])]["n_never"]) >= 1)
        line.append(f"{pl} {np.mean([am(PL[pl][k]) for k in ks]):.2f} (+{ex:.2f}, strand {strand}/{len(ks)})")
    print(f"  {c}: " + "; ".join(line))

hdr("Sec. VII-D: neighbour cost at M=40 per family and planner (median over instances of the cheaper neighbour, min J(K*+-1)/J* - 1)")
M40 = {"sa": inst(read(A("m40_sa.csv"))), "rr_tour": inst(read(A("m40_rr.csv"))), "cpsat": inst(read(A("m40_cp_final.csv"))),
       "cluster_patrol": inst(read(A("m40_cluster.csv")))}
for pl, D in M40.items():
    for fam in ("paper", "ring"):
        v = []
        for k, b in D.items():
            if k[0] != fam: continue
            a = am(b); nb = [J(b, K) / J(b, a) - 1 for K in (a - 1, a + 1) if K in b]
            if nb: v.append(min(nb))
        print(f"  {pl:15s} {fam}: {np.median(v):.1%} (n={len(v)})")
for (p1, p2) in (("rr_tour", "cpsat"),):
    ks = set(M40[p1]) & set(M40[p2])
    print(f"  M=40 {p1} within one of {p2}: {np.mean([abs(am(M40[p1][k]) - am(M40[p2][k])) <= 1 for k in ks]):.2f} (n={len(ks)})")

hdr("Sec. VII-E field size: per L, rule vs argmin, regret of the misses; r_max/r_c")
for L in (8000, 10000, 12600, 16000, 20000):
    ks = [k for k in SAall if k[:4] == ("paper", 100, 1.5e6, float(L))]
    ams = [am(SAall[k]) for k in ks]; ls = [ell(SAall[k]) for k in ks]
    kf = [float(SAall[k][min(SAall[k])]["K_reach_i"]) * 0 for k in ks]
    regs = [J(SAall[k], ell(SAall[k])) / J(SAall[k], am(SAall[k])) - 1 for k in ks if am(SAall[k]) != ell(SAall[k])]
    nn = [float(SAall[k][am(SAall[k])]["n_never"]) for k in ks]
    print(f"  L={L}: n={len(ks)} argmin {np.mean(ams):.2f} rule {np.mean(ls):.2f} misses {len(regs)} regret {min(regs) if regs else 0:.1%}-{max(regs) if regs else 0:.1%}"
          f"  never-served at the optimum {min(nn):.0f}-{max(nn):.0f}  argmins {sorted(ams)} rules {sorted(ls)}")

hdr("Sec. VII-F depot: rule one UAV too high on the ring; ex-ante changes per family")
for name in ("depot_q.csv", "depot_e.csv"):
    D = defaultdict(dict)
    for r in read(A(name)):
        D[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = r
    for fam in ("paper", "core", "ring"):
        ks = [k for k in D if k[0] == fam]
        if not ks: continue
        hi = sum(1 for k in ks if ell(D[k]) - am(D[k]) == 1)
        print(f"  {name} {fam}: rule exactly one above the optimum in {hi} of {len(ks)}")

hdr("Sec. VII-J ex-ante: P_bar-hat accuracy, r_max/r_c-hat")
X = [x for x in read(A("exante.csv"))]
pm = np.array([float(x["Pb_meas"]) for x in X]); pe = np.array([float(x["Pb_exante"]) for x in X])
print(f"  P_bar-hat: corr {np.corrcoef(pm, pe)[0,1]:.3f}, mean |error| {np.mean(abs(pm-pe)):.1f} W, measured range {pm.min():.0f}-{pm.max():.0f} W, best constant {np.mean(abs(pm-np.median(pm))):.1f} W")
for fam in ("paper", "ring", "core"):
    rr = [float(x["rc_exante"]) / float(x["rc_meas"]) for x in X if x["layout"] == fam]
    ro = [float(x["rc_oracle_n"]) / float(x["rc_meas"]) for x in X if x["layout"] == fam]
    print(f"  {fam}: r_c-hat/r_c median {np.median(rr):.2f} [{np.quantile(rr,.1):.2f},{np.quantile(rr,.9):.2f}], with measured n {np.median(ro):.2f}")
reach_cells = defaultdict(list)
for x in X:
    reach_cells[(x["layout"], x["M"], x["Emax"], x["L"])].append(float(x["ratio_exante"]))
rl = {c: np.mean(v) for c, v in reach_cells.items() if c[0] != "ring"}
print(f"  smallest ex-ante r_max/r_c over reach-limited cells {min(rl.values()):.1f}; core cells: " +
      " ".join(f"{c[1]}/{float(c[2])/1e6:.1f}:{v:.1f}" for c, v in sorted(rl.items()) if c[0] == "core"))
print(f"  core per-instance range: {min(float(x['ratio_exante']) for x in X if x['layout']=='core'):.1f}-{max(float(x['ratio_exante']) for x in X if x['layout']=='core'):.1f}")

# ------------------------------------------------------------------------------------------------ planners
hdr("Sec. VII-K: C4 vs SA per clustered cell (own optimum, median J_C4/J_SA - 1, C4 lower)")
for c in sorted({k[:3] for k in SA if k[0] == "paper" and k[1] <= 200}):
    ks = [k for k in SA if k[:3] == c and k in PL["cyc"]]
    v = [J(PL["cyc"][k], am(PL["cyc"][k])) / J(SA[k], am(SA[k])) - 1 for k in ks]
    print(f"  {c}: {np.median(v):+.1%}, C4 lower in {sum(1 for x in v if x < 0)}/{len(v)}")
hdr("Sec. VII-K: tour patrol and sweep patrol J vs SA by family (own optimum, median)")
for pl in ("rr_tour", "rr_sweep", "cluster_patrol"):
    for fam in ("paper", "ring", "core"):
        ks = [k for k in SA if k[0] == fam and k in PL[pl] and k[1] <= 200]
        if not ks: continue
        v = [J(PL[pl][k], am(PL[pl][k])) / J(SA[k], am(SA[k])) - 1 for k in ks]
        print(f"  {pl} {fam}: median {np.median(v):+.1%} (n={len(v)})")
hdr("Sec. VII-K: Gao et al. misses (|argmin - l| > 1)")
g = PL["gzz_rev"]; miss = [k for k in g if k in SA and k[1] <= 200 and abs(am(g[k]) - ell(SA[k])) > 1]
und3 = [k for k in miss if k[2] == 3e6 and k[0] in ("paper", "ring") and am(g[k]) < ell(SA[k])]
print(f"  misses {len(miss)}; undershoots at 3 MJ paper/ring {len(und3)} (by exactly two: {sum(1 for k in und3 if ell(SA[k]) - am(g[k]) == 2)});"
      f" core overshoots {sum(1 for k in miss if k[0]=='core' and am(g[k]) > ell(SA[k]))}; other {len(miss) - len(und3) - sum(1 for k in miss if k[0]=='core' and am(g[k]) > ell(SA[k]))}")
print(f"  of the 3 MJ undershoots, where the test passes or does not apply: {sum(1 for k in und3 if VER.get(k) in ('pass','not_applicable'))} (ring: {sum(1 for k in und3 if k[0]=='ring')})")
hdr("Liu et al.: C4 vs Liu on the 36 previously seen deployments (M=50, 1.5 MJ) and the rest")
lk = [k for k in PL["liu_mpga"] if k in PL["cyc"] and k[1] <= 200]
rel = {k: J(PL["cyc"][k], am(PL["cyc"][k])) / J(PL["liu_mpga"][k], am(PL["liu_mpga"][k])) - 1 for k in lk}
seen = [k for k in lk if k[1] == 50 and k[2] == 1.5e6]
print(f"  seen {len(seen)}: median {np.median([rel[k] for k in seen]):+.1%};  rest {len(lk)-len(seen)}: median {np.median([rel[k] for k in lk if k not in seen]):+.1%}")

hdr("App. B: visit-rate argmax (K n/T_s from each planner's own rows) within one of the AoI optimum, ring")
def visit_rate_argmax(b):
    g_ = {}
    for K, r in b.items():
        Ts = float(r["T_s_over_t_c"]) * P.t_c
        if Ts > 0: g_[K] = K * float(r["mean_n"]) / Ts
    return max(g_, key=g_.get)
for pl in ("sa", "cyc", "cluster_patrol", "gzz_rev", "asghar_rev", "liu_mpga", "rr_tour"):
    ks = [k for k in PL[pl] if k[0] == "ring" and k[1] <= 200]
    v = [abs(visit_rate_argmax(PL[pl][k]) - am(PL[pl][k])) <= 1 for k in ks]
    v3 = [abs(visit_rate_argmax(PL[pl][k]) - am(PL[pl][k])) <= 1 for k in ks if k[2] == 3e6]
    print(f"  {pl:15s} all ring {np.mean(v):.0%} (n={len(v)}); 3 MJ cells {np.mean(v3):.0%}")
hdr("App. B: optima on the 3 MJ ring cells per planner (cell means)")
for pl in ("sa", "cyc", "cluster_patrol", "asghar_rev", "gzz_rev", "liu_mpga", "rr_tour", "rr_sweep"):
    print(f"  {pl:15s} " + "  ".join(f"M={M}: {np.mean([am(PL[pl][k]) for k in PL[pl] if k[:3]==('ring',M,3e6)]):.2f}"
                                      for M in (50, 100, 200)))
hdr("App. B: stop count calibrated from each planner's two smallest stranding-free swept K (pilot runs), then argmax of K n/(t_c + n tau_s)")
def calibrated(b):
    rows = [(K, r) for K, r in sorted(b.items()) if float(r["n_never"]) == 0 and float(r["mean_n"]) > 0]
    if len(rows) < 2: return None
    (K1, r1), (K2, r2) = rows[0], rows[1]
    E = float(r1.get("Emax")); U = (1 - P.rho) * E
    tc_ = np.mean([2 * float(r["r_c"]) / P.v for _, r in (rows[0], rows[1])])
    x1, x2 = U / K1 - P.Pf * tc_, U / K2 - P.Pf * tc_
    if x1 <= 0 or x2 <= 0 or x1 == x2: return None
    a = math.log(float(r1["mean_n"]) / float(r2["mean_n"])) / math.log(x1 / x2); c = float(r1["mean_n"]) / x1 ** a
    taus = np.mean([(float(r["T_s_over_t_c"]) * P.t_c - 2 * float(r["r_c"]) / P.v) / float(r["mean_n"]) for _, r in (rows[0], rows[1])])
    def G_(K):
        e = U / K - P.Pf * tc_
        if e <= 0: return -1
        n = c * e ** a; return K * n / (tc_ + n * taus)
    return max(range(1, 100), key=G_)
for pl in ("sa", "cyc", "cluster_patrol", "asghar_rev", "gzz_rev", "liu_mpga"):
    for scope, f_ in (("all ring", lambda k: True), ("3 MJ ring", lambda k: k[2] == 3e6)):
        ks = [k for k in PL[pl] if k[0] == "ring" and k[1] <= 200 and f_(k)]
        cal = [calibrated(PL[pl][k]) for k in ks]
        v_cal = [abs(cv - am(PL[pl][k])) <= 1 for k, cv in zip(ks, cal) if cv is not None]
        v_cf = [abs(ell(SA[k]) - am(PL[pl][k])) <= 1 for k in ks]
        per = ""
        if scope == "3 MJ ring":
            per = "  per cell calibrated: " + " ".join(f"M={M}:{np.mean([abs(cv - am(PL[pl][k])) <= 1 for k, cv in zip(ks, cal) if cv is not None and k[1]==M]):.0%}" for M in (50, 100, 200)) + \
                  "  closed form: " + " ".join(f"M={M}:{np.mean([abs(ell(SA[k]) - am(PL[pl][k])) <= 1 for k in ks if k[1]==M]):.0%}" for M in (50, 100, 200))
        print(f"  {pl:15s} {scope:9s}: calibrated within one {sum(v_cal)}/{len(v_cal)}, closed form {sum(v_cf)}/{len(v_cf)}{per}")

# ------------------------------------------------------------------------------------------------ CP-SAT budgets
hdr("Sec. VII-K: CP-SAT budget study on the ring (J changes, argmin agreement with SA)")
d5 = inst([r for r in read(R("m40_cpsat_d5.csv")) if r["layout"] == "ring"]); d15 = inst(read(R("m40_cpsat_d15.csv"))); d45 = inst(read(R("m40_cpsat_d45.csv")))
sa40 = M40["sa"]
ch = [J(d15[k], K) / J(d5[k], K) - 1 for k in d15 if k in d5 for K in d15[k] if K in d5[k]]
print(f"  d5 -> d15: J change {min(ch):+.1%} to {max(ch):+.1%}, median {np.median(ch):+.1%} (n={len(ch)})")
ch2 = [(k, K, J(d45[k], K) / J(d15[k], K) - 1) for k in d45 for K in d45[k] if k in d15 and K in d15[k]]
print("  d15 -> d45: " + ", ".join(f"s{k[3]} K{K} {x:+.2%}" for k, K, x in ch2))
for lab, D in (("d5", d5), ("d15", d15)):
    ks = [k for k in D if k in sa40 and k[0] == "ring"]
    print(f"  ring argmin {lab} = SA in {np.mean([am(D[k]) == am(sa40[k]) for k in ks]):.2f} (n={len(ks)})")
ks = [k for k in d15 if k in sa40 and am(d15[k]) != am(sa40[k])]
for k in ks:
    b = d15[k]; a = am(b); s = am(sa40[k])
    print(f"  d15 differs from SA at {k}: {a} vs {s}; J(SA's K)/J(CP's K)-1 under CP-SAT {J(b, s)/J(b, a)-1:+.1%}; proven share at those K "
          f"{b[a]['cp_optimal']}/{b[a]['cp_decisions']} and {b[s]['cp_optimal']}/{b[s]['cp_decisions']}")

# ------------------------------------------------------------------------------------------------ decomposition
hdr("Sec. VII-K decomposition: ring tour patrol vs SA; J2/J_age; cluster vs SA")
rg = defaultdict(dict)
for r in read(R("regularity_ring.csv")): rg[(int(r["K"]), int(r["seed"]))][r["planner"]] = r
for (K, s), b in sorted(rg.items()):
    if "sa" not in b or "rr_tour" not in b: continue
    a, t = b["sa"], b["rr_tour"]; f = lambda c: float(t[c]) / float(a[c])
    meas = f("J_age"); rate = float(a["Gamma"]) / float(t["Gamma"]); alloc = f("E_alloc"); regl = f("E_reg")
    print(f"  K={K} s={s}: J_age ratio patrol/SA {meas:.3f}; rate factor {rate:.3f}, alloc {alloc:.3f}, reg {regl:.3f}; product {rate*alloc*regl:.3f}"
          f"  (tau_hop {float(t['tau_hop']):.1f} vs {float(a['tau_hop']):.1f} s)")
chk = []; lab = []
for n in ("regularity_ring.csv", "regularity_core_cl.csv", "regularity_paper_cl.csv"):
    for r in read(R(n)):
        chk.append(abs(float(r["J2_alloc_cv"]) / float(r["J_age"]) - 1)); lab.append(f"{r['planner']} {r['layout']} K={r['K']} s={r['seed']}")
chk = np.array(chk)
print(f"  J2 vs J_age over {len(chk)} decomposed runs: within 0.5% in {int((chk <= .005).sum())}, within 0.6% in {int((chk <= .006).sum())},"
      f" max deviation {chk.max():.1%}")
print("  runs outside 0.6%: " + "; ".join(f"{l} ({c:.1%})" for l, c in zip(lab, chk) if c > .006))
for n in ("regularity_core_cl.csv", "regularity_paper_cl.csv"):
    for r in read(R(n)):
        print(f"  {n} {r['planner']:14s} s={r['seed']}: r_c {float(r['r_c']):.0f} m  Gamma {float(r['Gamma'])*3600:.0f}/h  tau_hop {float(r['tau_hop']):.1f} s  E_reg {float(r['E_reg']):.3f}  E_alloc {float(r['E_alloc']):.3f}  J {float(r['J']):.3e}")

hdr("Sec. V: sufficient-type criterion variants on the zero-regret clustered instances (criterion_variants.py forms)")
_argv = sys.argv; sys.argv = ["criterion_variants.py", A("grid_final.csv"), "-"]
_g = {}; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "criterion_variants.py")).read().split("def row(lab, R):")[0], _g)
sys.argv = _argv
for v in ("corrected_KR", "KG", "global"):
    z = [t for t in _g["res"][v] if t[0] == "paper" and t[2] == 0]
    print(f"  {v}: zero-regret clustered {len(z)}, flagged {sum(1 for t in z if t[1] is False)} ({np.mean([t[1] is False for t in z]):.0%})")

hdr("Sec. VII-F depot: ex-ante rule changes per family (analyze_depot.py) and sweeps extended by the fills")
import subprocess, re
for name in ("depot_q.csv", "depot_e.csv"):
    if not os.path.exists(A(name)): continue
    out = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_depot.py"), A(name)],
                         capture_output=True, text=True).stdout
    ch = defaultdict(int)
    for l in out.splitlines():
        m = re.match(r"(\w+) M=(\d+) .*argmin (\d+)\s+rule (\d+) \(ex-ante (\d+)\)", l)
        if m and m.group(4) != m.group(5):
            a_, lm, le = int(m.group(3)), int(m.group(4)), int(m.group(5))
            ch[(m.group(1), "toward the optimum" if abs(le - a_) < abs(lm - a_) else "away or level")] += 1
    print(f"  {name}: ex-ante changes {dict(ch)}")
for name in ("depot_q_fill.csv", "depot_e_fill.csv"):
    print(f"  {name}: {len({(r['layout'], r['M'], r['Emax'], r['seed']) for r in read(R(name))})} annealer sweeps extended")

hdr("Sec. VII-F depot: C4 vs cluster patrol own optima, censoring (argmin at an end of the sweep) and the comparison without censored deployments")
for tag in ("q", "e"):
    own = {}
    for pl, name in (("cyc", f"depot_{tag}_cyc.csv"), ("cluster_patrol", f"depot_{tag}_cluster_patrol.csv")):
        d = defaultdict(dict)
        for r in read(A(name)):
            if r.get("planner") == pl: d[(r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))][int(float(r["K"]))] = float(r["J"])
        own[pl] = {k: (min(b, key=b.get), b[min(b, key=b.get)], (min(b, key=b.get) == max(b)) or (min(b, key=b.get) == min(b) and min(b) > 1)) for k, b in d.items()}
    ks = [k for k in own["cyc"] if k in own["cluster_patrol"]]
    cen = [k for k in ks if own["cyc"][k][2] or own["cluster_patrol"][k][2]]
    print(f"  depot_{tag}: {len(ks)} paired deployments; censored own optima: C4 {sum(own['cyc'][k][2] for k in ks)}, cluster patrol "
          f"{sum(own['cluster_patrol'][k][2] for k in ks)} (" + ", ".join(f"{k[0]} M={k[1]} {k[2]/1e6:.1f} MJ s={k[3]}" for k in cen) + ")")
    for fam in sorted({k[0] for k in ks}):
        for lab, sel in (("all", [k for k in ks if k[0] == fam]), ("uncensored", [k for k in ks if k[0] == fam and k not in cen])):
            v = np.array([own["cyc"][k][1] / own["cluster_patrol"][k][1] - 1 for k in sel])
            print(f"    {fam:5s} {lab:10s}: median {np.median(v):+.1%}, C4 lower {int((v < 0).sum())}/{len(v)}")

hdr("Sec. VII-I: mean K_commute at M=100, 1.5 MJ (smallest-K inputs)")
for c in [("paper", 100, 1.5e6), ("core", 100, 1.5e6)]:
    print(f"  {c}: {np.mean([float(b[min(b)]['K_commute_i']) for k, b in SA.items() if k[:3] == c]):.1f}")

hdr("Sec. VII-K C4: vs cluster patrol per ring cell and largest loss; rule agreement per family vs cluster (paired bootstrap)")
c4, cl = PL["cyc"], PL["cluster_patrol"]
for c in sorted({k[:3] for k in c4 if k[0] == "ring"}):
    ks = [k for k in c4 if k[:3] == c and k in cl]
    v = [J(c4[k], am(c4[k])) / J(cl[k], am(cl[k])) - 1 for k in ks]
    print(f"  {c}: median {np.median(v):+.1%}, C4 lower {sum(x < 0 for x in v)}/{len(v)}, worst {max(v):+.1%}")
allv = [J(c4[k], am(c4[k])) / J(cl[k], am(cl[k])) - 1 for k in c4 if k in cl]
print(f"  largest single loss of C4 against the cluster patrol: {max(allv):+.1%}")
rng = np.random.default_rng(20261011)
for fam in ("paper", "ring", "core", None):
    ks = [k for k in c4 if k in cl and k in SA and (fam is None or k[0] == fam)]
    a = np.array([abs(am(c4[k]) - ell(SA[k])) <= 1 for k in ks], float); b = np.array([abs(am(cl[k]) - ell(SA[k])) <= 1 for k in ks], float)
    e = np.array([am(c4[k]) == ell(SA[k]) for k in ks], float)
    cells = defaultdict(list)
    for i, k in enumerate(ks): cells[k[:3]].append(i)
    reps = []
    for _ in range(4000):
        idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()]); reps.append(a[idx].mean() - b[idx].mean())
    print(f"  {fam or 'all'}: C4 within one {a.mean():.0%} (exact {e.mean():.0%}), cluster {b.mean():.0%}; difference {100*(a.mean()-b.mean()):+.1f} points [{100*np.quantile(reps,.025):+.1f}, {100*np.quantile(reps,.975):+.1f}]")
