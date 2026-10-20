"""commute_exact.py -- Prop. tau made exact (review H2): with hover at Ph, hops at Pf and the fleet's hover rate fixed at
H = sum_i lambda_i / R_u by conservation, the productive power is not a free input: Pbar(K) = Pf + (Ph - Pf) H / Phi(K).
The sortie lasts U / (K Pf + (Ph - Pf) H), so Phi(K) = K (U - K Pf t_c - (Ph - Pf) H t_c) / U, maximised at
    K_H = (U / t_c - (Ph - Pf) H) / (2 Pf).
This script (1) checks the conservation law on the stranding-free primary runs (hover rate implied by the measured Pbar
against sum lambda / R_u); (2) compares K_H with the paper's K_commute (Pbar held fixed) per instance, with the commute
inputs read at the reference K; (3) reports how often the rule ell changes and the agreement with the annealer's optimum.

usage: python commute_exact.py grid_final.csv
"""
import sys, csv, math, collections
import numpy as np
from dyn_env import DynParams, SensorField
from kref import kref

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("planner", "sa") == "sa" and float(r["Th"]) == 43200
        and r["coord"] == "exclude" and r["replan"] == "launch"]
print("(1) conservation of hover time: implied hover rate Phi (Pbar - Pf)/(Ph - Pf) over sum_i lambda_i / R_u, with each")
print("    instance's own generation rates lambda_i (regenerated from its seed); stranding-free runs, L = 12.6 km;")
print("    percentiles 10/50/90. Also shown: the nominal M lambda-bar (as in v2.2.1) and the filter on < 5% empty sorties.")
lamsum = {}
def H_actual(r, p):
    key = (r["layout"], p.M, p.Emax, int(float(r["seed"])))
    if key not in lamsum:
        lamsum[key] = float(SensorField(DynParams(M=p.M, Emax=p.Emax, layout=r["layout"]), np.random.default_rng(key[3])).lam_bits.sum())
    return lamsum[key] / p.R
agg = collections.defaultdict(list)
for r in rows:
    if float(r["L"]) != 12600 or float(r["n_never"]) > 0: continue
    p = DynParams(M=int(float(r["M"])), Emax=float(r["Emax"]), layout=r["layout"])
    K = int(float(r["K"])); Ts = float(r["T_s_over_t_c"]) * p.t_c; tc = 2 * float(r["r_c"]) / p.v
    Phi = K * (Ts - tc) / Ts; hov = Phi * (float(r["P_bar"]) - p.Pf) / (p.Ph - p.Pf)
    agg[r["layout"]].append((hov / H_actual(r, p), hov / (p.M * p.lam_bits / p.R), float(r["empty_frac"]) < 0.05))
pc = lambda v: f"{np.percentile(v, 10):.2f} / {np.median(v):.2f} / {np.percentile(v, 90):.2f}"
allx = []
for k in ("paper", "ring", "core"):
    v = np.array(agg[k]); f = v[:, 2] > 0; allx.append(v)
    print(f"    {k:5s} actual rates: all n={len(v):4d} {pc(v[:, 0])} | <5% empty n={f.sum():4d} {pc(v[f, 0])}"
          f"   || nominal rate, <5% empty: {pc(v[f, 1])}")
v = np.vstack(allx); f = v[:, 2] > 0
print(f"    pooled actual rates: all {pc(v[:, 0])}; <5% empty {pc(v[f, 0])} | nominal, <5% empty {pc(v[f, 1])}")

by = collections.defaultdict(dict)
for r in rows: by[(r["layout"], int(float(r["M"])), float(r["Emax"]), float(r["L"]), int(float(r["seed"])))][int(float(r["K"]))] = r
print("\n(2)-(3) K_H against K_commute (Pbar fixed), per instance; rule ell = min(floor K_reach, round K_commute)")
tot = collections.Counter(); cell = collections.defaultdict(list)
for key, b in sorted(by.items()):
    r = b[kref(b)]
    p = DynParams(M=key[1], Emax=key[2], layout=key[0], L=key[3])
    U = (1 - p.rho) * p.Emax; tc = 2 * float(r["r_c"]) / p.v
    if tc <= 0: continue
    H = p.M * p.lam_bits / p.R
    KH = (U / tc - (p.Ph - p.Pf) * H) / (2 * p.Pf)
    Kc = float(r["K_commute_i"]); Kr = float(r["K_reach_i"])
    ell = min(math.floor(Kr), round(Kc)); ellH = min(math.floor(Kr), round(KH))
    kst = min(b, key=lambda K: float(b[K]["J"]))
    cell[key[:4]].append((KH / Kc - 1, ell != ellH, ell == kst, ellH == kst, abs(ell - kst) <= 1, abs(ellH - kst) <= 1,
                          (p.Ph - p.Pf) * H / (2 * p.Pf)))
allv = [x for v in cell.values() for x in v]
print(f"  instances {len(allv)}: K_H / K_commute - 1 median {100*np.median([x[0] for x in allv]):+.1f}% "
      f"(range {100*min(x[0] for x in allv):+.1f}% .. {100*max(x[0] for x in allv):+.1f}%)")
print(f"  dwell shift (Ph - Pf) H / (2 Pf) = {min(x[6] for x in allv):.3f} .. {max(x[6] for x in allv):.3f} UAV")
print(f"  ell changes in {sum(x[1] for x in allv)} of {len(allv)}; exact agreement {np.mean([x[2] for x in allv]):.2f} -> "
      f"{np.mean([x[3] for x in allv]):.2f}; within one {np.mean([x[4] for x in allv]):.2f} -> {np.mean([x[5] for x in allv]):.2f}")
for c in sorted(cell):
    v = cell[c]
    if any(x[1] for x in v):
        print(f"    {c[0]:5s} M={c[1]:3d} {c[2]/1e6:.1f} MJ L={c[3]/1e3:.1f} km: ell changes {sum(x[1] for x in v)}/{len(v)}, "
              f"exact {sum(x[2] for x in v)} -> {sum(x[3] for x in v)}, within one {sum(x[4] for x in v)} -> {sum(x[5] for x in v)}")
