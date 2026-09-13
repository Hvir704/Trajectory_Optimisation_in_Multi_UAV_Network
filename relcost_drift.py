"""relcost_drift.py -- Prop. 4 relative-cost profile invariance (§V-B).

usage: python relcost_drift.py ["nodes_paper_M100_*_K*_s*_exclude_launch.npz"]

What this measures, and why the obvious version does not work
-------------------------------------------------------------
c_i = 2 P_f r_i / v depends on r_i only, and SensorField is built from (seed, M, layout) -- never
from K. So within a seed the radius vector is BYTE-IDENTICAL across K. Any comparison that masks
two K's radius vectors to a common support and normalises each by the mean OVER THAT SUPPORT is
comparing a vector to itself, and returns exactly 0.0 no matter what the data does.

The quantity that actually varies with K is the served SET: raising K shrinks E_u = U/K, so fewer
and nearer sensors get visited. That moves the normalising constant c-bar(K) = mean of c over the
nodes served at K. So for nodes served at BOTH K and K0, this script compares

    a_i = c_i / c-bar(K)        against        b_i = c_i / c-bar(K0)

and reports mean |a_i - b_i|. Pairs are matched by seed, then averaged across seeds.

Interpretation: the profile SHAPE is K-invariant by construction; what drifts is the scale, driven
entirely by which sensors get served. Report the number as evidence about set composition, not as an
independent confirmation of Prop. 4's invariance assumption.
"""
import sys, glob, re, collections
import numpy as np
from dyn_env import DynParams

pat = sys.argv[1] if len(sys.argv) > 1 else "nodes_paper_M100_*_K*_s*_exclude_launch.npz"
p = DynParams()

byK = collections.defaultdict(dict)          # K -> {seed: filename}
for f in sorted(glob.glob(pat)):
    m = re.search(r"_K(\d+)_s(\d+)_", f)
    if not m: continue
    byK[int(m.group(1))][int(m.group(2))] = f
if not byK:
    print(f"no files match {pat}"); sys.exit(1)

K0 = min(byK)
print(f"{'K':>3}{'seeds':>7}{'served':>9}{'mean r (m)':>12}{'c-bar ratio':>13}{'drift vs K0':>13}")
for K in sorted(byK):
    drifts, served_frac, rbar, ratios = [], [], [], []
    for seed, f in sorted(byK[K].items()):
        f0 = byK[K0].get(seed)
        if f0 is None: continue
        d, d0 = np.load(f), np.load(f0)
        c, c_0 = 2 * p.Pf * d["r"] / p.v, 2 * p.Pf * d0["r"] / p.v
        sK, s0 = d["visits"] > 0, d0["visits"] > 0
        if sK.sum() == 0 or s0.sum() == 0: continue
        served_frac.append(float(sK.mean())); rbar.append(float(d["r"][sK].mean()))
        cbarK, cbar0 = c[sK].mean(), c_0[s0].mean()
        ratios.append(float(cbarK / cbar0))
        both = sK & s0
        if both.sum() == 0: continue
        drifts.append(float(np.abs(c[both] / cbarK - c_0[both] / cbar0).mean()))
    if not drifts: continue
    print(f"{K:>3}{len(drifts):>7}{np.mean(served_frac):>9.3f}{np.mean(rbar):>12.0f}"
          f"{np.mean(ratios):>13.4f}{np.mean(drifts):>13.4f}")

print(f"\nc-bar ratio < 1 means the served set at that K is closer to the depot than at K={K0}.")
print("Drift is driven entirely by that scale change -- the profile shape is K-invariant by")
print("construction, since c_i depends only on r_i and the layout does not depend on K.")