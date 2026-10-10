"""mass_static.py -- the static battery-mass numbers of Remark 1 (no simulation).

The pack holds E_max/K per airframe (usable U/K plus its share of the reserve). Take-off mass
m(K) = m_dry + (E_max/K)/e_b; Pf = 300 (m/m_ref)^gamma, Ph = 400 (m/m_ref)^1.5, m_ref = m_dry + m_bref.
For each primary instance (L = 12.6 km) prints, for gamma in {0.5, 1.5}: the battery share of take-off mass at the
constant-power reach cap, the fixed point K = vU/(2 Pf(K) r_max + v Ph(K) B/R_u) relative to the constant-power K_r,
and how many instances change floor(K_r).
usage: python mass_static.py GRID.csv [--pack usable]   (--pack usable reproduces the old U/K sizing)
"""
import sys, csv, math
import numpy as np
from dyn_env import DynParams

usable = "--pack" in sys.argv and sys.argv[sys.argv.index("--pack") + 1] == "usable"
mdry, mbref, e_b = 2.0, 0.5, 180 * 3600.0; mref = mdry + mbref
inst = {}
for r in csv.DictReader(open(sys.argv[1])):
    if float(r.get("L", 12600)) != 12600: continue
    k = (r["layout"], int(float(r["M"])), float(r["Emax"]), int(float(r["seed"])))
    inst.setdefault(k, float(r["r_max"]))
p = DynParams()
for gam in (0.5, 1.5):
    share, ratio, changed = [], [], 0
    for (lay, M, E, s), rmax in inst.items():
        U = (1 - p.rho) * E; pack = (U if usable else E)
        m = lambda K: mdry + (pack / K) / e_b
        Pf = lambda K: 300.0 * (m(K) / mref) ** gam; Ph = lambda K: 400.0 * (m(K) / mref) ** 1.5
        Kr = p.v * U / (2 * 300.0 * rmax + p.v * 400.0 * p.B_bits / p.R)
        g = lambda K: p.v * U / (2 * Pf(K) * rmax + p.v * Ph(K) * p.B_bits / p.R)
        lo, hi = 0.2 * Kr, 5 * Kr
        for _ in range(100):
            mid = (lo + hi) / 2; lo, hi = (mid, hi) if g(mid) > mid else (lo, mid)
        Kf = (lo + hi) / 2
        share.append((pack / Kr) / e_b / m(Kr)); ratio.append(Kf / Kr - 1); changed += math.floor(Kf) != math.floor(Kr)
    print(f"gamma={gam}: n={len(share)}  battery share of take-off mass at K_r {min(share):.1%}-{max(share):.1%} "
          f"(median {np.median(share):.1%}); fixed point vs K_r median {np.median(ratio):+.1%} "
          f"(range {min(ratio):+.1%} to {max(ratio):+.1%}); floor(K_r) changes in {changed}")
