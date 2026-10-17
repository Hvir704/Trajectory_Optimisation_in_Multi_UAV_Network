"""theory_v34.py -- v3.4: the v3.3 rule with a crowding-corrected hop, for planners whose UAVs share the field.

theory_v3.py stays byte-identical to the version registered in the second held-out test (manifest 9497b1f4...); this
module adds one thing.

Crowding hop. In v3/v3.3 the a-priori hop is d_S = (closed NN + 2-opt tour through S) / |S| at every K. When the K
UAVs share the field under the exclusion protocol, the other K - 1 airborne UAVs hold on average half their routes, a
share x = (K - 1) n(K) / (2 |S|) of the served sensors, which a launching UAV may not choose. Thinning a point set by x
stretches nearest-neighbour spacing by 1 / sqrt(1 - x), so the hop is d_S / sqrt(1 - x), x capped at 0.9. n(K) is the
a-priori stop count at that hop, n = (U/K - Pf t_c) / (P_bar tau_s), tau_s = (hop / v) / (1 - h), h the hover share
(fixed point). No fitted constant.

What it gets right and wrong (SA, design grid, stranding-free runs; scratch/kappa/hop_check.py):
- LEVEL: model hop / measured hop has median 1.04-1.11 per cell (the uniform hop: 0.60-0.95, worst at core 3 MJ).
- TREND with K: measured hop falls with K in every cell (within-instance Spearman -0.80 to -1.00). The model's hop
  falls with K only on the ring at 3 MJ (-0.74), where n falls faster than 1/K; that is what moves the ring optimum
  above K_c. On the core and clustered layouts the model's hop RISES with K (+1.00). There the measured fall has
  another cause (smaller budgets select tighter groups of stops, Sec. V-B), which this model does not contain.

For fixed-territory planners (C4, cluster patrol, Liu et al., Gao et al., Asghar et al.) no other UAV enters a
territory: x = 0 and v3.4 is v3.3. The planner's coordination model is a declared input.
"""
import math
import numpy as np
import theory_v3
from theory_v3 import field, old_rule, tour_len, bracket, caps          # re-exported

SHARED = {"sa", "rr_tour", "rr_sweep", "greedy"}


def surrogate_rule(p, F, rc, Pb, scale=1.0, Kmax=200, return_curve=False, key=None, horizon=None, gray="exclude",
                   shared=True):
    """argmin_K of the served-set surrogate with gray exclusion (v3.3) and, if `shared`, the crowding hop (v3.4)."""
    U = (1 - p.rho) * p.Emax; tc = 2 * rc / p.v
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; lam = np.full(len(r), p.lam_bits)
    Tb = p.T_burnin; T = (p.T_horizon if horizon is None else horizon) - Tb
    Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    tours = {}; curve = {}; best = None
    for K in range(1, Kmax + 1):
        reach = p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
        keep = r <= reach
        if keep.sum() < 2: break
        j = int((~keep).sum())
        if gray == "exclude" and j > 0 and np.any(r[~keep] <= p.v * U / (2 * p.Pf * K)):
            continue
        if j not in tours:
            ck = None if key is None else key + (j,)
            if ck is not None and ck in theory_v3.TOUR_CACHE: tl = theory_v3.TOUR_CACHE[ck]
            else:
                tl = tour_len(F.pos[keep], p.home)
                if ck is not None: theory_v3.TOUR_CACHE[ck] = tl
            tours[j] = tl / keep.sum()
        S_n = int(keep.sum()); H = lam[keep].sum() / p.R; d = tours[j]
        if shared:
            h = min(max(H / max(Phi(K), 1e-9), 0.0), 0.95)
            for _ in range(20):
                n = max((U / K - p.Pf * tc) / (Pb * (d / p.v) / (1 - h)), 1.0)
                d = tours[j] / math.sqrt(1 - min(0.9, (K - 1) * n / (2 * S_n)))
        g = p.v * (Phi(K) - H) / d
        if g <= 0:
            if K > 1 and Phi(K) < Phi(K - 1): break
            continue
        pi = r[keep] / r[keep].mean()
        served = scale * np.sqrt(w[keep] * pi).sum() * np.sqrt(w[keep] / pi).sum() / (2 * g)
        Jk = served + (Tb + T / 2) * w[~keep].sum(); curve[K] = (Jk, j, served, d)
        if best is None or Jk < best[0]: best = (Jk, K, j)
    return (best[1], best[2], curve) if return_curve else (best[1], best[2])
