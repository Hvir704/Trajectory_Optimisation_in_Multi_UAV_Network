"""theory_v3.py -- the v3 fleet-sizing theory: served-set surrogate rule and commute bracket.

Everything here is computable before deployment from the site survey (sensor positions, priorities), nominal
generation rates, the airframe (v, Pf, Ph, R, B) and the energy inventory. No simulation output is read.

Surrogate (Props. sqrt and tau of the paper, combined):
  served set at fleet size K   R(K) = {i : r_i <= r_reach(K)},  r_reach(K) = v (U/K - Ph B / R) / (2 Pf)
  visit rate                   Gamma_S(K) = v (Phi(K) - H_S) / d_S,  H_S = sum_{i in S} lam_i / R (nominal lam),
                               d_S = (length of a closed NN + 2-opt tour through S) / |S|
  served-set age               A_S B_S / (2 Gamma_S(K)),  A_S = sum sqrt(w pi), B_S = sum sqrt(w / pi), pi_i = r_i / mean_S r
  stranded age                 (T_b + T/2) sum_{i not in S} w_i   (age integral of a never-served sensor over the window)
  K_hat = argmin_K [ served + stranded ]  over integers 1 <= K <= Kmax (default 200) with |R(K)| >= 2 and
  Phi(K) > H_R(K); the scan stops once K is infeasible past the peak of Phi.
Structure (Theorem 1 in THEORY_V3.md): the minimiser is one of the nested candidates, the integer argmax K_c* of Phi
clipped to each interval of K on which the served set is S_j = all sensors but the j farthest; j = 0 is the paper's
rule. (round(Kc) = K_c* in every instance used here; the theorem does not rely on it.)

Commute bracket (Prop. 2, conditional): if age is minimised where the visit rate peaks (a) and tau_hop(K) >= tau_hop(K_c*)
for K < K_c* (b), some optimum lies at or above K_c*; without stranding, [K_c*, floor(Kr)].

Commute inputs (r_c, P_bar) are the paper's ex-ante estimates (exante_rc.exante), passed in by the caller at the first K
of the instance's original sweep (the registered held-out scripts: min k_range; retro_v3 / horizon_v3: K_ref).
`horizon` is the mission length T_b + T in seconds (None = the run's 12 h).
"""
import math
import numpy as np
from dyn_env import DynParams, SensorField
from rr_planner import tour_order


def field(layout, M, Emax, L, seed):
    p = DynParams(M=M, Emax=Emax, layout=layout, L=L)
    F = SensorField(DynParams(M=M, Emax=Emax, layout=layout, L=L), np.random.default_rng(seed))
    return p, F


def tour_len(pos, home):
    o = tour_order(pos, home); P = pos[o]
    return float(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1).sum())


def caps(p, r, rc, Pb):
    U = (1 - p.rho) * p.Emax
    Kc = U * p.v / (2 * rc * (p.Pf + math.sqrt(p.Pf * Pb)))
    Kr = p.v * U / (2 * p.Pf * r.max() + p.v * p.Ph * p.B_bits / p.R)
    return Kc, Kr


def old_rule(p, r, rc, Pb):
    Kc, Kr = caps(p, r, rc, Pb)
    return int(min(math.floor(Kr), round(Kc)))


TOUR_CACHE = {}          # optional: {(layout, M, L, seed, j): total tour length}; filled by callers that persist it


def surrogate_rule(p, F, rc, Pb, scale=1.0, Kmax=200, return_curve=False, key=None, horizon=None, gray="charge"):
    """argmin_K of the served-set surrogate; returns (K_hat, n_stranded_at_K_hat[, curve]). Brute force over integer K,
    so the result does not rely on the candidate-set theorem.
    `scale` multiplies the served term (1 = a priori; a pilot sets it from one measured run).
    `gray`: "charge" (v3, registered): every sensor outside the full-buffer reach counts as stranded.
            "exclude" (v3.3): a fleet size with j > 0 is a candidate only if no sensor lies in its gray bracket
            r_reach(K) < r_i <= r_flight(K) = v U / (2 Pf K), i.e. every stranded sensor is beyond flight-only reach
            and so cannot draw energy from the fleet. Fleet sizes with j = 0 are unaffected."""
    U = (1 - p.rho) * p.Emax; tc = 2 * rc / p.v
    r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base; lam = np.full(len(r), p.lam_bits)
    # mission window: burn-in T_b, then T scored. `horizon` (s) overrides the total T_b + T (the rule is for a
    # mission of that length; as horizon -> infinity stranding never pays and the rule tends to j = 0, the plain rule)
    Tb = p.T_burnin; T = (p.T_horizon if horizon is None else horizon) - Tb
    Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    tours = {}; curve = {}; best = None
    for K in range(1, Kmax + 1):
        reach = p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf)
        keep = r <= reach
        if keep.sum() < 2: break
        j = int((~keep).sum())
        if gray == "exclude" and j > 0 and np.any(r[~keep] <= p.v * U / (2 * p.Pf * K)):
            continue                                       # stranded set contains gray-bracket sensors: not a candidate
        if j not in tours:
            ck = None if key is None else key + (j,)
            if ck is not None and ck in TOUR_CACHE: tl = TOUR_CACHE[ck]
            else:
                tl = tour_len(F.pos[keep], p.home)
                if ck is not None: TOUR_CACHE[ck] = tl
            tours[j] = tl / keep.sum()
        g = p.v * (Phi(K) - lam[keep].sum() / p.R) / tours[j]
        if g <= 0:
            if K > 1 and Phi(K) < Phi(K - 1): break       # past the commute peak and infeasible: stop
            continue
        pi = r[keep] / r[keep].mean()
        served = scale * np.sqrt(w[keep] * pi).sum() * np.sqrt(w[keep] / pi).sum() / (2 * g)
        Jk = served + (Tb + T / 2) * w[~keep].sum(); curve[K] = (Jk, j, served)
        if best is None or Jk < best[0]: best = (Jk, K, j)
    return (best[1], best[2], curve) if return_curve else (best[1], best[2])


def pilot_scale(p, F, rc, Pb, K_pilot, J_age_pilot, key=None):
    """Served-term scale from one measured run at K_pilot (stranding-free): measured J_age / surrogate served term."""
    _, _, curve = surrogate_rule(p, F, rc, Pb, 1.0, Kmax=max(K_pilot, 1), return_curve=True, key=key)
    if K_pilot not in curve or curve[K_pilot][1] != 0: return 1.0
    return J_age_pilot / curve[K_pilot][2]


def bracket(p, r, rc, Pb):
    Kc, Kr = caps(p, r, rc, Pb)
    return (int(round(Kc)), int(math.floor(Kr))) if Kc < Kr else None
