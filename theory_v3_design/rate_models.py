"""Visit-rate models against the measured rate on the existing grid (SA, design seeds 1-12, L = 12.6 km, 12 h).
measured  Gamma_m = K * mean_n / T_s,  T_s = (T_s/t_c) x nominal t_c   [visits per second, fleet]
M0 (v3)   Gamma = v (Phi(K; rc0, Pb0) - H_S) / d_S,   rc0, Pb0 = exante at K0 (first K of the original sweep)
M1        as M0 with rc, Pb = exante at K (the paper's own estimator, evaluated at each K)
M2        as M1 with hop d_S / sqrt(1 - c), c = min(0.9, (K - 1) * n(K) / 2 / |S|)  (other UAVs' claims thin the field)
M3        as M1 with hop d_S * sqrt(|S| / (|S| - (K-1) n / 2))  -- same as M2 written differently (check)
Only stranding-free rows (n_never = 0) are used, so S = all sensors.
"""
import csv, collections, numpy as np, sys, os, pickle, math
sys.path.insert(0, ".")
import theory_v3
from theory_v3 import field, tour_len
from exante_rc import exante
if os.path.exists("tour_cache_v3.pkl"): theory_v3.TOUR_CACHE.update(pickle.load(open("tour_cache_v3.pkl", "rb")))
rows = collections.defaultdict(dict)
for r in csv.DictReader(open(os.environ.get('V3_ASSEMBLED', 'assembled') + '/grid_final.csv')):
    if r.get('planner','sa')!='sa' or r['coord']!='exclude' or r['replan']!='launch' or float(r['Th'])!=43200 or float(r['L'])!=12600: continue
    rows[(r['layout'],int(r['M']),float(r['Emax']),int(r['seed']))][int(r['K'])]=r
out = collections.defaultdict(list)
for k, b in sorted(rows.items()):
    lay, M, E, s = k
    if M == 400 or E == 6e6: continue
    p, F = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F.pos - p.home, axis=1); lam = np.full(M, p.lam_bits)
    K0 = int(float(next(iter(b.values()))['K_ref'])); rc0, Pb0, _ = exante(p, r, lam, F.pos, K0)
    key = (lay, M, 12600.0, s, 0)
    dS = (theory_v3.TOUR_CACHE[key] if key in theory_v3.TOUR_CACHE else tour_len(F.pos, p.home)) / M
    H = M * p.lam_bits / p.R; U = (1 - p.rho) * E
    def Phi(K, rc, Pb):
        tc = 2 * rc / p.v; return K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf))
    for K, x in b.items():
        if int(float(x['n_never'])) != 0 or float(x['mean_n']) <= 0: continue
        Ts = float(x['T_s_over_t_c']) * p.t_c; Gm = K * float(x['mean_n']) / Ts
        rcK, PbK, nK = exante(p, r, lam, F.pos, K)
        g0 = p.v * (Phi(K, rc0, Pb0) - H) / dS
        g1 = p.v * (Phi(K, rcK, PbK) - H) / dS
        c = min(0.9, (K - 1) * nK / 2 / M); g2 = g1 * math.sqrt(1 - c)
        for nm, g in (("M0", g0), ("M1", g1), ("M2", g2)):
            if g > 0: out[(lay, E, nm)].append(g / Gm); out[(lay, E, nm, 'K')].append((K, g / Gm))
pickle.dump(theory_v3.TOUR_CACHE, open("tour_cache_v3.pkl", "wb"))
print("ratio model / measured visit rate (= k_rate), stranding-free SA rows; median [10th, 90th], and Spearman vs K")
from itertools import groupby
for lay in ("paper", "ring", "core"):
    for E in (1.5e6, 3e6):
        line = f"{lay:5s} {E/1e6:.1f} MJ"
        for nm in ("M0", "M1", "M2"):
            v = np.array(out[(lay, E, nm)]); kv = np.array(out[(lay, E, nm, 'K')])
            rk = lambda a: np.argsort(np.argsort(a))
            rho = np.corrcoef(rk(kv[:,0]), rk(kv[:,1]))[0,1]
            line += f" | {nm} {np.median(v):.2f} [{np.quantile(v,.1):.2f},{np.quantile(v,.9):.2f}] rho_K {rho:+.2f}"
        print(line + f"  n={len(v)}")
