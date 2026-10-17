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
    K0 = int(float(next(iter(b.values()))['K_ref'])); rc0, Pb0, n0 = exante(p, r, lam, F.pos, K0)
    dS = theory_v3.TOUR_CACHE[(lay, M, 12600.0, s, 0)] / M
    H = M * p.lam_bits / p.R; U = (1 - p.rho) * E
    for K, x in b.items():
        if int(float(x['n_never'])) != 0 or float(x['mean_n']) <= 0: continue
        n = float(x['mean_n']); Ts = float(x['T_s_over_t_c']) * p.t_c; rcm = float(x['r_c']); Pbm = float(x['P_bar'])
        tcm = 2 * rcm / p.v; h = (Pbm - p.Pf) / (p.Ph - p.Pf)
        tau_m = max(Ts - tcm, 1e-9) * (1 - h) / n                    # measured hop time per stop
        tc0 = 2 * rc0 / p.v
        Phi0 = K * (U - K * p.Pf * tc0) / (U + K * tc0 * (Pb0 - p.Pf))
        Phim = K * (Ts - tcm) / Ts                                    # measured productive UAV-equivalents
        out[(lay, E, 'hop')].append(tau_m / (dS / p.v))               # > 1: real hops longer than d_S
        out[(lay, E, 'phi')].append((Phi0 - H) / max(Phim - H, 1e-9))  # > 1: surrogate's productive time too high
        out[(lay, E, 'rc')].append(rcm / rc0)
        out[(lay, E, 'n')].append(n)
        out[(lay, E, 'nhop')].append((n, tau_m / (dS / p.v)))
print("per stranding-free SA run: medians [10th, 90th]")
for lay in ("paper", "ring", "core"):
    for E in (1.5e6, 3e6):
        f = lambda nm: (lambda v: f"{np.median(v):.2f} [{np.quantile(v,.1):.2f},{np.quantile(v,.9):.2f}]")(np.array(out[(lay,E,nm)]))
        nh = np.array(out[(lay,E,'nhop')]); sl = np.polyfit(np.log(nh[:,0]), np.log(nh[:,1]), 1)[0]
        print(f"{lay:5s} {E/1e6:.1f} MJ  hop_meas/d_S {f('hop')}  (Phi-H)_sur/(Phi-H)_meas {f('phi')}  rc_meas/rc0 {f('rc')}  n {f('n')}  d log hop/d log n {sl:+.2f}")

print("\nhop_meas/d_S against crowding x = (K-1) n / (2 M) (share of sensors claimed by other airborne UAVs, mean-field)")
allp = []
for (lay, E, nm), v in out.items():
    if nm != 'nhop': continue
for k, b in sorted(rows.items()):
    pass
