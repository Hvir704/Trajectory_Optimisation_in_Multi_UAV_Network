import os
import csv, collections, numpy as np, sys, pickle, math
sys.path.insert(0, ".")
import theory_v3
from theory_v34 import field, surrogate_rule
from exante_rc import exante
if os.path.exists("tour_cache_v3.pkl"): theory_v3.TOUR_CACHE.update(pickle.load(open("tour_cache_v3.pkl", "rb")))
rows = collections.defaultdict(dict)
for r in csv.DictReader(open(os.environ.get('V3_ASSEMBLED', 'assembled') + '/grid_final.csv')):
    if r.get('planner','sa')!='sa' or r['coord']!='exclude' or r['replan']!='launch' or float(r['Th'])!=43200 or float(r['L'])!=12600: continue
    rows[(r['layout'],int(r['M']),float(r['Emax']),int(r['seed']))][int(r['K'])]=r
res = collections.defaultdict(lambda: collections.defaultdict(list))
for k, b in sorted(rows.items()):
    lay, M, E, s = k
    if M == 400 or E == 6e6: continue
    p, F = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    K0 = int(float(next(iter(b.values()))['K_ref'])); rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
    key = (lay, M, 12600.0, s)
    _, _, cu = surrogate_rule(p, F, rc, Pb, key=key, shared=True, return_curve=True)
    _, _, cu0 = surrogate_rule(p, F, rc, Pb, key=key, shared=False, return_curve=True)
    pairs = []
    for K, x in b.items():
        if int(float(x['n_never'])) != 0 or float(x['mean_n']) <= 0 or K not in cu: continue
        n = float(x['mean_n']); Ts = float(x['T_s_over_t_c']) * p.t_c; tcm = 2 * float(x['r_c']) / p.v
        h = (float(x['P_bar']) - p.Pf) / (p.Ph - p.Pf); tau = max(Ts - tcm, 1e-9) * (1 - h) / n
        pairs.append((K, tau, cu[K][3] / p.v, cu0[K][3] / p.v))
    if len(pairs) < 3: continue
    P = np.array(sorted(pairs)); rk = lambda a: np.argsort(np.argsort(a))
    res[(lay, E)]['ratio_crowd'] += list(P[:, 2] / P[:, 1]); res[(lay, E)]['ratio_uni'] += list(P[:, 3] / P[:, 1])
    res[(lay, E)]['rho_meas'].append(np.corrcoef(rk(P[:, 0]), rk(P[:, 1]))[0, 1])
    res[(lay, E)]['rho_model'].append(np.corrcoef(rk(P[:, 0]), rk(P[:, 2]))[0, 1])
print("hop: model / measured per run (median [10th, 90th]); within-instance Spearman of hop with K (median), measured vs model")
for (lay, E), d in sorted(res.items()):
    f = lambda v: f"{np.median(v):.2f} [{np.quantile(v,.1):.2f},{np.quantile(v,.9):.2f}]"
    print(f"{lay:5s} {E/1e6:.1f} MJ  uniform {f(d['ratio_uni'])}  crowd {f(d['ratio_crowd'])}  trend: measured {np.median(d['rho_meas']):+.2f}  crowd model {np.median(d['rho_model']):+.2f}")
