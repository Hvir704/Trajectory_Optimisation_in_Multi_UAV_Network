"""Scratch: crowding-corrected hop in the surrogate (v3.4 candidate), evaluated on ALL seen data. Zero fitted constants:
hop_eff(K) = d_S / sqrt(1 - x), x = min(0.9, (K-1) n(K) / (2 |S|)), n(K) the a-priori stop count at hop_eff (fixed point)."""
import csv, collections, numpy as np, sys, os, pickle, math
sys.path.insert(0, ".")
import theory_v3
from theory_v3 import field, old_rule, tour_len
from exante_rc import exante
from run_grid import k_range
if os.path.exists("tour_cache_v3.pkl"): theory_v3.TOUR_CACHE.update(pickle.load(open("tour_cache_v3.pkl", "rb")))

def rule(p, F, rc, Pb, key, crowd, horizon=None, gray="exclude", Kmax=200):
    U = (1 - p.rho) * p.Emax; tc = 2 * rc / p.v; r = np.linalg.norm(F.pos - p.home, axis=1); w = F.wi_base
    Tb = p.T_burnin; T = (p.T_horizon if horizon is None else horizon) - Tb
    Phi = lambda K: K * (U - K * p.Pf * tc) / (U + K * tc * (Pb - p.Pf)); best = None
    for K in range(1, Kmax + 1):
        reach = p.v * (U / K - p.Ph * p.B_bits / p.R) / (2 * p.Pf); keep = r <= reach
        if keep.sum() < 2: break
        j = int((~keep).sum())
        if gray == "exclude" and j > 0 and np.any(r[~keep] <= p.v * U / (2 * p.Pf * K)): continue
        ck = key + (j,)
        if ck not in theory_v3.TOUR_CACHE: theory_v3.TOUR_CACHE[ck] = tour_len(F.pos[keep], p.home)
        S = keep.sum(); dS = theory_v3.TOUR_CACHE[ck] / S; H = S * p.lam_bits / p.R
        d = dS
        if crowd:
            h = min(max(H / max(Phi(K), 1e-9), 0), 0.95)
            for _ in range(20):
                tau_s = (d / p.v) / (1 - h); n = max((U / K - p.Pf * tc) / (Pb * tau_s), 1.0)
                x = min(0.9, (K - 1) * n / (2 * S)); d = dS / math.sqrt(1 - x)
        g = p.v * (Phi(K) - H) / d
        if g <= 0:
            if K > 1 and Phi(K) < Phi(K - 1): break
            continue
        pi = r[keep] / r[keep].mean()
        Jk = np.sqrt(w[keep] * pi).sum() * np.sqrt(w[keep] / pi).sum() / (2 * g) + (Tb + T / 2) * w[~keep].sum()
        if best is None or Jk < best[0]: best = (Jk, K, j)
    return best[1]

def load(path, th=43200.0):
    D = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if float(r.get('Th', 43200)) != th or float(r.get('L', 12600)) != 12600: continue
        if r.get('coord', 'exclude') != 'exclude' or r.get('replan', 'launch') != 'launch': continue
        D[(r.get('planner', 'sa'), r['layout'], int(float(r['M'])), float(r['Emax']), int(float(r['seed'])))][int(float(r['K']))] = (float(r['J']), r.get('K_ref'))
    return D
A = os.environ.get('V3_ASSEMBLED', 'assembled') + '/'
R = {}
for f in ['grid_final.csv', 'rr_tour_win.csv', 'eval_all_v2.csv']: R.update(load(A + f))
H1 = load(os.environ.get('V3_RESULTS', 'results') + '/heldout_v3.csv')
H2p = os.environ.get('V3_RESULTS', 'results') + '/heldout2_v3.csv'
H2 = load(H2p); H2_24 = load(H2p, 86400.0)
def nr(x): x = np.sort(x); return x[max(0, math.ceil(.9 * len(x)) - 1)]
def evaluate(D, label, kref_fn, planners=('sa', 'cyc', 'rr_tour'), extra=None):
    preds = {}
    for k in sorted({k[1:] for k in D if k[0] == 'sa'}):
        lay, M, E, s = k
        if (lay, M, E) == ('paper', 100, 6e6) or M == 400: continue
        p, F = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F.pos - p.home, axis=1)
        rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, kref_fn(k)); key = (lay, M, 12600.0, s)
        preds[k] = dict(old=old_rule(p, r, rc, Pb), v33=rule(p, F, rc, Pb, key, False), v34=rule(p, F, rc, Pb, key, True))
    for pl in planners:
        for fam in ('core', 'paper', 'ring'):
            S = collections.defaultdict(list)
            for k, pr in preds.items():
                if k[0] != fam or (pl,) + k not in D: continue
                c = {K: v[0] for K, v in D[(pl,) + k].items()}; Ks = sorted(c); am = min(c, key=c.get)
                for nm, K in pr.items():
                    S[nm + 'w'].append(abs(am - K) <= 1); S[nm + 'g'].append(c[min(Ks, key=lambda q: abs(q - K))] / c[am] - 1)
            if S['oldw']:
                print(f"{label:6s} {pl:7s} {fam:5s} n={len(S['oldw']):3d} " + " | ".join(
                    f"{nm} w1 {np.mean(S[nm+'w']):.2f} p90 {100*nr(S[nm+'g']):5.1f} max {100*max(S[nm+'g']):5.1f}" for nm in ('old', 'v33', 'v34')))
    return preds
def kr(k):
    c = R[('sa',) + k]; v = next(iter(c.values()))[1]; return int(float(v)) if v not in (None, '') else min(c)
evaluate(R, 'retro', kr)
evaluate(H1, 'held1', lambda k: min(k_range(k[0], k[1], k[2], k[3], 12600.0)))
evaluate(H2, 'held2', lambda k: min(k_range(k[0], k[1], k[2], k[3], 12600.0)))
# horizon arm of test 2 under v3.4
hz = []
for k, c in H2_24.items():
    if k[0] != 'sa': continue
    lay, M, E, s = k[1:]
    p, F = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, min(k_range(lay, M, E, s, 12600.0))); key = (lay, M, 12600.0, s)
    J = {K: v[0] for K, v in c.items()}; am = min(J, key=J.get)
    a33 = rule(p, F, rc, Pb, key, False, horizon=86400.0); a34 = rule(p, F, rc, Pb, key, True, horizon=86400.0)
    hz.append((abs(am - a33) <= 1, abs(am - a34) <= 1))
print("held2 horizon arm (45 instances, 24 h): within one of the 24 h optimum  v3.3", sum(a for a, b in hz), " v3.4", sum(b for a, b in hz), "of", len(hz))
pickle.dump(theory_v3.TOUR_CACHE, open("tour_cache_v3.pkl", "wb"))
