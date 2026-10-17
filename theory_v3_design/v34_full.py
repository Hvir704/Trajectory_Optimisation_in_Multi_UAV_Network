import csv, collections, numpy as np, sys, os, pickle, math
sys.path.insert(0, ".")
import theory_v3, theory_v34
from theory_v34 import field, old_rule, surrogate_rule as rule34, SHARED
from theory_v3 import surrogate_rule as rule3
from exante_rc import exante
from run_grid import k_range
if os.path.exists("tour_cache_v3.pkl"): theory_v3.TOUR_CACHE.update(pickle.load(open("tour_cache_v3.pkl", "rb")))
def load(path, th=43200.0):
    D = collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if float(r.get('Th', 43200)) != th or float(r.get('L', 12600)) != 12600: continue
        if r.get('coord', 'exclude') != 'exclude' or r.get('replan', 'launch') != 'launch': continue
        D[(r.get('planner', 'sa'), r['layout'], int(float(r['M'])), float(r['Emax']), int(float(r['seed'])))][int(float(r['K']))] = (float(r['J']), r.get('K_ref'))
    return D
A = os.environ.get('V3_ASSEMBLED', 'assembled') + '/'; R = {}
for f in ['grid_final.csv','rr_tour_win.csv','eval_all_v2.csv','cluster_win.csv','gzz_eval.csv','asghar_eval.csv','liu_all.csv']: R.update(load(A + f))
H1 = load(os.environ.get('V3_RESULTS', 'results') + '/heldout_v3.csv')
H2 = load(os.environ.get('V3_RESULTS', 'results') + '/heldout2_v3.csv')
def nr(x): x = np.sort(x); return x[max(0, math.ceil(.9 * len(x)) - 1)]
cache = {}
def preds(k, K0):
    if (k, K0) in cache: return cache[(k, K0)]
    lay, M, E, s = k; p, F = field(lay, M, E, 12600.0, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0); key = (lay, M, 12600.0, s)
    out = dict(old=old_rule(p, r, rc, Pb), v33=rule3(p, F, rc, Pb, key=key, gray="exclude")[0],
               v34s=rule34(p, F, rc, Pb, key=key, shared=True)[0])
    cache[(k, K0)] = out; return out
def evaluate(D, label, kref_fn, planners):
    tot = collections.defaultdict(lambda: collections.defaultdict(list))
    for pl in planners:
        for k in sorted({k[1:] for k in D if k[0] == pl}):
            if ('sa',) + k not in D and label == 'retro': continue
            pr = preds(k, kref_fn(k)); v34 = pr['v34s'] if pl in SHARED else pr['v33']
            c = {K: v[0] for K, v in D[(pl,) + k].items()}; Ks = sorted(c); am = min(c, key=c.get)
            for fam in (k[0], 'all'):
                for nm, K in (('old', pr['old']), ('v33', pr['v33']), ('v34', v34)):
                    tot[(pl, fam)][nm + 'w'].append(abs(am - K) <= 1); tot[(pl, fam)][nm + 'g'].append(c[min(Ks, key=lambda q: abs(q - K))] / c[am] - 1)
    for (pl, fam), S in sorted(tot.items()):
        print(f"{label:6s} {pl:14s} {fam:5s} n={len(S['oldw']):3d} " + " | ".join(
            f"{nm} w1 {np.mean(S[nm+'w']):.2f} p90 {100*nr(S[nm+'g']):5.1f} max {100*max(S[nm+'g']):5.1f}" for nm in ('old', 'v33', 'v34')))
def kr(k):
    c = R[('sa',) + k]; v = next(iter(c.values()))[1]; return int(float(v)) if v not in (None, '') else min(c)
kh = lambda k: min(k_range(k[0], k[1], k[2], k[3], 12600.0))
# NOTE (added for the verification bundle): k = (planner, layout, M, Emax, seed), so `k[3] != 400` is a no-op;
# the second condition drops the paper M=100 6 MJ cell (12 instances per planner). THEORY_V3 section 11 states this.
evaluate({k: v for k, v in R.items() if k[3] != 400 and not (k[1] == 'paper' and k[2] == 100 and k[3] == 6e6)}, 'retro', kr,
         ['sa', 'rr_tour', 'cyc', 'cluster_patrol', 'liu_mpga', 'gzz_rev', 'asghar_rev'])
evaluate(H1, 'held1', kh, ['sa', 'rr_tour', 'cyc'])
evaluate(H2, 'held2', kh, ['sa', 'rr_tour', 'cyc', 'cluster_patrol'])
pickle.dump(theory_v3.TOUR_CACHE, open("tour_cache_v3.pkl", "wb"))
