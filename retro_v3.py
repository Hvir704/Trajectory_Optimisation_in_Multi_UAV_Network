"""retro_v3.py -- the v3 theory on the existing 272 instances (retrospective; the rule was designed on these).

Ex-ante commute inputs (exante_rc.exante at the reference K = first K of the original sweep, nominal rates), served-set surrogate rule
(theory_v3), commute bracket. Reports exact / within-one / regret against the annealer's swept optimum, for the
annealer and (with --planners) for the other planners' own optima on the 216 held-out deployments.

usage: python retro_v3.py results_v2/report/assembled [--planners]
"""
import sys, os, csv, collections, numpy as np
import pickle, theory_v3
from theory_v3 import field, old_rule, surrogate_rule, bracket, pilot_scale
from exante_rc import exante


def kref(byK):
    """Reference fleet size of an instance: the first K of its ORIGINAL sweep (column K_ref, written by reproduce_v2.py
    v2.2.1). Fills added for other planners (union grid) must not move the ex-ante or pilot fleet size. Older assembled
    files have no K_ref column; the smallest swept K is then the original first K."""
    for v in byK.values():
        x = v.get("K_ref") if isinstance(v, dict) else None
        if x not in (None, ""):
            k = int(float(x)); return k if k in byK else min(byK)
        break
    return min(byK)

A = sys.argv[1]; PLANNERS = "--planners" in sys.argv
D = collections.defaultdict(dict)
for r in csv.DictReader(open(os.path.join(A, "grid_final.csv"))):
    if r["coord"] == "exclude" and r["replan"] == "launch" and float(r["Th"]) == 43200 and r.get("planner", "sa") == "sa":
        D[(r["layout"], int(float(r["M"])), float(r["Emax"]), float(r["L"]), int(r["seed"]))][int(float(r["K"]))] = r

CACHE = os.environ.get("TOUR_CACHE", "tour_cache_v3.pkl")
if os.path.exists(CACHE): theory_v3.TOUR_CACHE.update(pickle.load(open(CACHE, "rb")))
PRED = {}
for k, byK in sorted(D.items()):
    lay, M, E, L, s = k; p, F = field(lay, M, E, L, s)
    r = np.linalg.norm(F.pos - p.home, axis=1)
    K0 = kref(byK); rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
    PRED[k] = dict(old=old_rule(p, r, rc, Pb), new=surrogate_rule(p, F, rc, Pb, key=(lay, M, L, s))[0], br=bracket(p, r, rc, Pb),
                   p=p, F=F, rc=rc, Pb=Pb, K0=K0, key=(lay, M, L, s))
pickle.dump(theory_v3.TOUR_CACHE, open(CACHE, "wb"))


def evaluate(J, label, own_pilot=None):
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, c in J.items():
        if k not in PRED: continue
        P = PRED[k]; Ks = sorted(c); am = min(c, key=c.get)
        rg = lambda K: c[min(Ks, key=lambda x: abs(x - K))] / c[am] - 1
        preds = dict(old=P["old"], new=P["new"])
        if own_pilot is not None and k in own_pilot:
            preds["pilot"] = own_pilot[k]
        for f in (k[0], "all"):
            for nm, K in preds.items():
                fam[f][nm + "_ex"].append(am == K); fam[f][nm + "_w1"].append(abs(am - K) <= 1); fam[f][nm + "_rg"].append(rg(K))
            if P["br"] is not None: fam[f]["br"].append(P["br"][0] <= am <= P["br"][1])
    for f in ("all", "core", "paper", "ring"):
        R = fam[f]
        if not R["old_ex"]: continue
        line = f"{label:15s} {f:5s} n={len(R['old_ex']):3d}"
        for nm in ("old", "new", "pilot"):
            if R[nm + "_ex"]:
                g = np.array(R[nm + "_rg"])
                line += (f" | {nm}: exact {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} regret med {np.median(g):.1%}"
                         f" p90 {np.quantile(g, .9):.1%} max {g.max():.1%}")
        if R["br"]: line += f" | bracket covers {np.mean(R['br']):.2f} of {len(R['br'])}"
        print(line)


J = {k: {K: float(r["J"]) for K, r in byK.items()} for k, byK in D.items()}
pil = {}
for k, byK in D.items():
    sf = [K for K in sorted(byK) if K >= kref(byK) and int(float(byK[K]["n_never"])) == 0]   # pilot from the original sweep
    if sf:
        P = PRED[k]; sc = pilot_scale(P["p"], P["F"], P["rc"], P["Pb"], sf[0], float(byK[sf[0]]["J_age"]), key=P["key"])
        pil[k] = surrogate_rule(P["p"], P["F"], P["rc"], P["Pb"], scale=sc, key=P["key"])[0]
evaluate(J, "sa", pil)

if PLANNERS:
    for f, pl in [("rr_tour_win.csv", "rr_tour"), ("cluster_win.csv", "cluster_patrol"), ("eval_all_v2.csv", "cyc"),
                  ("gzz_eval.csv", "gzz_rev"), ("asghar_eval.csv", "asghar_rev"), ("liu_all.csv", "liu_mpga")]:
        R = collections.defaultdict(dict)
        for r in csv.DictReader(open(os.path.join(A, f))):
            if r.get("planner") == pl and float(r.get("Th", 43200)) == 43200:
                R[(r["layout"], int(float(r["M"])), float(r["Emax"]), float(r.get("L", 12600)), int(r["seed"]))][int(float(r["K"]))] = r
        Jp = {k: {K: float(x["J"]) for K, x in byK.items()} for k, byK in R.items()}
        own = {}
        for k, byK in R.items():
            if k not in PRED: continue
            sf = [K for K in sorted(byK) if K >= kref(byK) and int(float(byK[K]["n_never"])) == 0]
            if sf:
                P = PRED[k]; sc = pilot_scale(P["p"], P["F"], P["rc"], P["Pb"], sf[0], float(byK[sf[0]]["J_age"]), key=P["key"])
                own[k] = surrogate_rule(P["p"], P["F"], P["rc"], P["Pb"], scale=sc, key=P["key"])[0]
        evaluate(Jp, pl, own)
