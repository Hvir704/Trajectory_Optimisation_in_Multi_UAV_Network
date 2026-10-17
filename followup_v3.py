"""followup_v3.py -- analyses and one experiment added AFTER the registered held-out run of heldout_v3.py.

The registered protocol, predictions and evaluation are untouched (heldout_v3.py, heldout_v3_predictions.csv,
sha256 4c77e53e...). Everything here is labelled as post-registration, with one exception: the horizon experiment
below has its own predictions file, fixed and hashed before its own run.

posthoc     (on the finished 12 h held-out results; suggested by the independent review, decided after the run)
  - edge check: optima at the smallest swept K (> 1) or at the largest; the registered sweep only checked the top.
  - paired comparison on the instances where v3 != the old rule (22 of 160): better / worse, two-sided sign test,
    bootstrap 95% interval of the mean regret change, per family.
  - ring bracket: width per instance and where each planner's optimum falls.

horizon     a new experiment, registered 2026-10-12 before it is run.
  question  v3 is the rule for a mission of length T; its stranding branch should switch off as T grows. Does the
            measured optimum move the way v3 says when the mission is 24 h instead of 12 h?
  instances core M=100 and M=200 at 1.5 MJ, seeds 13-20 (16 instances), SA, T = 24 h (3 h burn-in).
  predictions heldout_v3_horizon_predictions.csv, sha256 (LF-normalised)
            eb955eb077a865301b5678cbd4697420a7f8ab01855d65cc52a3e724cb26eda7. v3(24 h) differs from v3(12 h) on 4 of
            the 16 instances (core M=100 s=17, M=200 s=14, 18, 20: from 14 back to the plain rule).
  sweep     contiguous K_lo..K_hi, then extended two-sided (up: the paper's rule; down: while the argmin is the
            smallest K swept and > 1).
  H5        on the 4 instances that differ, the measured 24 h optimum is closer to v3(24 h) than to v3(12 h) in at
            least 3. Reported on all 16, whatever the outcome: regret of old, v3(12 h) and v3(24 h) at 24 h.

usage:
  python followup_v3.py posthoc results_v2/heldout_v3.csv
  python followup_v3.py run-horizon results_v2/heldout_v3_h24.csv --sha eb955eb0... [--procs 22]
  python followup_v3.py evaluate-horizon results_v2/heldout_v3_h24.csv
"""
import sys, os, csv, hashlib, argparse, itertools, collections, time
from math import comb
import numpy as np
from multiprocessing import Pool
from heldout_v3 import load_pred, read_out, L
from run_grid import one, FIELDS

HPRED = "heldout_v3_horizon_predictions.csv"; TH = 86400.0


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def argmin(c): return min(c, key=lambda K: float(c[K]["J"]))


def regret(c):
    Jv = {K: float(x["J"]) for K, x in c.items()}; Ks = sorted(Jv); am = min(Jv, key=Jv.get)
    return lambda K: Jv[min(Ks, key=lambda x: abs(x - K))] / Jv[am] - 1


def needs_more(c):
    Ks = sorted(c); Jv = {K: float(c[K]["J"]) for K in Ks}; am = min(Jv, key=Jv.get)
    if am >= Ks[-1] - 1 or not all(Jv[K] > 1.5 * Jv[am] for K in Ks[-3:]): return "up"
    if am == Ks[0] and Ks[0] > 1: return "down"
    return None


def posthoc(path):
    P = load_pred(); D = read_out(path)
    print("post-registration analyses of", path, "(predictions sha256", sha_lf("heldout_v3_predictions.csv") + ")")
    print("\n1) sweep edges: optimum at the smallest swept K (> 1) / at the largest")
    for pl in sorted({k[0] for k in D}):
        lo = [k[1:] for k in D if k[0] == pl and argmin(D[k]) == min(D[k]) and min(D[k]) > 1]
        hi = [k[1:] for k in D if k[0] == pl and argmin(D[k]) == max(D[k])]
        print(f"   {pl:8s} lower edge {len(lo)} " + ", ".join(f"{a} M={b} {c/1e6:.1f} MJ s={d}" for a, b, c, d in lo) + f" | upper edge {len(hi)}")
    print("\n2) paired comparison where v3 != old (regret change = v3 - old; negative = v3 better)")
    for pl in sorted({k[0] for k in D}):
        per = []
        for k, pr in P.items():
            if pr["old"] == pr["new"] or (pl,) + k not in D: continue
            g = regret(D[(pl,) + k]); per.append((k[0], g(int(pr["new"])) - g(int(pr["old"]))))
        for fam in (None, "core", "paper"):
            d = np.array([v for f, v in per if fam is None or f == fam])
            if not len(d): continue
            w, l = int((d < -1e-12).sum()), int((d > 1e-12).sum()); n = w + l
            p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n) if n else 1.0
            bs = [d[np.random.default_rng(20261012 + b).integers(0, len(d), len(d))].mean() for b in range(4000)]
            print(f"   {pl:8s} {fam or 'all':5s} n={len(d):2d}: v3 better {w:2d}, worse {l:2d}; sign test p={p:.3f}; mean change "
                  f"{d.mean():+.1%} [{np.quantile(bs, .025):+.1%}, {np.quantile(bs, .975):+.1%}]")
    print("\n3) ring bracket [round Kc, floor Kr]")
    wd = collections.Counter(int(r["br_hi"]) - int(r["br_lo"]) + 1 for r in P.values() if r["br_lo"])
    print(f"   width in UAVs (instances): {dict(sorted(wd.items()))}")
    for pl in sorted({k[0] for k in D}):
        pos = collections.Counter()
        for k, pr in P.items():
            if not pr["br_lo"] or (pl,) + k not in D: continue
            am = argmin(D[(pl,) + k]); lo, hi = int(pr["br_lo"]), int(pr["br_hi"])
            pos["below" if am < lo else "above" if am > hi else ("at Kc" if am == lo else "inside, above Kc")] += 1
        print(f"   {pl:8s} optimum: {dict(pos)}")


def run_horizon(path, sha, procs):
    if sha_lf(HPRED) != sha: sys.exit(f"{HPRED} sha256 {sha_lf(HPRED)} != registered {sha}")
    H = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(HPRED))}
    new = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f, Pool(procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader(); f.flush()
        for rnd in itertools.count():
            done = read_out(path); jobs = []
            for (lay, M, E, s), pr in H.items():
                have = done.get(("sa", lay, M, E, s), {})
                if rnd == 0 or not have: Ks = range(int(pr["K_lo"]), int(pr["K_hi"]) + 1)
                else:
                    nm = needs_more(have)
                    Ks = (range(max(have) + 1, max(have) + 3) if nm == "up" else
                          range(max(1, min(have) - 2), min(have)) if nm == "down" else [])
                jobs += [(lay, M, E, K, s, "exclude", "launch", 0, 0.0, "prior", 0.0, "sa", L, TH, 1200) for K in Ks if K not in have]
            if not jobs:
                print("ALL DONE: every sweep interior and extended", flush=True); return
            jobs.sort(key=lambda j: (-j[1], -j[3])); print(f"round {rnd}: {len(jobs)} runs", flush=True); t0 = time.time()
            for i, row in enumerate(pool.imap_unordered(one, jobs)):
                w.writerow(row); f.flush()
                if i % 10 == 0: print(f"[{i+1}/{len(jobs)}] core M={row['M']} K={row['K']} s={row['seed']} ({time.time()-t0:.0f}s)", flush=True)


def evaluate_horizon(path):
    H = {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(HPRED))}
    D = read_out(path); print(f"horizon experiment (T = 24 h), predictions sha256 {sha_lf(HPRED)}")
    R = []; closer = []
    for k, pr in sorted(H.items()):
        c = D.get(("sa",) + k)
        if not c: continue
        g = regret(c); am = argmin(c); o, a, b = int(pr["old"]), int(pr["new12"]), int(pr["new24"])
        R.append((g(o), g(a), g(b)))
        flag = "  <- v3(12h) != v3(24h)" if a != b else ""
        print(f"  {k[0]} M={k[1]} s={k[3]}: 24 h optimum {am}{' (sweep needs extension)' if needs_more(c) else ''} | old {o} ({g(o):.1%})"
              f"  v3(12h) {a} ({g(a):.1%})  v3(24h) {b} ({g(b):.1%}){flag}")
        if a != b: closer.append(abs(am - b) < abs(am - a))
    if R:
        R = np.array(R)
        for i, lab in enumerate(("old", "v3(12h)", "v3(24h)")):
            print(f"  {lab:8s} regret median {np.median(R[:, i]):.1%}  max {R[:, i].max():.1%}")
        print(f"  H5: {'HOLDS' if sum(closer) >= 3 else 'FAILS'} ({sum(closer)} of {len(closer)} closer to v3(24h))")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["posthoc", "run-horizon", "evaluate-horizon"])
    ap.add_argument("out"); ap.add_argument("--sha"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    if a.mode == "posthoc": posthoc(a.out)
    elif a.mode == "run-horizon": run_horizon(a.out, a.sha, a.procs)
    else: evaluate_horizon(a.out)
