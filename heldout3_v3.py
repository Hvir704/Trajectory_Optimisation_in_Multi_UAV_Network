"""heldout3_v3.py -- third registered held-out test: v3.4 (crowding-corrected hop) against v3.3 (registered 2026-10-13).

v3.4 (theory_v34.py) was designed on seeds 1-40 after the second test. For planners whose UAVs share the field it
replaces the uniform a-priori hop d_S by d_S / sqrt(1 - x), x = (K - 1) n(K) / (2 |S|), the share of served sensors
held by the other airborne UAVs. For fixed-territory planners it equals v3.3 by construction, so only shared-field
planners are tested: the annealer (sa) and the tour patrol (rr_tour).

Fresh fields: seeds 41 and 43-49. Seeds 1-40 were used to design or test v3-v3.4. Seed 42 occurs in an old file;
no other seed >= 41 occurs in any CSV of the repository or of results_v2 (checked 2026-10-12/13).

RULES (a priori; ex-ante commute inputs at K0 = min k_range):
  old   min(floor Kr, round Kc)
  v33   theory_v3.surrogate_rule(gray="exclude")
  v34   theory_v34.surrogate_rule(shared=True)              <- PRIMARY (planners sa and rr_tour)

DESIGN: 20 primary cells x 8 seeds = 160 instances, T_mis = 12 h, planners sa and rr_tour. Sweeps from
K_lo = K0 to K_hi = max(max k_range, every predicted K + 2, floor K_flight(S_1) + 2), extended in rounds at BOTH edges
until every optimum is interior (or at K = 1) and the three largest K exceed 1.5 x the minimum.

REGISTERED HYPOTHESES, each tested separately for sa and for rr_tour (8 verdicts). Regret = J(predicted K) / J(swept
optimum) - 1, with the nearest swept K if the predicted K was not swept; p90 = nearest rank.
  A  all 160:   v34 within-one rate > v33 within-one rate (strictly).
  B  ring 48:   v34 within-one rate >= 0.85 and > v33 within-one rate.
  C  all 160:   v34 p90 regret < v33 p90 regret.
  D  all 160:   where v34 != v33, v34 has lower regret more often than higher; one-sided exact sign test p < 0.05
                (equal regrets dropped).
REPORTED WITHOUT A VERDICT: old, v33, v34 by family; exact agreement; mean, median, max; linear p90; sweeps censored.

REGISTRATION: `predict` writes heldout3_predictions.csv and prints a MANIFEST sha256 over the LF-normalised bytes of
heldout3_predictions.csv, heldout3_v3.py, theory_v34.py and theory_v3.py. Publish it (and nothing else) before `run`;
`run` refuses to start if any of the four files differs.

usage:
  python heldout3_v3.py predict
  python heldout3_v3.py run results_v2/heldout3_v3.csv --sha <manifest sha256> [--procs 22]
  python heldout3_v3.py evaluate results_v2/heldout3_v3.csv
"""
import sys, os, csv, math, hashlib, argparse, itertools, collections, time
import numpy as np
from multiprocessing import Pool
from theory_v3 import field, old_rule, surrogate_rule
import theory_v34
from exante_rc import exante
from run_grid import one, k_range, FIELDS, SIM_VERSION

CELLS = ([("paper", M, 1.5e6) for M in (50, 100, 200, 400)] + [("paper", M, 3e6) for M in (50, 100, 200)]
         + [("paper", 100, 6e6)] + [(lay, M, E) for lay in ("ring", "core") for E in (1.5e6, 3e6) for M in (50, 100, 200)])
SEEDS = [41, 43, 44, 45, 46, 47, 48, 49]
PLANNERS = ["sa", "rr_tour"]
L = 12600.0; TH = 43200.0; PRED = "heldout3_predictions.csv"
MANIFEST = [PRED, "heldout3_v3.py", "theory_v34.py", "theory_v3.py"]


def manifest_sha():
    h = hashlib.sha256()
    for f in MANIFEST:
        h.update(hashlib.sha256(open(f, "rb").read().replace(b"\r\n", b"\n")).hexdigest().encode() + b"  " + f.encode() + b"\n")
    return h.hexdigest()


def instance(lay, M, E, s):
    p, F = field(lay, M, E, L, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    kr = k_range(lay, M, E, s, L); K0 = min(kr)
    rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
    key = (lay, M, L, s)
    old = old_rule(p, r, rc, Pb)
    v33, j33 = surrogate_rule(p, F, rc, Pb, key=key, gray="exclude")
    v34, j34 = theory_v34.surrogate_rule(p, F, rc, Pb, key=key, shared=True)
    U = (1 - p.rho) * E; kfl1 = math.floor(p.v * U / (2 * p.Pf * np.sort(r)[-2]))
    hi = max(max(kr), old + 2, v33 + 2, v34 + 2, kfl1 + 2)
    f = lambda x, n: round(float(x), n)
    return dict(layout=lay, M=int(M), Emax=float(E), L=float(L), seed=int(s), K0=int(K0), rc_exante=f(rc, 3),
                Pb_exante=f(Pb, 3), old=int(old), v33=int(v33), v33_stranded=int(j33), v34=int(v34),
                v34_stranded=int(j34), K_lo=int(K0), K_hi=int(hi))


def predict():
    rows = []
    for (lay, M, E), s in itertools.product(CELLS, SEEDS):
        d = instance(lay, M, E, s); rows.append(d)
        print(f"{lay} M={M} E={E/1e6} s={s}: old {d['old']} v33 {d['v33']} v34 {d['v34']} sweep {d['K_lo']}-{d['K_hi']}", flush=True)
    with open(PRED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} instances; v34 != v33 in {sum(r['v34'] != r['v33'] for r in rows)}")
    print("MANIFEST sha256", manifest_sha())


def load_pred():
    return {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(PRED))}


def read_out(path):
    J = collections.defaultdict(dict)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        for r in csv.DictReader(open(path)):
            if r.get("sim_version") != SIM_VERSION: sys.exit(f"{path}: row with sim_version {r.get('sim_version')}")
            J[(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]))][int(r["K"])] = r
    return J


def extension(byK):
    Ks = sorted(byK); Jv = {K: float(byK[K]["J"]) for K in Ks}; am = min(Jv, key=Jv.get); add = set()
    if am >= Ks[-1] - 1 or not all(Jv[K] > 1.5 * Jv[am] for K in Ks[-3:]): add |= {Ks[-1] + 1, Ks[-1] + 2}
    if am == Ks[0] and Ks[0] > 1: add |= {K for K in (Ks[0] - 1, Ks[0] - 2) if K >= 1}
    return sorted(add - set(Ks))


def run(a):
    sha = manifest_sha()
    if sha != a.sha: sys.exit(f"manifest sha256 {sha} != registered {a.sha}: a registered file changed")
    P = load_pred(); new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader(); f.flush()
        for rnd in itertools.count():
            done = read_out(a.out); jobs = []
            for (lay, M, E, s), pr in P.items():
                for pl in PLANNERS:
                    have = done.get((pl, lay, M, E, s), {})
                    Ks = range(int(pr["K_lo"]), int(pr["K_hi"]) + 1) if (rnd == 0 or not have) else extension(have)
                    jobs += [(lay, M, E, K, s, "exclude", "launch", 0, 0.0, "prior", 0.0, pl, L, TH, 1200) for K in Ks if K not in have]
            if not jobs:
                print("ALL DONE: every sweep interior (or at K = 1) and extended", flush=True); return
            jobs.sort(key=lambda j: (-j[1], -j[3]))
            print(f"round {rnd}: {len(jobs)} runs", flush=True); t0 = time.time()
            for i, row in enumerate(pool.imap_unordered(one, jobs)):
                w.writerow(row); f.flush()
                if i % 20 == 0:
                    print(f"[{i+1}/{len(jobs)}] {row['planner']} {row['layout']} M={row['M']} K={row['K']} s={row['seed']} "
                          f"({time.time()-t0:.0f}s)", flush=True)


def p90nr(x):
    if len(x) == 0: return float("nan")
    x = np.sort(np.asarray(x)); return float(x[max(0, math.ceil(0.9 * len(x)) - 1)])


def binom_tail(k, n):
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def evaluate(a):
    P = load_pred(); D = read_out(a.out); RULES = ("old", "v33", "v34")
    print(f"manifest sha256 {manifest_sha()}")
    S = collections.defaultdict(lambda: collections.defaultdict(list)); cens = collections.Counter(); paired = collections.defaultdict(list)
    for pl in PLANNERS:
        for k, pr in P.items():
            byK = D.get((pl,) + k)
            if not byK: continue
            if extension(byK): cens[pl] += 1
            Jv = {K: float(x["J"]) for K, x in byK.items()}; Ks = sorted(Jv); am = min(Jv, key=Jv.get)
            rg = lambda K: Jv[min(Ks, key=lambda x: abs(x - K))] / Jv[am] - 1
            for fam in (k[0], "all"):
                for nm in RULES:
                    K = int(pr[nm]); S[(pl, fam)][nm + "_w1"].append(abs(am - K) <= 1)
                    S[(pl, fam)][nm + "_ex"].append(am == K); S[(pl, fam)][nm + "_rg"].append(rg(K))
            if int(pr["v34"]) != int(pr["v33"]): paired[pl].append(rg(int(pr["v34"])) - rg(int(pr["v33"])))
    for pl in PLANNERS:
        print(f"\n== {pl}  (sweeps still needing extension: {cens[pl]})")
        for fam in ("all", "core", "paper", "ring"):
            R = S[(pl, fam)]
            if not R["old_rg"]: continue
            line = f"  {fam:5s} n={len(R['old_rg']):3d}"
            for nm in RULES:
                g = np.array(R[nm + "_rg"])
                line += (f" | {nm}: ex {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} mean {g.mean():.1%} "
                         f"med {np.median(g):.1%} p90 {p90nr(g):.1%} (lin {np.quantile(g, .9):.1%}) max {g.max():.1%}")
            print(line)
    rate = lambda pl, fam, nm: float(np.mean(S[(pl, fam)][nm + "_w1"])) if S[(pl, fam)][nm + "_w1"] else float("nan")
    p90 = lambda pl, fam, nm: p90nr(S[(pl, fam)][nm + "_rg"])
    print("\nREGISTERED HYPOTHESES (primary rule v34; each for sa and rr_tour)")
    v = {}
    for pl in PLANNERS:
        a34, a33 = rate(pl, "all", "v34"), rate(pl, "all", "v33"); v[f"A {pl}"] = a34 > a33
        print(f"  A {pl}: within-one all {a34:.3f} vs v33 {a33:.3f}: {'HOLDS' if v[f'A {pl}'] else 'FAILS'}")
        r34, r33 = rate(pl, "ring", "v34"), rate(pl, "ring", "v33"); v[f"B {pl}"] = r34 >= 0.85 and r34 > r33
        print(f"  B {pl}: within-one ring {r34:.3f} (>= 0.85) vs v33 {r33:.3f}: {'HOLDS' if v[f'B {pl}'] else 'FAILS'}")
        q34, q33 = p90(pl, "all", "v34"), p90(pl, "all", "v33"); v[f"C {pl}"] = q34 < q33
        print(f"  C {pl}: p90 regret all {q34:.1%} vs v33 {q33:.1%}: {'HOLDS' if v[f'C {pl}'] else 'FAILS'}")
        d = paired[pl]; b = sum(x < 0 for x in d); wo = sum(x > 0 for x in d); pv = binom_tail(b, b + wo) if b + wo else 1.0
        v[f"D {pl}"] = b > wo and pv < 0.05
        print(f"  D {pl}: where v34 != v33 (n={len(d)}): better {b}, worse {wo}, ties {len(d)-b-wo}, one-sided p {pv:.4f}: "
              f"{'HOLDS' if v[f'D {pl}'] else 'FAILS'}")
    print("\nSUMMARY " + "  ".join(f"{h} {'HOLDS' if ok else 'FAILS'}" for h, ok in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["predict", "run", "evaluate"])
    ap.add_argument("out", nargs="?"); ap.add_argument("--sha"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"predict": lambda: predict(), "run": lambda: run(a), "evaluate": lambda: evaluate(a)}[a.mode]()
