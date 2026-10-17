"""heldout4_v3.py -- fourth registered held-out test: the two-candidate local check, and a replication of the rule.

THE LOCAL CHECK (localcheck.py). The rule gives a fleet size c1 (v3.4 for the annealer and the tour patrol, v3.3 for
the fixed-territory planners C4 and the cluster patrol). Theorem 1's runner-up c2 is the surrogate's best fleet size
at an adjacent stranding level (one sensor more or fewer stranded). The operator simulates or flies {c-1, c, c+1}
for c in {c1, c2} (4 sizes on average) and keeps the best measured one. It was worked out post hoc on test 3 and evaluated on all seen data
(seeds 1-49); its thresholds below were set from that data before any test-4 run.

FRESH FIELDS: seeds 50-57. Seeds 1-49 were used to design or test v3-v3.4 (42 and 123 occur in old files). No seed in
50-69 occurs in any CSV of the repository (sim-v2 at 230c5ff) or of the results_v2 archive (checked 2026-10-13).

DESIGN: the 20 primary cells x seeds 50-57 = 160 instances, T_mis = 12 h, planners sa, rr_tour, cyc (C4),
cluster_patrol. Sweeps from K_lo = min(K0, every checked size) to K_hi = max(max k_range, old + 2, v33 + 2, v34 + 2,
floor K_flight(S_1) + 2, every checked size), extended at both edges until every optimum is interior (or at K = 1)
and the three largest K exceed 1.5 x the minimum. Regret = J(K) / J(swept optimum) - 1; p90 is nearest rank.
For each planner, rule = v34 for sa and rr_tour, v33 for cyc and cluster_patrol; check = the local check of that
rule's candidates.

REGISTERED HYPOTHESES (24 verdicts).
Primary, the local check, each planner separately (16):
  L1  mean check regret < 1.0%.
  L2  share of instances where the check finds the swept optimum exactly >= 0.85 (sa, rr_tour), >= 0.80 (cyc,
      cluster_patrol).
  L3  instances with check regret > 10%: at most 3 of 160 (sa, rr_tour), at most 8 of 160 (cyc, cluster_patrol).
  L4  worst check regret <= 15% (sa, rr_tour), <= 25% (cyc, cluster_patrol).
Secondary, replication of the rule alone (8):
  R1  sa, rr_tour: v3.4 within-one, all 160 >= 0.85.
  R2  sa, rr_tour: v3.4 within-one, ring 48 >= 0.85.
  R3  sa, rr_tour: v3.4 p90 regret < old (v2.2 paper rule) p90 regret, all 160.
  R4  cyc, cluster_patrol: v3.3 worst regret < old worst regret, all 160.
REPORTED WITHOUT A VERDICT: every rule (old, v33, v34) alone and with a one-candidate +-1 check, by family; checked
sizes per instance; exact agreement; mean, median, max; check sizes that were not swept (should be 0).

REGISTRATION: `predict` writes heldout4_predictions.csv and prints a MANIFEST sha256 over the LF-normalised bytes of
heldout4_predictions.csv, heldout4_v3.py, localcheck.py, theory_v34.py and theory_v3.py. Publish it (and nothing
else) before `run`; `run` refuses to start if any of the five files differs.

usage:
  python heldout4_v3.py predict
  python heldout4_v3.py run results_v2/heldout4_v3.csv --sha <manifest sha256> [--procs 22]
  python heldout4_v3.py evaluate results_v2/heldout4_v3.csv
"""
import sys, os, csv, math, hashlib, argparse, itertools, collections, time
import numpy as np
from multiprocessing import Pool
from theory_v3 import field, old_rule, surrogate_rule
import theory_v34
import localcheck
from exante_rc import exante
from run_grid import one, k_range, FIELDS, SIM_VERSION

CELLS = ([("paper", M, 1.5e6) for M in (50, 100, 200, 400)] + [("paper", M, 3e6) for M in (50, 100, 200)]
         + [("paper", 100, 6e6)] + [(lay, M, E) for lay in ("ring", "core") for E in (1.5e6, 3e6) for M in (50, 100, 200)])
SEEDS = list(range(50, 58))
PLANNERS = ["sa", "rr_tour", "cyc", "cluster_patrol"]
SHARED = {"sa", "rr_tour"}                       # v3.4 tested for these; cyc and cluster_patrol use v3.3
L = 12600.0; TH = 43200.0; PRED = "heldout4_predictions.csv"
MANIFEST = [PRED, "heldout4_v3.py", "localcheck.py", "theory_v34.py", "theory_v3.py"]
THR = {"sa": (0.85, 3, 0.15), "rr_tour": (0.85, 3, 0.15), "cyc": (0.80, 8, 0.25), "cluster_patrol": (0.80, 8, 0.25)}


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
    v33, _ = surrogate_rule(p, F, rc, Pb, key=key, gray="exclude")
    v34, _ = theory_v34.surrogate_rule(p, F, rc, Pb, key=key, shared=True)
    sh1, sh2 = localcheck.candidates(p, F, rc, Pb, key=key, shared=True)
    tr1, tr2 = localcheck.candidates(p, F, rc, Pb, key=key, shared=False)
    assert sh1 == v34 and tr1 == v33, (lay, M, E, s, sh1, v34, tr1, v33)
    chk = localcheck.check_set(sh1, sh2) + localcheck.check_set(tr1, tr2)
    U = (1 - p.rho) * E; kfl1 = math.floor(p.v * U / (2 * p.Pf * np.sort(r)[-2]))
    lo = max(1, min([K0] + chk)); hi = max([max(kr), old + 2, v33 + 2, v34 + 2, kfl1 + 2] + chk)
    f = lambda x, n: round(float(x), n); g = lambda c: "" if c is None else int(c)
    return dict(layout=lay, M=int(M), Emax=float(E), L=float(L), seed=int(s), K0=int(K0), rc_exante=f(rc, 3),
                Pb_exante=f(Pb, 3), old=int(old), v33=int(v33), v34=int(v34), sh_c2=g(sh2), tr_c2=g(tr2),
                K_lo=int(lo), K_hi=int(hi))


def predict():
    rows = []
    for (lay, M, E), s in itertools.product(CELLS, SEEDS):
        d = instance(lay, M, E, s); rows.append(d)
        print(f"{lay} M={M} E={E/1e6} s={s}: old {d['old']} v33 {d['v33']} (c2 {d['tr_c2']}) v34 {d['v34']} "
              f"(c2 {d['sh_c2']}) sweep {d['K_lo']}-{d['K_hi']}", flush=True)
    with open(PRED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ns = [len(localcheck.check_set(r["v34"], r["sh_c2"] or None)) for r in rows]
    nt = [len(localcheck.check_set(r["v33"], r["tr_c2"] or None)) for r in rows]
    print(f"{len(rows)} instances; v34 != v33 in {sum(r['v34'] != r['v33'] for r in rows)}; "
          f"checked sizes per instance: shared {np.mean(ns):.2f}, territorial {np.mean(nt):.2f}; "
          f"sweep runs per planner {sum(r['K_hi'] - r['K_lo'] + 1 for r in rows)}")
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


def evaluate(a):
    P = load_pred(); D = read_out(a.out)
    print(f"manifest sha256 {manifest_sha()}")
    S = collections.defaultdict(lambda: collections.defaultdict(list)); cens = collections.Counter(); unswept = collections.Counter()
    for pl in PLANNERS:
        for k, pr in P.items():
            byK = D.get((pl,) + k)
            if not byK: continue
            if extension(byK): cens[pl] += 1
            Jv = {K: float(x["J"]) for K, x in byK.items()}; Ks = sorted(Jv); am = min(Jv, key=Jv.get)
            Jn = lambda K: Jv[min(Ks, key=lambda x: abs(x - K))]
            rg = lambda K: Jn(K) / Jv[am] - 1
            c2 = {"v34": pr["sh_c2"], "v33": pr["tr_c2"]}
            rule = "v34" if pl in SHARED else "v33"
            for fam in (k[0], "all"):
                R = S[(pl, fam)]
                for nm in ("old", "v33", "v34"):
                    K = int(pr[nm]); R[nm + "_w1"].append(abs(am - K) <= 1); R[nm + "_ex"].append(am == K); R[nm + "_rg"].append(rg(K))
                    one_set = localcheck.check_set(K, None)
                    R[nm + "_pm1"].append(min(Jn(q) for q in one_set) / Jv[am] - 1)
                chk = localcheck.check_set(int(pr[rule]), int(c2[rule]) if c2[rule] not in ("", None) else None)
                if fam == "all": unswept[pl] += sum(q not in Jv for q in chk)
                best = min(Jn(q) for q in chk) / Jv[am] - 1
                R["chk_rg"].append(best); R["chk_ex"].append(best < 1e-12); R["chk_n"].append(len(chk))
    for pl in PLANNERS:
        print(f"\n== {pl}  (rule {'v34' if pl in SHARED else 'v33'}; sweeps still needing extension: {cens[pl]}; "
              f"checked sizes not swept: {unswept[pl]})")
        for fam in ("all", "core", "paper", "ring"):
            R = S[(pl, fam)]
            if not R["old_rg"]: continue
            line = f"  {fam:5s} n={len(R['old_rg']):3d}"
            for nm in ("old", "v33", "v34"):
                g = np.array(R[nm + "_rg"]); h = np.array(R[nm + "_pm1"])
                line += (f" | {nm}: ex {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} mean {g.mean():.1%} "
                         f"p90 {p90nr(g):.1%} max {g.max():.1%} [+-1: mean {h.mean():.2%} max {h.max():.1%}]")
            c = np.array(R["chk_rg"])
            line += (f" | CHECK ({np.mean(R['chk_n']):.1f} sizes): ex {np.mean(R['chk_ex']):.2f} mean {c.mean():.2%} "
                     f"med {np.median(c):.1%} p90 {p90nr(c):.1%} max {c.max():.1%} >10%: {(c > 0.10).sum()}")
            print(line)
    v = {}
    print("\nREGISTERED HYPOTHESES")
    for pl in PLANNERS:
        R = S[(pl, "all")]
        if not R["chk_rg"]: continue
        c = np.array(R["chk_rg"]); ex_t, n10_t, mx_t = THR[pl]
        v[f"L1 {pl}"] = c.mean() < 0.01
        v[f"L2 {pl}"] = np.mean(R["chk_ex"]) >= ex_t
        v[f"L3 {pl}"] = int((c > 0.10).sum()) <= n10_t
        v[f"L4 {pl}"] = c.max() <= mx_t
        print(f"  L1 {pl}: mean check regret {c.mean():.2%} (< 1%): {'HOLDS' if v[f'L1 {pl}'] else 'FAILS'}")
        print(f"  L2 {pl}: exact {np.mean(R['chk_ex']):.3f} (>= {ex_t}): {'HOLDS' if v[f'L2 {pl}'] else 'FAILS'}")
        print(f"  L3 {pl}: > 10% in {(c > 0.10).sum()} of {len(c)} (<= {n10_t}): {'HOLDS' if v[f'L3 {pl}'] else 'FAILS'}")
        print(f"  L4 {pl}: worst {c.max():.1%} (<= {mx_t:.0%}): {'HOLDS' if v[f'L4 {pl}'] else 'FAILS'}")
    for pl in ("sa", "rr_tour"):
        A, G = S[(pl, "all")], S[(pl, "ring")]
        if not A["v34_w1"]: continue
        a, r = float(np.mean(A["v34_w1"])), float(np.mean(G["v34_w1"]))
        q34, qo = p90nr(A["v34_rg"]), p90nr(A["old_rg"])
        v[f"R1 {pl}"] = a >= 0.85; v[f"R2 {pl}"] = r >= 0.85; v[f"R3 {pl}"] = q34 < qo
        print(f"  R1 {pl}: v3.4 within-one all {a:.3f} (>= 0.85): {'HOLDS' if v[f'R1 {pl}'] else 'FAILS'}")
        print(f"  R2 {pl}: v3.4 within-one ring {r:.3f} (>= 0.85): {'HOLDS' if v[f'R2 {pl}'] else 'FAILS'}")
        print(f"  R3 {pl}: p90 v3.4 {q34:.1%} vs old {qo:.1%}: {'HOLDS' if v[f'R3 {pl}'] else 'FAILS'}")
    for pl in ("cyc", "cluster_patrol"):
        A = S[(pl, "all")]
        if not A["v33_rg"]: continue
        m33, mo = max(A["v33_rg"]), max(A["old_rg"]); v[f"R4 {pl}"] = m33 < mo
        print(f"  R4 {pl}: worst v3.3 {m33:.1%} vs old {mo:.1%}: {'HOLDS' if v[f'R4 {pl}'] else 'FAILS'}")
    print("\nSUMMARY " + "  ".join(f"{h} {'HOLDS' if ok else 'FAILS'}" for h, ok in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["predict", "run", "evaluate"])
    ap.add_argument("out", nargs="?"); ap.add_argument("--sha"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"predict": lambda: predict(), "run": lambda: run(a), "evaluate": lambda: evaluate(a)}[a.mode]()
