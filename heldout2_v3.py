"""heldout2_v3.py -- second registered held-out test of the v3 fleet-sizing theory (registered 2026-10-12).

Why a second test: the first (heldout_v3.py, seeds 13-20) turned out to include the planner comparison's pilot seeds
13-15 (fields swept before v3 existed), and it had no hypothesis for C4, a top-edge-only censoring check, a weak
horizon test (4 discriminating instances) and a median-based H1 that a tie at 0% regret can fail.

Fresh fields: seeds 21-40. No seed >= 21 appears in any CSV of the repository or of results_v2 (checked
2026-10-12; seeds 42 and 123 occur in two old files and are not used).

RULES (all a priori: site survey, nominal rates, airframe, inventory, mission length; ex-ante commute inputs at the
first K of the registered sweep, K0 = min k_range):
  old    min(floor Kr, round Kc)                         the paper's rule
  v3     argmin of the served-set surrogate (S1)         theory_v3.surrogate_rule(gray="charge"), as in test 1
  v33    v3 with gray-bracket exclusion                  theory_v3.surrogate_rule(gray="exclude")  <- PRIMARY
         (a fleet size that strands sensors is a candidate only if every stranded sensor is beyond flight-only
         reach, so none can still draw energy; designed on seeds 1-20 after the second review)
  v33_24 v33 for a 24 h mission (T_b + T = 86400 s)

ARMS
  main     20 primary cells x seeds 21-28 (160 instances), T_mis = 12 h, planners sa, rr_tour, cyc (C4),
           cluster_patrol.
  horizon  SA, T_mis = 24 h, on
           (a) every instance of the core and clustered cells with seeds 21-40 where v33 != v33_24 (the
               discriminating set, decided by the predictions only), and
           (b) controls: every core M=100 and M=200 instance (both budgets) with seeds 21-28 where v33 == v33_24.
           Discriminating instances with seed > 28 also get a 12 h SA sweep (seeds 21-28 have one from the main arm).

SWEEPS: K from K_lo = K0 to K_hi = max(max k_range, every predicted K + 2, floor K_flight(S_1) + 2), then extended
in rounds at BOTH edges until every sweep's optimum is interior (or at K = 1) and its three largest K exceed
1.5 x its minimum.

REGISTERED HYPOTHESES (primary rule v33; regret = J(predicted K) / J(swept optimum) - 1, predicted K replaced by the
nearest swept K if not swept; p90 = nearest rank, the smallest observed regret with at least 90% at or below it):
  H1  SA, core (48):     v33 p90 regret < old p90 regret  AND  v33 mean regret < old mean regret.
  H2  SA, all (160):     v33 within-one rate >= old within-one rate.
  H3  SA, ring (48):     commute bracket [round Kc, floor Kr] contains the optimum in >= 90%.
  H4  SA, all (160):     on the instances where v33 != old, v33 has lower regret than old more often than higher;
                         one-sided exact sign test p < 0.05 (equal regrets dropped).
  H5  C4, core (48):     v33 p90 regret < old p90 regret  AND  v33 within-one rate >= old within-one rate.
  H6  clustered (64), for rr_tour and for C4 separately:  v33 mean regret <= old mean regret + 1.0 percentage point
                         (the gray-exclusion fix: v3 broke this on clustered fields for the tour patrol).
  H7  horizon, discriminating set: the 24 h SA optimum is strictly closer to v33_24 than to v33 (12 h) in >= 75%
                         of instances AND one-sided binomial p < 0.05 (ties count against).
REPORTED WITHOUT A VERDICT (registered to be reported): every rule for every planner and family; cluster_patrol;
v3 (unmodified); the horizon arm's 2x2 table of predicted switches (v33 != v33_24) against observed ones (the SA
optimum moves by >= 2 UAVs between 12 h and 24 h; controls included); medians;
linear-interpolation p90 alongside nearest rank; sweeps still censored (must be 0).

REGISTRATION: `predict` writes heldout2_predictions.csv and prints a MANIFEST sha256 over the LF-normalised bytes of
heldout2_predictions.csv, heldout2_v3.py and theory_v3.py (in that order): the predictions, the evaluation code and
the rule. Publish the manifest hash (and nothing else) before `run`; `run` refuses to start if any of the three files
differs from it. (exante_rc.py is used only to make the predictions, which anyone can regenerate with `predict`.)

usage:
  python heldout2_v3.py predict
  python heldout2_v3.py run results_v2/heldout2_v3.csv --sha <manifest sha256> [--procs 22]
  python heldout2_v3.py evaluate results_v2/heldout2_v3.csv
"""
import sys, os, csv, math, hashlib, argparse, itertools, collections, time
import numpy as np
from multiprocessing import Pool
from theory_v3 import field, old_rule, surrogate_rule, bracket, caps
from exante_rc import exante
from run_grid import one, k_range, FIELDS, SIM_VERSION

CELLS = ([("paper", M, 1.5e6) for M in (50, 100, 200, 400)] + [("paper", M, 3e6) for M in (50, 100, 200)]
         + [("paper", 100, 6e6)] + [(lay, M, E) for lay in ("ring", "core") for E in (1.5e6, 3e6) for M in (50, 100, 200)])
MAIN_SEEDS = list(range(21, 29)); HOR_SEEDS = list(range(21, 41))
MAIN_PLANNERS = ["sa", "rr_tour", "cyc", "cluster_patrol"]
L = 12600.0; TH12 = 43200.0; TH24 = 86400.0; PRED = "heldout2_predictions.csv"
MANIFEST = [PRED, "heldout2_v3.py", "theory_v3.py"]


def lf_bytes(path):
    return open(path, "rb").read().replace(b"\r\n", b"\n")


def manifest_sha():
    h = hashlib.sha256()
    for f in MANIFEST: h.update(hashlib.sha256(lf_bytes(f)).hexdigest().encode() + b"  " + f.encode() + b"\n")
    return h.hexdigest()


def instance(lay, M, E, s):
    p, F = field(lay, M, E, L, s); r = np.linalg.norm(F.pos - p.home, axis=1)
    kr = k_range(lay, M, E, s, L); K0 = min(kr)
    rc, Pb, _ = exante(p, r, np.full(M, p.lam_bits), F.pos, K0)
    key = (lay, M, L, s)
    old = old_rule(p, r, rc, Pb)
    v3, j3 = surrogate_rule(p, F, rc, Pb, key=key)
    v33, j33 = surrogate_rule(p, F, rc, Pb, key=key, gray="exclude")
    v33_24, j24 = surrogate_rule(p, F, rc, Pb, key=key, gray="exclude", horizon=TH24)
    br = bracket(p, r, rc, Pb); Kc, Kr = caps(p, r, rc, Pb)
    U = (1 - p.rho) * E; kfl1 = math.floor(p.v * U / (2 * p.Pf * np.sort(r)[-2]))
    hi = max(max(kr), old + 2, v3 + 2, v33 + 2, v33_24 + 2, kfl1 + 2)
    f = lambda x, n: round(float(x), n)                         # plain floats: same text on every numpy version
    return dict(layout=lay, M=int(M), Emax=float(E), L=float(L), seed=int(s), K0=int(K0), rc_exante=f(rc, 3),
                Pb_exante=f(Pb, 3), Kc=f(Kc, 4), Kr=f(Kr, 4), old=int(old), v3=int(v3), v3_stranded=int(j3), v33=int(v33),
                v33_stranded=int(j33), v33_24=int(v33_24), v33_24_stranded=int(j24), br_lo=(int(br[0]) if br else ""),
                br_hi=(int(br[1]) if br else ""), K_lo=int(K0), K_hi=int(hi))


def predict():
    rows = []
    for (lay, M, E), s in itertools.product(CELLS, HOR_SEEDS):
        main = s in MAIN_SEEDS
        if not main and lay == "ring": continue                     # seeds 29-40 only feed the horizon arm
        d = instance(lay, M, E, s)
        disc = d["v33"] != d["v33_24"]
        ctrl = (main and lay == "core" and M in (100, 200) and not disc)
        d.update(main=int(main), horizon=("disc" if disc and lay in ("core", "paper") else "ctrl" if ctrl else ""))
        if main or d["horizon"]: rows.append(d)
        print(f"{lay} M={M} E={E/1e6} s={s}: old {d['old']} v3 {d['v3']} v33 {d['v33']} v33_24 {d['v33_24']} "
              f"main {d['main']} horizon {d['horizon'] or '-'} sweep {d['K_lo']}-{d['K_hi']}", flush=True)
    with open(PRED, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_main = sum(r["main"] for r in rows); n_disc = sum(r["horizon"] == "disc" for r in rows)
    n_ctrl = sum(r["horizon"] == "ctrl" for r in rows)
    print(f"main {n_main} instances; horizon: {n_disc} discriminating, {n_ctrl} controls")
    print("MANIFEST sha256", manifest_sha())


def load_pred():
    return {(r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"])): r for r in csv.DictReader(open(PRED))}


def read_out(path):
    J = collections.defaultdict(dict)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        for r in csv.DictReader(open(path)):
            if r.get("sim_version") != SIM_VERSION: sys.exit(f"{path}: row with sim_version {r.get('sim_version')}")
            J[(r["planner"], r["layout"], int(r["M"]), float(r["Emax"]), int(r["seed"]), float(r["Th"]))][int(r["K"])] = r
    return J


def extension(byK):
    """Fleet sizes still to add: two-sided (optimum interior or at K = 1) plus the 1.5x rule at the top."""
    Ks = sorted(byK); Jv = {K: float(byK[K]["J"]) for K in Ks}; am = min(Jv, key=Jv.get); add = set()
    if am >= Ks[-1] - 1 or not all(Jv[K] > 1.5 * Jv[am] for K in Ks[-3:]): add |= {Ks[-1] + 1, Ks[-1] + 2}
    if am == Ks[0] and Ks[0] > 1: add |= {K for K in (Ks[0] - 1, Ks[0] - 2) if K >= 1}
    return sorted(add - set(Ks))


def wanted(P):
    """(planner, instance key, Th) triples of the design."""
    for k, pr in P.items():
        if int(pr["main"]):
            for pl in MAIN_PLANNERS: yield pl, k, TH12
        if pr["horizon"]:
            yield "sa", k, TH24
            if not int(pr["main"]): yield "sa", k, TH12


def run(a):
    sha = manifest_sha()
    if sha != a.sha: sys.exit(f"manifest sha256 {sha} != registered {a.sha}: a registered file changed")
    P = load_pred(); new = (not os.path.exists(a.out)) or os.path.getsize(a.out) == 0
    with open(a.out, "a", newline="") as f, Pool(a.procs) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new: w.writeheader(); f.flush()
        for rnd in itertools.count():
            done = read_out(a.out); jobs = []
            for pl, (lay, M, E, s), Th in wanted(P):
                pr = P[(lay, M, E, s)]; have = done.get((pl, lay, M, E, s, Th), {})
                Ks = range(int(pr["K_lo"]), int(pr["K_hi"]) + 1) if (rnd == 0 or not have) else extension(have)
                jobs += [(lay, M, E, K, s, "exclude", "launch", 0, 0.0, "prior", 0.0, pl, L, Th, 1200)
                         for K in Ks if K not in have]
            if not jobs:
                print("ALL DONE: every sweep interior (or at K = 1) and extended", flush=True); return
            jobs.sort(key=lambda j: (-j[13], -j[1], -j[3]))
            print(f"round {rnd}: {len(jobs)} runs", flush=True); t0 = time.time()
            for i, row in enumerate(pool.imap_unordered(one, jobs)):
                w.writerow(row); f.flush()
                if i % 20 == 0:
                    print(f"[{i+1}/{len(jobs)}] {row['planner']} {row['layout']} M={row['M']} K={row['K']} s={row['seed']} "
                          f"Th={float(row['Th'])/3600:.0f}h ({time.time()-t0:.0f}s)", flush=True)


def p90nr(x):
    if len(x) == 0: return float("nan")                      # empty group: every comparison is False (FAILS)
    x = np.sort(np.asarray(x)); return float(x[max(0, math.ceil(0.9 * len(x)) - 1)])


def binom_tail(k, n):          # P(X >= k), X ~ Bin(n, 1/2)
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def evaluate(a):
    P = load_pred(); D = read_out(a.out)
    print(f"manifest sha256 {manifest_sha()}")
    RULES = ("old", "v3", "v33")
    S = collections.defaultdict(lambda: collections.defaultdict(list)); cens = collections.Counter(); paired = []
    for pl in MAIN_PLANNERS:
        for k, pr in P.items():
            if not int(pr["main"]): continue
            byK = D.get((pl,) + k + (TH12,))
            if not byK: continue
            if extension(byK): cens[pl] += 1
            Jv = {K: float(x["J"]) for K, x in byK.items()}; Ks = sorted(Jv); am = min(Jv, key=Jv.get)
            rg = lambda K: Jv[min(Ks, key=lambda x: abs(x - K))] / Jv[am] - 1
            for fam in (k[0], "all"):
                for nm in RULES:
                    K = int(pr[nm]); S[(pl, fam)][nm + "_w1"].append(abs(am - K) <= 1); S[(pl, fam)][nm + "_rg"].append(rg(K))
                    S[(pl, fam)][nm + "_ex"].append(am == K)
                if pr["br_lo"] != "": S[(pl, fam)]["br"].append(int(pr["br_lo"]) <= am <= int(pr["br_hi"]))
            if pl == "sa" and int(pr["v33"]) != int(pr["old"]): paired.append(rg(int(pr["v33"])) - rg(int(pr["old"])))
    for pl in MAIN_PLANNERS:
        print(f"\n== {pl}  (sweeps still needing extension: {cens[pl]})")
        for fam in ("all", "core", "paper", "ring"):
            R = S[(pl, fam)]
            if not R["old_rg"]: continue
            line = f"  {fam:5s} n={len(R['old_rg']):3d}"
            for nm in RULES:
                g = np.array(R[nm + "_rg"])
                line += (f" | {nm}: ex {np.mean(R[nm+'_ex']):.2f} w1 {np.mean(R[nm+'_w1']):.2f} mean {g.mean():.1%} "
                         f"med {np.median(g):.1%} p90 {p90nr(g):.1%} (lin {np.quantile(g, .9):.1%}) max {g.max():.1%}")
            if R["br"]: line += f" | bracket {sum(R['br'])}/{len(R['br'])}"
            print(line)

    def mean_rg(pl, fam, nm): g = S[(pl, fam)][nm + "_rg"]; return float(np.mean(g)) if g else float("nan")
    def p90(pl, fam, nm): return p90nr(S[(pl, fam)][nm + "_rg"])
    print("\nREGISTERED HYPOTHESES (primary rule v33)")
    v = {}
    v["H1"] = p90("sa", "core", "v33") < p90("sa", "core", "old") and mean_rg("sa", "core", "v33") < mean_rg("sa", "core", "old")
    print(f"  H1 SA core: p90 {p90('sa','core','v33'):.1%} vs {p90('sa','core','old'):.1%}, mean "
          f"{mean_rg('sa','core','v33'):.1%} vs {mean_rg('sa','core','old'):.1%}: {'HOLDS' if v['H1'] else 'FAILS'}")
    rate = lambda pl, fam, nm: float(np.mean(S[(pl, fam)][nm + "_w1"])) if S[(pl, fam)][nm + "_w1"] else float("nan")
    w33, w0 = rate("sa", "all", "v33"), rate("sa", "all", "old")
    v["H2"] = w33 >= w0; print(f"  H2 SA all within-one: {w33:.3f} vs {w0:.3f}: {'HOLDS' if v['H2'] else 'FAILS'}")
    br = S[("sa", "ring")]["br"]; v["H3"] = bool(br) and np.mean(br) >= 0.9
    print(f"  H3 SA ring bracket: {sum(br)}/{len(br)}: {'HOLDS' if v['H3'] else 'FAILS'}")
    b = sum(d < 0 for d in paired); wo = sum(d > 0 for d in paired); pv = binom_tail(b, b + wo) if b + wo else 1.0
    v["H4"] = b > wo and pv < 0.05
    print(f"  H4 SA paired (v33 != old, n={len(paired)}): better {b}, worse {wo}, ties {len(paired)-b-wo}, "
          f"one-sided p {pv:.4f}: {'HOLDS' if v['H4'] else 'FAILS'}")
    c33, c0 = rate("cyc", "core", "v33"), rate("cyc", "core", "old")
    v["H5"] = p90("cyc", "core", "v33") < p90("cyc", "core", "old") and c33 >= c0
    print(f"  H5 C4 core: p90 {p90('cyc','core','v33'):.1%} vs {p90('cyc','core','old'):.1%}, within-one {c33:.3f} vs "
          f"{c0:.3f}: {'HOLDS' if v['H5'] else 'FAILS'}")
    h6 = []
    for pl in ("rr_tour", "cyc"):
        d = mean_rg(pl, "paper", "v33") - mean_rg(pl, "paper", "old"); h6.append(d <= 0.01)
        print(f"  H6 {pl} clustered: mean regret v33 - old = {100*d:+.2f} points (limit +1.00): {'HOLDS' if d <= 0.01 else 'FAILS'}")
    v["H6"] = all(h6)
    # horizon arm
    closer = []; tab = collections.Counter(); lines = []
    for k, pr in P.items():
        if not pr["horizon"]: continue
        c24 = D.get(("sa",) + k + (TH24,)); c12 = D.get(("sa",) + k + (TH12,))
        if not c24 or not c12: continue
        J24 = {K: float(x["J"]) for K, x in c24.items()}; J12 = {K: float(x["J"]) for K, x in c12.items()}
        a24 = min(J24, key=J24.get); a12 = min(J12, key=J12.get); n12, n24 = int(pr["v33"]), int(pr["v33_24"])
        obs = abs(a12 - a24) >= 2; pred = (n12 != n24); tab[(pred, obs)] += 1       # a move of 1 UAV is sweep noise
        if pr["horizon"] == "disc": closer.append(abs(a24 - n24) < abs(a24 - n12))
        Ks = sorted(J24); g = lambda K: J24[min(Ks, key=lambda x: abs(x - K))] / J24[a24] - 1
        lines.append(f"    {pr['horizon']:4s} {k[0]} M={k[1]} E={k[2]/1e6} s={k[3]}: opt 12 h {a12}, 24 h {a24} | v33 {n12} "
                     f"({g(n12):.1%})  v33_24 {n24} ({g(n24):.1%})  old {pr['old']} ({g(int(pr['old'])):.1%})"
                     + (f"  sweeps need extension" if extension(c24) or extension(c12) else ""))
    print("\n  horizon arm (24 h regrets):"); print("\n".join(sorted(lines)))
    n = len(closer); kk = sum(closer); pv7 = binom_tail(kk, n) if n else 1.0
    v["H7"] = n > 0 and kk >= 0.75 * n and pv7 < 0.05
    print(f"  H7 horizon: closer to v33_24 in {kk} of {n}, one-sided p {pv7:.4f}: {'HOLDS' if v['H7'] else 'FAILS'}")
    print(f"  2x2 (predicted switch, observed optimum change 12 h -> 24 h): "
          f"TP {tab[(True, True)]}  FN {tab[(False, True)]}  FP {tab[(True, False)]}  TN {tab[(False, False)]}")
    print("\nSUMMARY " + "  ".join(f"{h} {'HOLDS' if ok else 'FAILS'}" for h, ok in v.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["predict", "run", "evaluate"])
    ap.add_argument("out", nargs="?"); ap.add_argument("--sha"); ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    {"predict": lambda: predict(), "run": lambda: run(a), "evaluate": lambda: evaluate(a)}[a.mode]()
