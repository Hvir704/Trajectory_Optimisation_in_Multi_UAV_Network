"""
validate.py -- decisive tests for the dynamic K* law (CONTEXT_74).

Two modes:

  --mode paired    PAIRED-DIFFERENCE TEST (the decisive one).
                   Tests whether K_cross actually minimises J, using paired
                   per-instance differences on identical instances. This is the
                   test the J argmin could not do: instance-layout variance is
                   ~25% of J_mean and swamps real differences in an unpaired
                   comparison, but cancels exactly under pairing.

  --mode audit     ASSUMPTION AUDIT. Measures the three quantities the
                   derivation assumes away in the T_rev -> J step:
                     CV of inter-visit times   (assumed 0 / periodic)
                     corr(weight, age)         (assumed 0)
                     abandoned fraction        (assumed 0)
                   plus the T_rev formula-vs-measured ratio.

WHY THESE TWO
  CONTEXT_74 found the derivation's central step (min T_rev => min J) rests on
  three assumptions that all fail measurably. But the paired test at M=100 showed
  the formula's answer still lands inside the true optimal region {4,5}. The open
  question is whether that survives at M=400, where every neglected factor is
  largest (abandonment 21-30%, corr(w,age) -0.65). THAT is the discriminating
  experiment.

REQUIREMENTS
  scipy is used for the paired t-test if available; falls back to a sign test
  and a normal approximation if not. Nothing else beyond dyn_env/sa_sortie.

RUN
  python validate.py --mode paired --M 400 --K 3 4 5 6 --instances 8
  python validate.py --mode audit  --M 100 400 --K 3 4 5 6
"""
from __future__ import annotations

import argparse
import math

import numpy as np

from dyn_env import DynParams, DynSim, SensorField
from sa_sortie import build_sa_planner

try:
    from scipy import stats as _st
except Exception:
    _st = None


# ---------------------------------------------------------------- visit logging
_orig_visit = SensorField.visit


def _logged_visit(self, t, i):
    if not hasattr(self, "_vlog"):
        self._vlog = {}
    self._vlog.setdefault(i, []).append(t)
    return _orig_visit(self, t, i)


def run_one(M, K, seed, iters, Th, Tb, log=False):
    if log:
        SensorField.visit = _logged_visit
    else:
        SensorField.visit = _orig_visit
    p = DynParams(M=M, K=K, T_horizon=Th, T_burnin=Tb)
    s = DynSim(p, planner=build_sa_planner(iters=iters), seed=seed)
    m = s.run()
    return s, m


# ---------------------------------------------------------------- paired test
def paired_pvalue(a, b):
    """H0: no difference. Returns (p, method). a,b paired samples."""
    d = np.asarray(b) - np.asarray(a)
    if _st is not None:
        return float(_st.ttest_rel(b, a).pvalue), "paired t"
    n = len(d)
    if d.std(ddof=1) == 0:
        return (0.0 if d.mean() != 0 else 1.0), "degenerate"
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(p), "normal approx"


def mode_paired(a):
    print("=" * 96)
    print(f"  PAIRED-DIFFERENCE TEST   M={a.M}  K={a.K}  n={a.instances} "
          f"identical instances  iters={a.iters}")
    print("  H0: J(K) equals J(K_best).  Pairing removes instance-layout variance.")
    print("=" * 96)

    for M in a.M:
        seeds = list(range(1, a.instances + 1))
        res = {}
        for K in a.K:
            res[K] = [run_one(M, K, s, a.iters, a.T_horizon, a.T_burnin)[1]["J_timeavg"]
                      for s in seeds]
        best = min(a.K, key=lambda K: np.mean(res[K]))

        print(f"\n--- M={M} ---")
        for K in a.K:
            print(f"  K={K}: mean J = {np.mean(res[K]):>14,.0f}"
                  f"{'   <= best' if K == best else ''}")
        print()
        print(f"  {'vs':>6} {'mean paired diff':>18} {'favour best':>12} {'p':>9}  verdict")
        tied = [best]
        for K in a.K:
            if K == best:
                continue
            d = np.array(res[K]) - np.array(res[best])   # >0 => best is better
            p, meth = paired_pvalue(res[best], res[K])
            win = int(sum(1 for x in d if x > 0))
            if p < 0.05:
                verdict = f"K={best} genuinely better"
            else:
                verdict = f"TIE with K={best}"
                tied.append(K)
            print(f"  K={K:>4} {d.mean():>+18,.0f} {win:>7}/{len(d)} {p:>9.4f}  {verdict}")

        print(f"\n  OPTIMAL REGION (not significantly different): "
              f"{sorted(tied)}   [{meth}]")
        print(f"  Compare against K_cross from kstar_crossing.py for this cell.")
        print(f"  If K_cross falls INSIDE this region, the formula is defensible here.")
        print(f"  If OUTSIDE, the formula fails at this M -- report it.")


# ---------------------------------------------------------------- audit
def mode_audit(a):
    print("=" * 108)
    print("  ASSUMPTION AUDIT -- the three things the T_rev -> J step assumes away")
    print("  derivation assumes: CV=0 (periodic), corr(w,age)=0, abandonment=0")
    print("=" * 108)
    print(f"{'M':>5} {'K':>3} {'J':>13} {'aband%':>7} {'CV':>6} {'1+CV^2':>7} "
          f"{'Trev_meas':>10} {'Trev_frm':>9} {'frm/meas':>9} {'corr(w,age)':>11}")

    for M in a.M:
        rows = []
        for K in a.K:
            s, m = run_one(M, K, a.seed, a.iters, a.T_horizon, a.T_burnin, log=True)
            F = s.field
            vlog = getattr(F, "_vlog", {})
            n_ab = M - len(set(vlog.keys()))
            cvs, means = [], []
            for i, ts in vlog.items():
                ts = [t for t in ts if t >= a.T_burnin]
                if len(ts) >= 3:
                    d = np.diff(ts)
                    if d.mean() > 0:
                        cvs.append(d.std() / d.mean())
                        means.append(d.mean())
            cv = float(np.mean(cvs)) if cvs else float("nan")
            tmeas = float(np.mean(means)) if means else float("nan")
            tfrm = m["T_rev"]
            corr = float(np.corrcoef(F.wi_base, F.age(s._clock))[0, 1])
            ratio = tfrm / tmeas if tmeas == tmeas and tmeas > 0 else float("nan")
            print(f"{M:>5} {K:>3} {m['J_timeavg']:>13,.0f} {100*n_ab/M:>6.1f}% "
                  f"{cv:>6.2f} {1+cv**2:>7.2f} {tmeas:>10.0f} {tfrm:>9.0f} "
                  f"{ratio:>9.2f} {corr:>11.3f}")
            rows.append((K, cv, corr, 100 * n_ab / M, ratio))

        cvf = [1 + r[1] ** 2 for r in rows]
        print(f"\n  M={M} summary:")
        print(f"    (1+CV^2) varies {100*(max(cvf)/min(cvf)-1):.0f}% across K "
              f"-- derivation drops this entirely, and it is NOT constant")
        print(f"    corr(w,age) in [{min(r[2] for r in rows):.2f}, "
              f"{max(r[2] for r in rows):.2f}] -- assumed 0")
        print(f"    abandonment {min(r[3] for r in rows):.1f}%-{max(r[3] for r in rows):.1f}% "
              f"-- assumed 0; these nodes are outside the T_rev accounting entirely")
        print(f"    T_rev formula overstates measured by up to "
              f"{100*(max(r[4] for r in rows)-1):.0f}%\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["paired", "audit"], required=True)
    ap.add_argument("--M", type=int, nargs="+", default=[100])
    ap.add_argument("--K", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--instances", type=int, default=8, help="paired mode only")
    ap.add_argument("--seed", type=int, default=1, help="audit mode only")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--T-horizon", type=float, default=4 * 3600.0)
    ap.add_argument("--T-burnin", type=float, default=1 * 3600.0)
    a = ap.parse_args()
    (mode_paired if a.mode == "paired" else mode_audit)(a)


if __name__ == "__main__":
    main()