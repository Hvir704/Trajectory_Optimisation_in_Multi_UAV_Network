r"""
kstar_frozen.py -- FROZEN static fleet-sizing predictor.   WS10 deliverable.
============================================================================
(M, Emax) -> K*, plus every derivation term, plus honest validity flags.

Stdlib only. No solver, no project imports, no data files. Run it anywhere:

    python kstar_frozen.py --M 100 --Emax 50000
    python kstar_frozen.py --M 50 100 200 400 --Emax 25000 50000 100000
    python kstar_frozen.py --surface            # reprint the validated surface
    python kstar_frozen.py --selftest           # verify against measured K*

WHAT THIS SCRIPT IS
-------------------
The headline law is EMPIRICAL and GLOBAL:

    K* = round(Emax / e*),   e* = 12,500 J   (single constant, M-free)

It reproduces the SA-measured K* in 8 of 9 validated cells (see --selftest).
The one miss is M=50 / Emax=100k, where coverage saturates (see FLAGS below).

The DECOMPOSITION of e* is LOCAL, not global. Budget invariance was tested
and FAILED: e_min drifts +95..155% over a 4x budget range, so e* computed
from coverage primitives drifts +45..49% and is only valid near the budget it
was measured at. The script therefore reports the decomposition as a
diagnostic anchored to the nearest measured budget, NEVER as the predictor.
Do not substitute e_star_local for E_STAR when predicting.

    e*_local = sqrt( e_min^2 + mid*e_min ),    mid = 2*r_cap/a

CONSTANTS AND THEIR PROVENANCE
------------------------------
  E_STAR   12,500 J   empirical global fit; reproduces 2/4/8 at 25k/50k/100k
  C_R      2.47       r_cap = C_R*sqrt(M); from measured plateau
                      17.25/25.333/33.833/50.34 at M=50/100/200/400
                      (independently corroborated by the timing channel:
                       predicted 17.5/23.3/32.0 under uniform weights)
  ALPHA    0.21-0.28  Rg^2 = ALPHA*n*L^2/M, across-cell relation, M-free,
                      weakly budget-dependent (+10..20% per budget doubling)
  A_TAB/EMIN_TAB      per-(M,Emax) coverage-fit primitives, measured on the
                      12/2000/2 grid at instance seed 2025

FLAGS THIS SCRIPT RAISES
------------------------
  BUDGET-SLACK    Emax > 100k. Law breaks (Remark 5): coverage saturates and
                  the linear predictor under-shoots by 1.5-2, one-sided.
  SATURATED       saturation index s = K*_law*C_R/sqrt(M) >= 2.5. The only
                  measured cell in this regime (M=50, 100k, s=2.79) had
                  K*_SA = 9 vs K*_law = 8, i.e. the law under-predicts by +1.
  NEAR-SATURATED  2.0 <= s < 2.5. No measured deviation, but adjacent to one.
  EXTRAPOLATED    (M, Emax) outside the validated box M in [50,400],
                  Emax in [25k,100k].

WHAT IS NOT CLAIMED
-------------------
  * e* is NOT derived exactly. 12,500 is a global fit; the derived-band
    values were 11,947/11,404/10,974 (M=50/100/200) against a measured
    11,765 band. Quote against the band, never as an exact derivation.
  * The clustered-tour geometry assumption is FALSIFIED (see the WS10
    closure doc). Rg^2 vs n is flat WITHIN a fixed energy budget under both
    heterogeneous and uniform weights, at both 50k and 100k. Step 0 survives
    only in its energy-parameterised form: per-drone energy sets swept area,
    and n follows from area x density. Do NOT write "derived from
    clustered-tour geometry".
  * K* = Emax/e* with a LOCAL e* predicts sublinear growth in Emax
    (~Emax^0.72) whereas SA measures exact doubling. Only the global
    constant reproduces the measured surface.
"""
import argparse
import math

# ---------------------------------------------------------------- constants
E_STAR = 12500.0          # J   global empirical fleet-sizing constant
C_R = 2.47                # r_cap = C_R*sqrt(M)
ALPHA_LO, ALPHA_HI = 0.21, 0.28

# physical constants (compare_baseline.py)
PF, PH, V, L = 150.0, 200.0, 20.0, 1000.0
EPS = PF / V              # 7.5 J/m
TH1, TH2 = 0.01, 1.0
RHO = 0.38260             # mean depot distance / L, centre depot
KAPPA = 1.042             # d_last*sqrt(M)/L, uniform weights, +/-4.6%
MID_DERIVED = 2.0 * PF * (TH2 / TH1)      # 30,000 J

VALID_M = (50, 400)
VALID_E = (25000.0, 100000.0)

# coverage-fit primitives: (M, Emax) -> (a [nodes/J], e_min [J])
# fitted on N(K) = A - B*K strictly past the coverage peak, 12/2000/2, seed 2025
A_TAB = {
    (50, 25000): 1.0240e-3, (100, 25000): 1.3404e-3, (200, 25000): 1.5974e-3,
    (50, 50000): 0.9713e-3, (100, 50000): 1.3367e-3, (200, 50000): 1.6925e-3,
    (100, 100000): 1.2850e-3, (200, 100000): 1.9834e-3,
}
EMIN_TAB = {
    (50, 25000): 2363.0, (100, 25000): 2070.0, (200, 25000): 1613.0,
    (50, 50000): 3287.0, (100, 50000): 3002.0, (200, 50000): 2533.0,
    (100, 100000): 4028.0, (200, 100000): 4117.0,
}
# M=50 @ 100k omitted: 99.5% coverage-clipped, A/B not identifiable.

# SA-measured K* (solver of record). 12/2000/2, instance seed 2025.
KSTAR_SA = {
    (50, 25000): 2, (100, 25000): 2, (200, 25000): 2,
    (50, 50000): 4, (100, 50000): 4, (200, 50000): 4,
    (50, 100000): 9, (100, 100000): 8, (200, 100000): 8,
}


def r_cap(M):
    """Objective-cap plateau: max nodes one unconstrained UAV will serve."""
    return C_R * math.sqrt(M)


def kstar(Emax):
    """The frozen law. Global constant, M-free."""
    return max(1, int(round(Emax / E_STAR)))


def _nearest_budget(M, Emax):
    keys = [k for k in A_TAB if k[0] == M]
    if not keys:
        return None
    return min(keys, key=lambda k: abs(k[1] - Emax))


def local_decomposition(M, Emax):
    """Diagnostic only. Anchored at the nearest measured budget; NOT global."""
    key = _nearest_budget(M, Emax)
    if key is None:
        return None
    a, e_min = A_TAB[key], EMIN_TAB[key]
    mid = 2.0 * r_cap(M) / a
    e_star_local = math.sqrt(e_min ** 2 + mid * e_min)
    return dict(anchor_M=key[0], anchor_Emax=key[1], a=a, e_min=e_min,
                mid=mid, e_star_local=e_star_local,
                K_local=Emax / e_star_local,
                extrapolated=(key[1] != Emax))


def flags(M, Emax, K):
    out = []
    s = K * C_R / math.sqrt(M)
    if Emax > VALID_E[1]:
        out.append('BUDGET-SLACK: law breaks above 100k (Remark 5); '
                   'linear predictor under-shoots 1.5-2, one-sided')
    if s >= 2.5:
        out.append(f'SATURATED (s={s:.2f}): coverage ceiling active; the one '
                   f'measured cell here (M=50,100k,s=2.79) gave K*_SA=9 vs '
                   f'K*_law=8 -- expect the law to under-predict by +1')
    elif s >= 2.0:
        out.append(f'NEAR-SATURATED (s={s:.2f}): no measured deviation, but '
                   f'adjacent to the saturated cell')
    if not (VALID_M[0] <= M <= VALID_M[1]) or not (VALID_E[0] <= Emax <= VALID_E[1]):
        out.append('EXTRAPOLATED: outside the validated box '
                   'M in [50,400], Emax in [25k,100k]')
    return out, s


def predict(M, Emax):
    K = kstar(Emax)
    fl, s = flags(M, Emax, K)
    return dict(M=M, Emax=Emax, K_star=K, r_cap=r_cap(M),
                saturation_index=s, coverage_at_kstar=K * r_cap(M),
                flags=fl, local=local_decomposition(M, Emax),
                K_star_SA=KSTAR_SA.get((M, Emax)))


def report(M, Emax):
    p = predict(M, Emax)
    bar = '-' * 74
    print(bar)
    print(f'  M = {M}   Emax = {int(Emax):,} J')
    print(bar)
    print(f'  K*  =  round(Emax / e*)  =  round({Emax:.0f} / {E_STAR:.0f})  '
          f'=  {p["K_star"]}')
    if p['K_star_SA'] is not None:
        ok = 'MATCH' if p['K_star_SA'] == p['K_star'] else 'DEVIATES'
        print(f'  K*_SA (measured, solver of record) = {p["K_star_SA"]}   [{ok}]')
    print()
    print(f'  r_cap            {p["r_cap"]:.2f} nodes   (= {C_R}*sqrt(M))')
    print(f'  K* * r_cap       {p["coverage_at_kstar"]:.1f} nodes vs M = {M}')
    print(f'  saturation index {p["saturation_index"]:.2f}   '
          f'(K*_law*C_R/sqrt(M); >=2.5 is saturated)')
    loc = p['local']
    if loc:
        tag = ' [ANCHOR EXTRAPOLATED]' if loc['extrapolated'] else ''
        print()
        print(f'  LOCAL decomposition, anchored at Emax='
              f'{int(loc["anchor_Emax"]):,}{tag}   -- DIAGNOSTIC ONLY')
        print(f'    a       {loc["a"]*1e5:.3f} e-5 nodes/J   (budget-invariant '
              f'to ~10%: the one primitive that survived)')
        print(f'    e_min   {loc["e_min"]:.0f} J             (NOT budget-invariant: '
              f'+95..155% over 4x)')
        print(f'    mid     {loc["mid"]:.0f} J             (= 2*r_cap/a; derived '
              f'value 2*PF*(th2/th1) = {MID_DERIVED:.0f} J)')
        print(f'    e*_local {loc["e_star_local"]:.0f} J            -> K = '
              f'{loc["K_local"]:.2f}  (do NOT use as predictor)')
    if p['flags']:
        print()
        for f in p['flags']:
            print(f'  !! {f}')
    print(bar)
    return p


def surface():
    print('VALIDATED SURFACE  (SA, 12/2000/2, instance seed 2025)')
    print(f'  {"M":>5} {"Emax":>8} {"K*_SA":>6} {"K*_law":>7} {"match":>6} '
          f'{"sat idx":>8}')
    ok = 0
    for (M, E), k in sorted(KSTAR_SA.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        kl = kstar(E)
        s = kl * C_R / math.sqrt(M)
        m = 'yes' if kl == k else 'NO'
        ok += (kl == k)
        print(f'  {M:>5} {int(E):>8} {k:>6} {kl:>7} {m:>6} {s:>8.2f}')
    print(f'  {ok}/{len(KSTAR_SA)} exact. The miss is M=50/100k (saturated).')


def selftest():
    print('SELFTEST: frozen law vs SA-measured K*')
    bad = []
    for (M, E), k in sorted(KSTAR_SA.items()):
        kl = kstar(E)
        if kl != k:
            bad.append((M, E, k, kl))
    print(f'  {len(KSTAR_SA)-len(bad)}/{len(KSTAR_SA)} cells exact.')
    for M, E, k, kl in bad:
        s = kl * C_R / math.sqrt(M)
        print(f'  DEVIATION  M={M} Emax={int(E)}: SA={k} law={kl} '
              f'(saturation index {s:.2f} -- expected, documented)')
    exp_sat = all(kl * C_R / math.sqrt(M) >= 2.5 for M, E, k, kl in bad)
    print('  All deviations are inside the SATURATED flag: '
          + ('YES' if exp_sat else 'NO -- INVESTIGATE'))
    return len(bad), exp_sat


def main():
    ap = argparse.ArgumentParser(description='Frozen static K* predictor (WS10).')
    ap.add_argument('--M', type=int, nargs='+', default=[100])
    ap.add_argument('--Emax', type=float, nargs='+', default=[50000])
    ap.add_argument('--surface', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    g = ap.parse_args()
    if g.surface:
        surface(); return
    if g.selftest:
        selftest(); return
    for M in g.M:
        for E in g.Emax:
            report(M, E)


if __name__ == '__main__':
    main()
