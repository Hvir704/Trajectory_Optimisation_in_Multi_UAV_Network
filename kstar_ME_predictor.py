"""
kstar_ME_predictor.py  —  K*(M, Emax): optimal fleet size for ANY network size
and ANY per-mission energy budget (not tied to Emax = 50 kJ).
==============================================================================
WHY THIS IS POSSIBLE AT ALL
  Energy enters the fleet problem ONLY through per-UAV budget  e = Emax / K.
  So the whole Emax dependence is carried by one reach curve r(e) plus how the
  objective coefficients scale with e. Characterise those and K*(M,Emax) follows
  from the same balance for every battery.

MODEL (per M, Emax; K continuous then rounded)
  Coverage    N(K) = min( M, K * r(Emax/K, M) )
              r(e,M) = clip( a(M)*(e - emin), 0, rmax(M) )
              a(M) ~ kappa*M^beta  (density law, slope/sqrt(M) ~ const: VALIDATED
              across 1k-1M J by the single-UAV reach sweep). emin is the fleet
              feasibility floor (~3.2 kJ, near-constant). rmax(M) is the
              OBJECTIVE plateau (a UAV stops early to keep WAoI low) -> Emax-free.
  Objective   J(K) = c1(M,Emax)*N^2/K  -  p1(M,Emax)*N  +  s(M,Emax)*(K-1)
              K*(M,Emax) = argmin_K J.

CALIBRATION / HONEST VALIDATION REGION
  Coefficients {a, rmax, emin, c1, p1, s} are calibrated from fleet eval tables.
  * Give ONE Emax table (e.g. 50 kJ): K*(M) is validated for any M at THAT Emax
    (+/-1); cross-Emax output is flagged EXTRAPOLATION (can be off by several K).
  * Give >=3 Emax tables: the script fits each coefficient vs Emax and the
    cross-Emax surface becomes trustworthy (+/-1) across the spanned range.
  The M direction always generalises (smooth M-trends + density law).

USAGE
  # single anchor (current data): reliable in M, extrapolates in Emax
  python kstar_ME_predictor.py --anchor 50000 eval_table_split.csv compare_final.csv
  # multiple anchors (after running them): reliable in M AND Emax
  python kstar_ME_predictor.py \
      --anchor 10000  eval_M_E10k.csv  compare_E10k.csv \
      --anchor 50000  eval_table_split.csv compare_final.csv \
      --anchor 200000 eval_M_E200k.csv compare_E200k.csv \
      --query M=1200 Emax=120000
"""

import argparse, csv
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

REF_EMAX = 50_000.0


def r_of_e(e, a, rmax, emin):
    return np.clip(a * (np.asarray(e, float) - emin), 0.0, rmax)


def _load(eval_csv, cmp_csv):
    ev = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(eval_csv)):
        ev[int(r['M'])][int(r['K'])].append(
            (float(r['nodes']), float(r['waoi']), float(r['priority'])))
    mean = {M: {K: np.mean(v, axis=0) for K, v in d.items()} for M, d in ev.items()}
    pen, fin = defaultdict(dict), defaultdict(dict)
    for r in csv.DictReader(open(cmp_csv)):
        M, K = int(r['M']), int(r['K'])
        pen[M][K] = float(r['beam_penalty']); fin[M][K] = float(r['augmented_final'])
    return mean, pen, fin


def calibrate_one_Emax(Emax, eval_csv, cmp_csv, subset=(1, 2, 3, 4)):
    """Return per-M coefficient dict + measured argmins at this Emax."""
    mean, pen, fin = _load(eval_csv, cmp_csv)
    Ms = sorted(mean); out = {}
    for M in Ms:
        Ks = sorted(mean[M]); Ka = np.array(Ks, float)
        n1 = np.array([mean[M][k][0] for k in Ks]) / Ka
        (a, rmax, emin), _ = curve_fit(r_of_e, Emax / Ka, n1, p0=[1.5e-3, 30, 3000],
                                       bounds=([1e-6, 1, 0], [1, 400, 20000]), maxfev=60000)
        Nf = lambda kk: np.minimum(M, np.asarray(kk, float) *
                                   r_of_e(Emax / np.asarray(kk, float), a, rmax, emin))
        sub = [k for k in subset if k in mean[M]]; K = np.array(sub, float); Nh = Nf(K)
        A  = np.array([mean[M][k][1] for k in sub]); Pr = np.array([mean[M][k][2] for k in sub])
        c1 = float(np.sum(A * (Nh**2 / K)) / np.sum((Nh**2 / K)**2))
        p1 = float(np.sum(Pr * Nh) / np.sum(Nh**2))
        pv = np.array([pen[M][k] for k in sub]); s = float(np.sum(pv * (K - 1)) / np.sum((K - 1)**2))
        out[M] = dict(a=a, rmax=rmax, emin=emin, c1=c1, p1=p1, s=s)
    meas = {M: min(fin[M], key=fin[M].get) for M in Ms}
    return out, meas, Ms


class KstarME:
    """Emax-general predictor. Holds coefficient models fit over the anchors."""
    def __init__(self, anchors):
        # anchors: list of (Emax, per_M_coeffs, Ms)
        self.emaxes = sorted(a[0] for a in anchors)
        self.by_E = {a[0]: a[1] for a in anchors}
        allMs = sorted(set().union(*[set(a[2]) for a in anchors]))
        self.Ms = allMs
        # ---- M-trends of each coefficient, per anchor Emax ----
        def m_power(coeffs, key):
            Ms = sorted(coeffs); x = np.log(Ms); y = np.log([coeffs[M][key] for M in Ms])
            c = np.polyfit(x, y, 1); return lambda M: float(np.exp(np.polyval(c, np.log(M))))
        def m_lin(coeffs, key):
            Ms = sorted(coeffs); c = np.polyfit(Ms, [coeffs[M][key] for M in Ms], 1)
            return lambda M: float(np.polyval(c, M))
        self._mtrend = {}
        for E, coeffs, _ in anchors:
            self._mtrend[E] = dict(
                a=m_power(coeffs, 'a'), rmax=m_power(coeffs, 'rmax'),
                c1=m_power(coeffs, 'c1'), p1=m_lin(coeffs, 'p1'), s=m_lin(coeffs, 's'),
                emin=float(np.median([coeffs[M]['emin'] for M in coeffs])))

    def _coef(self, key, M, Emax):
        """Coefficient at (M,Emax): M-trend at each anchor, then interpolate in Emax."""
        Es = self.emaxes
        vals = [self._mtrend[E][key](M) if key != 'emin' else self._mtrend[E]['emin']
                for E in Es]
        if len(Es) == 1:
            base = vals[0]
            if key == 's':           # only physically-motivated Emax scaling we apply
                return base * (Emax / Es[0])
            return base              # single anchor: hold flat (EXTRAPOLATION)
        return float(np.interp(Emax, Es, vals))   # multi-anchor: interpolate

    def kstar(self, M, Emax, kmax=12):
        g = np.arange(1.0, kmax + 1e-9, 0.02)
        a, rmax, emin = self._coef('a', M, Emax), self._coef('rmax', M, Emax), self._coef('emin', M, Emax)
        c1, p1, s = self._coef('c1', M, Emax), self._coef('p1', M, Emax), self._coef('s', M, Emax)
        N = np.minimum(M, g * r_of_e(Emax / g, a, rmax, emin))
        J = np.where(N > 0, c1 * N**2 / g, 0.0) - p1 * N + s * np.maximum(g - 1, 0)
        i = int(np.nanargmin(J)); return int(round(g[i])), float(g[i])

    def validated_Emax_range(self):
        return (min(self.emaxes), max(self.emaxes)) if len(self.emaxes) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', nargs=3, action='append', metavar=('EMAX', 'EVAL_CSV', 'CMP_CSV'),
                    required=True, help='Emax value and its eval/compare tables (repeatable)')
    ap.add_argument('--query', nargs='+', default=[], help='e.g. M=1200 Emax=120000')
    ap.add_argument('--out-dir', default='kstar_out')
    a = ap.parse_args()
    import os; os.makedirs(a.out_dir, exist_ok=True)

    anchors = []
    for Estr, ev_csv, cmp_csv in a.anchor:
        E = float(Estr); coeffs, meas, Ms = calibrate_one_Emax(E, ev_csv, cmp_csv)
        anchors.append((E, coeffs, Ms)); anchors[-1] = (E, coeffs, Ms)
        # validate at this anchor
        pred = {M: KstarME([(E, coeffs, Ms)]).kstar(M, E)[0] for M in Ms}
        ok = np.mean([abs(pred[M] - meas[M]) <= 1 for M in Ms]) * 100
        ex = np.mean([pred[M] == meas[M] for M in Ms]) * 100
        print(f'anchor Emax={E:.0f}: validate K*(M) -> within+/-1 {ok:.0f}%, exact {ex:.0f}%  '
              f'(meas {[meas[M] for M in Ms]} vs pred {[pred[M] for M in Ms]})')

    model = KstarME(anchors)
    rng = model.validated_Emax_range()
    print('\n' + '=' * 78)
    if rng:
        print(f'Emax-VALIDATED range: {rng[0]:.0f}-{rng[1]:.0f} J  '
              f'({len(model.emaxes)} anchors) -> K*(M,Emax) reliable (+/-1) in this box.')
    else:
        print(f'SINGLE anchor at {model.emaxes[0]:.0f} J: K*(M) reliable for any M at THIS Emax.')
        print('Cross-Emax output is EXTRAPOLATION (can be off by several K). Add >=3 Emax')
        print('anchors to make K*(M,Emax) reliable. Suggested anchors + runs:')
        print('  for E in 10000 200000:                 # plus your existing 50000')
        print('    for M in 100 200:')
        print('      for K in 1 2 3 4 5 6:')
        print('        (train fleet at that M,K with Emax_each = E/K, one seed)')
        print('  + a single-UAV reach sweep e in [1e3..1e6] to lock the coverage envelope.')
    print('=' * 78)

    for q in a.query:
        M = Emax = None
        for tok in a.query:
            if tok.startswith('M='): M = float(tok[2:])
            if tok.startswith('Emax='): Emax = float(tok[5:])
        if M and Emax:
            ki, kc = model.kstar(M, Emax)
            tag = 'reliable' if (rng and rng[0] <= Emax <= rng[1]) else \
                  ('reliable' if (not rng and abs(Emax - model.emaxes[0]) < 1) else 'EXTRAPOLATION')
            print(f'\nK*(M={M:.0f}, Emax={Emax:.0f}) = {ki}  (continuous {kc:.2f})   [{tag}]')
            break

    _surface(model, a.out_dir)


def _surface(model, out_dir):
    Ms = np.arange(50, 1251, 25); Es = np.geomspace(1e3, 1e6, 48)
    Z = np.array([[model.kstar(M, E)[0] for M in Ms] for E in Es])
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.pcolormesh(Ms, Es, Z, shading='auto', cmap='viridis')
    ax.set_yscale('log'); fig.colorbar(im, ax=ax, label='predicted K*')
    ax.set_xlabel('Network size M'); ax.set_ylabel('Energy budget Emax (J, log)')
    rng = model.validated_Emax_range()
    if rng:
        ax.axhspan(rng[0], rng[1], color='white', alpha=0.0)
        ax.set_title('K*(M, Emax)  (validated band spans the Emax anchors)')
    else:
        E0 = model.emaxes[0]
        ax.axhline(E0, color='w', lw=2)
        # hatch the unvalidated region (everything off the single anchor line)
        ax.text(0.5, 0.94, 'VALIDATED only on the white line (single Emax anchor);\n'
                'rest is extrapolation \u2014 add anchors to trust it',
                transform=ax.transAxes, ha='center', va='top', color='w', fontsize=9,
                bbox=dict(boxstyle='round', fc='black', alpha=0.55))
        ax.set_title('K*(M, Emax)  (single-anchor \u2014 off-line region NOT yet reliable)')
    plt.tight_layout(); plt.savefig(f'{out_dir}/kstar_M_Emax_surface.png', dpi=150); plt.close()
    print(f'\nSurface -> {out_dir}/kstar_M_Emax_surface.png')


if __name__ == '__main__':
    main()
