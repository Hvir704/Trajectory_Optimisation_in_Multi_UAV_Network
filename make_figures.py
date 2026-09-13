"""Figures for main.tex from grid.csv.

usage: python make_figures.py grid.csv [L=12600] -> figs/*.pdf

CHANGED vs rev1 -- three defects, all introduced or exposed once grid.csv held more than one
field size:

  1. `cell()` keyed on (layout, M, Emax) with no L. grid.csv now holds L = 8000, 10000, 12600,
     16000 and 20000, so every J-vs-K curve averaged five different field sizes together. Silent.
  2. The regime plot keyed instances by (layout, M, Emax) then by seed and K, so rows differing
     only in L overwrote each other -- four of every five instances were discarded, silently, and
     which one survived depended on CSV row order.
  3. `Ks[0]` raised IndexError whenever a (layout, M, Emax) cell had no rows at all.

Fig 1 is now drawn at a single L (default 12600, the paper's field size; override on the command
line). The regime plot deliberately keeps every L -- different field sizes give different
r_max/r_c, which is exactly the spread that plot wants -- but treats each L as its own instance
family so nothing is overwritten.
"""
import sys, os, csv, collections, math, statistics as st
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SRC = sys.argv[1]
L_FIG1 = float(sys.argv[2]) if len(sys.argv) > 2 else 12600.0
os.makedirs("figs", exist_ok=True)

rows = list(csv.DictReader(open(SRC)))
for r in rows:
    for k in r:
        try: r[k] = float(r[k])
        except: pass

# planner and divert are filtered explicitly: grid.csv is sa/divert=0 today, but nothing stops a
# variant run from being appended to it later, and pooling planners would be silent.
base = [r for r in rows
        if r['coord'] == 'exclude' and r['replan'] == 'launch'
        and r['Th'] == 43200 and r.get('q', 0) == 0
        and r.get('planner', 'sa') == 'sa' and int(r.get('divert', 0)) == 0]

Ls = sorted({r.get('L', 12600.0) for r in base})
print(f"{len(base)} base rows; L values present: {Ls}")
if L_FIG1 not in Ls:
    print(f"  WARNING: requested L={L_FIG1:g} not in the data; Fig 1 will be empty")

def cell(layout, M, E, L):
    g = collections.defaultdict(list)
    for r in base:
        if (r['layout'], r['M'], r['Emax'], r.get('L', 12600.0)) == (layout, M, E, L):
            g[int(r['K'])].append(r)
    return g

# --- Fig 1: J vs K, three families, at one L ---
fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.5))
drew = 0
for a, lay in zip(ax, ('paper', 'ring', 'core')):
    g = cell(lay, 100, 1.5e6, L_FIG1); Ks = sorted(g)
    if not Ks:                                   # guard: was IndexError on Ks[0]
        print(f"  Fig 1: no rows for layout={lay} M=100 Emax=1.5e6 L={L_FIG1:g} -- panel left blank")
        a.set_title(f"{lay} (no data)", fontsize=9.5); a.set_xticks([]); a.set_yticks([])
        continue
    drew += 1
    J = [st.mean(r['J'] for r in g[K]) / 1e5 for K in Ks]
    Kr = st.mean(r['K_reach_i'] for r in g[Ks[0]]); Kc = st.mean(r['K_commute_i'] for r in g[Ks[0]])
    lo = min(J); a.plot(Ks, J, 'o-', color='k', ms=4, lw=1.2)
    am = Ks[J.index(lo)]; a.plot([am], [lo], 'o', ms=9, mfc='none', mec='k', mew=1.4)
    a.axvline(Kr, ls='--', color='C3', lw=1.2, label=r'$K_{\rm reach}$')
    if Kc < max(Ks) + 1: a.axvline(Kc, ls=':', color='C0', lw=1.4, label=r'$K_{\rm commute}$')
    a.set_title(f"{lay}", fontsize=9.5); a.set_xlabel('$K$', fontsize=9)
    a.set_ylim(0, min(max(J), 4 * lo) * 1.08); a.set_xticks(Ks[::2]); a.tick_params(labelsize=8)
    a.legend(fontsize=8, frameon=False)
    print(f"  Fig 1 {lay}: K={Ks[0]}..{Ks[-1]}, n/cell={len(g[Ks[0]])}, argmin K={am}, "
          f"J*={lo*1e5:.3e}, K_reach={Kr:.2f}, K_commute={Kc:.2f}")
ax[0].set_ylabel(r'$J$  ($\times10^5$)', fontsize=9)
fig.suptitle(f"$L$ = {L_FIG1/1000:g} km", fontsize=8, y=1.0)
fig.tight_layout(); fig.savefig('figs/fig_JvsK.pdf')

# --- Fig 2: regime plot, every L, each as its own instance family ---
fig, a = plt.subplots(figsize=(3.5, 2.9))
pts = collections.defaultdict(list)
families = {(r['layout'], r['M'], r['Emax'], r.get('L', 12600.0)) for r in base}
for (lay, M, E, L) in families:
    g = collections.defaultdict(dict)
    for r in base:
        if (r['layout'], r['M'], r['Emax'], r.get('L', 12600.0)) == (lay, M, E, L):
            g[r['seed']][int(r['K'])] = r
    for s, byK in g.items():
        served = {K: byK[K]['J_age'] if byK[K]['n_never'] == 0 else None for K in byK}
        ks = [K for K in served if served[K] is not None]
        if not ks: continue
        am = min(ks, key=lambda K: served[K]); r0 = byK[min(byK)]
        if not r0['K_reach_i']: continue
        pts[lay].append((r0['rmax_over_rc'], am / r0['K_reach_i'], r0['regime_bnd']))
for lay, c in zip(('paper', 'ring', 'core'), ('C3', 'C0', 'C2')):
    if not pts[lay]:
        print(f"  Fig 2: no usable instances for layout={lay}"); continue
    p = np.array(pts[lay]); a.scatter(p[:, 0], p[:, 1], s=12, color=c, label=lay, alpha=.7)
    print(f"  Fig 2 {lay}: {len(p)} instances")
a.axvline(st.mean(r['regime_bnd'] for r in base), color='k', ls='--', label=r'$1+\sqrt{\bar P/P_f}$')
a.axhline(1, color='gray', lw=.5); a.set_xscale('log')
a.set_xlabel(r'$r_{\max}/r_c$', fontsize=9); a.set_ylabel(r'served argmin $/\,K_{\rm reach}$', fontsize=9)
a.tick_params(labelsize=8); a.legend(fontsize=8, frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig('figs/fig_regime.pdf')
print("wrote figs/fig_JvsK.pdf figs/fig_regime.pdf")