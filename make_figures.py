"""Figures for main.tex from grid.csv. usage: python make_figures.py grid.csv  -> figs/*.pdf"""
import sys, csv, collections, math, statistics as st
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in r:
        try: r[k]=float(r[k])
        except: pass
base=[r for r in rows if r['coord']=='exclude' and r['replan']=='launch' and r['Th']==43200 and r.get('q',0)==0]
def cell(layout,M,E):
    g=collections.defaultdict(list)
    for r in base:
        if (r['layout'],r['M'],r['Emax'])==(layout,M,E): g[int(r['K'])].append(r)
    return g
# --- Fig 1: J vs K, three families ---
fig,ax=plt.subplots(1,3,figsize=(7.16,2.5))
for a,lay in zip(ax,('paper','ring','core')):
    g=cell(lay,100,1.5e6); Ks=sorted(g)
    J=[st.mean(r['J'] for r in g[K])/1e5 for K in Ks]; Kr=st.mean(r['K_reach_i'] for r in g[Ks[0]]); Kc=st.mean(r['K_commute_i'] for r in g[Ks[0]])
    lo=min(J); a.plot(Ks,J,'o-',color='k',ms=4,lw=1.2)
    am=Ks[J.index(lo)]; a.plot([am],[lo],'o',ms=9,mfc='none',mec='k',mew=1.4)
    # NOTE: this circles the argmin of MEAN J, a cell-level summary. The paper's agreement
    # statistics use the PER-INSTANCE argmin (see analyze.py); the two differ in 8 of 24 cells.
    # Fig 1 is illustrative of the curve shape; do not quote its circled K as "the optimum".
    a.axvline(Kr,ls='--',color='C3',lw=1.2,label=r'$K_{\rm reach}$')
    if Kc<max(Ks)+1: a.axvline(Kc,ls=':',color='C0',lw=1.4,label=r'$K_{\rm commute}$')
    a.set_title(f"{lay}",fontsize=9.5); a.set_xlabel('$K$',fontsize=9)
    a.set_ylim(0,min(max(J),4*lo)*1.08); a.set_xticks(Ks[::2]); a.tick_params(labelsize=8)
    a.legend(fontsize=8,frameon=False)
ax[0].set_ylabel(r'$J$  ($\times10^5$)',fontsize=9); fig.tight_layout(); fig.savefig('figs/fig_JvsK.pdf')
# --- Fig 2: regime plot ---
fig,a=plt.subplots(figsize=(3.5,2.9))
pts=collections.defaultdict(list)
for (lay,M,E) in {(r['layout'],r['M'],r['Emax']) for r in base}:
    g=collections.defaultdict(dict)
    for r in base:
        if (r['layout'],r['M'],r['Emax'])==(lay,M,E): g[r['seed']][int(r['K'])]=r
    for s,byK in g.items():
        served={K:byK[K]['J_age'] if byK[K]['n_never']==0 else None for K in byK}
        ks=[K for K in served if served[K] is not None]
        if not ks: continue
        am=min(ks,key=lambda K:served[K]); r0=byK[min(byK)]
        pts[lay].append((r0['rmax_over_rc'], am/r0['K_reach_i'], r0['regime_bnd']))
for lay,c in zip(('paper','ring','core'),('C3','C0','C2')):
    p=np.array(pts[lay]); a.scatter(p[:,0],p[:,1],s=12,color=c,label=lay,alpha=.7)
a.axvline(st.mean(r['regime_bnd'] for r in base),color='k',ls='--',label=r'$1+\sqrt{\bar P/P_f}$')
a.axhline(1,color='gray',lw=.5); a.set_xscale('log'); a.set_xlabel(r'$r_{\max}/r_c$',fontsize=9); a.set_ylabel(r'served argmin $/\,K_{\rm reach}$',fontsize=9)
a.tick_params(labelsize=8); a.legend(fontsize=8,frameon=False,loc='lower right')
fig.tight_layout(); fig.savefig('figs/fig_regime.pdf')
print("wrote figs/fig_JvsK.pdf figs/fig_regime.pdf")