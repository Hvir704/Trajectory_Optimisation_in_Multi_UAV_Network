"""Beam-augmented routing (per-instance portfolio) vs greedy+post-process, per M.
beam-augmented = per-instance best(greedy+pp, beam+pp) = port_pp_mean (d_port<=0 all cells)."""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

rows={ (d['M'],d['K']):d for d in (json.loads(l) for l in open('_beam_store.jsonl')) }
Ms=sorted({m for m,_ in rows}); KS=[1,2,3,4,5,6,7,8]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':9.5,'axes.edgecolor':'#444','axes.linewidth':0.8,
    'figure.facecolor':'white','axes.facecolor':'white'})
C_GRD='#1f4e79'; C_BEAM='#2e7d32'; C_GAP='#93c47d'
fig,axes=plt.subplots(2,4,figsize=(15.5,8.0)); axes=axes.ravel()
def val(M,K,key): return rows[(M,K)][key] if (M,K) in rows else np.nan
for ax,M in zip(axes,Ms):
    x=np.array(KS,float)
    grd=np.array([val(M,K,'greedy_pp_mean') for K in KS])
    aug=np.array([val(M,K,'port_pp_mean') for K in KS])
    astd=np.array([val(M,K,'port_pp_std') for K in KS])
    m=~np.isnan(grd)
    ax.fill_between(x[m],grd[m],aug[m],color=C_GAP,alpha=0.6,lw=0,zorder=1)
    ax.fill_between(x[m],(aug-astd)[m],(aug+astd)[m],color=C_BEAM,alpha=0.12,lw=0,zorder=1)
    ax.plot(x,grd,'-o',color=C_GRD,lw=1.8,ms=4.5,zorder=3)
    ax.plot(x,aug,'-o',color=C_BEAM,lw=1.8,ms=5,zorder=4)
    ax.set_title(f'M = {M}',fontweight='bold',pad=6)
    ax.set_xticks([1,2,3,4,5,6,8]); ax.grid(True,color='#e6e6e6',lw=0.7); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    if M in (Ms[0],Ms[4]): ax.set_ylabel('routing objective (lower better \u2193)')
for ax in axes[4:]: ax.set_xlabel('K (UAVs)')
lax=axes[7]; lax.axis('off')
lax.legend(handles=[
    Line2D([],[],color=C_GRD,lw=1.8,marker='o',label='greedy + post-process\n(current routing headline)'),
    Line2D([],[],color=C_BEAM,lw=1.8,marker='o',label='beam-augmented routing\n= per-instance best(greedy+pp, beam+pp)'),
    Patch(facecolor=C_GAP,alpha=0.6,label='beam gain (guaranteed \u22650)'),
    Patch(facecolor=C_BEAM,alpha=0.12,label='beam-augmented \u00b1 seed std'),
],loc='center',frameon=False,fontsize=8.8,handlelength=1.8,labelspacing=1.2,borderpad=1.0)
fig.suptitle('Beam-augmented routing (per-instance portfolio) vs greedy+post-process  '
             '(common 30-instance set, 3 seeds, beam_width=5)',fontsize=13,fontweight='bold',y=0.98)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('fig_beam_vs_greedy_routing.png', dpi=150, bbox_inches='tight')
print('wrote fig')
