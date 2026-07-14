"""Greedy pipeline vs beam pipeline, per M: routing (solid) and deconflicted FINAL (dashed).
Shows beam lowers the objective at BOTH stages and the deconfliction penalty (routing->FINAL
gap) is essentially the same for both methods. M=200 K=2 beam flagged (only cell beam loses)."""
import json, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

G = {
(50,1):(-44.33,-44.33),(50,2):(-87.08,-85.37),(50,3):(-114.57,-111.00),(50,4):(-128.36,-123.70),(50,5):(-130.10,-124.60),(50,6):(-124.68,-118.60),(50,8):(-98.02,-90.93),
(60,1):(-52.04,-52.04),(60,2):(-92.13,-90.13),(60,3):(-127.33,-123.26),(60,4):(-143.81,-138.59),(60,5):(-144.97,-138.35),(60,6):(-140.66,-133.45),(60,8):(-121.71,-113.56),
(80,1):(-60.31,-60.31),(80,2):(-103.10,-100.04),(80,3):(-147.93,-142.83),(80,4):(-173.67,-166.72),(80,5):(-175.68,-167.59),(80,6):(-169.17,-160.05),(80,8):(-143.11,-133.56),
(100,1):(-68.13,-68.13),(100,2):(-114.44,-111.34),(100,3):(-176.07,-169.74),(100,4):(-204.19,-195.79),(100,5):(-203.84,-193.78),(100,6):(-198.42,-187.21),(100,8):(-168.18,-156.59),
(120,1):(-74.84,-74.84),(120,2):(-129.57,-125.77),(120,3):(-197.05,-190.46),(120,4):(-230.05,-220.62),(120,5):(-236.97,-225.36),(120,6):(-228.36,-215.38),(120,8):(-191.57,-177.89),
(150,1):(-78.03,-78.03),(150,2):(-139.71,-135.49),(150,3):(-230.19,-221.81),(150,4):(-263.20,-252.13),(150,5):(-272.67,-259.33),(150,6):(-259.16,-244.62),(150,8):(-225.15,-208.93),
(200,1):(-83.89,-83.89),(200,2):(-165.16,-160.25),(200,3):(-275.51,-265.12),(200,4):(-315.02,-301.42),(200,5):(-327.81,-311.28),(200,6):(-310.75,-293.15),(200,8):(-273.70,-253.43),
}
B={ (d['M'],d['K']):d for d in (json.loads(l) for l in open('beam_deconf_store.jsonl')) }
Ms=sorted({m for m,_ in G}); KS=[1,2,3,4,5,6,7,8]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'axes.labelsize':9.5,'axes.edgecolor':'#444','axes.linewidth':0.8,
    'figure.facecolor':'white','axes.facecolor':'white'})
CG='#1f4e79'; CB='#2e7d32'
fig,axes=plt.subplots(2,4,figsize=(15.5,8.0)); axes=axes.ravel()
def arr(M,src,idx=None):
    out=[]
    for K in KS:
        if (M,K) not in G: out.append(np.nan); continue
        if src=='gr': out.append(G[(M,K)][idx])
        else: out.append(B[(M,K)]['routing_mean' if idx==0 else 'final_mean'])
    return np.array(out)
for ax,M in zip(axes,Ms):
    x=np.array(KS,float)
    gr_rt=arr(M,'gr',0); gr_fn=arr(M,'gr',1); b_rt=arr(M,'b',0); b_fn=arr(M,'b',1)
    ax.plot(x,gr_rt,'-o',color=CG,lw=1.6,ms=4,zorder=3,label='greedy routing')
    ax.plot(x,gr_fn,'--',color=CG,lw=1.4,zorder=2,label='greedy FINAL')
    ax.plot(x,b_rt,'-o',color=CB,lw=1.6,ms=4,zorder=4,label='beam routing')
    ax.plot(x,b_fn,'--',color=CB,lw=1.4,zorder=3,label='beam FINAL')
    # flag M=200 K=2 beam (only cell beam loses)
    if M==200:
        ax.plot(2,B[(200,2)]['routing_mean'],'o',ms=8,mfc='none',mec='#c0392b',mew=1.8,zorder=6)
        ax.annotate('beam loses here\n(use greedy)',(2,B[(200,2)]['routing_mean']),
                    textcoords='offset points',xytext=(14,-2),fontsize=7,color='#c0392b',fontweight='bold')
    ax.set_title(f'M = {M}',fontweight='bold',pad=6)
    ax.set_xticks([1,2,3,4,5,6,8]); ax.grid(True,color='#e6e6e6',lw=0.7); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    if M in (Ms[0],Ms[4]): ax.set_ylabel('objective (lower better \u2193)')
for ax in axes[4:]: ax.set_xlabel('K (UAVs)')
lax=axes[7]; lax.axis('off')
lax.legend(handles=[
    Line2D([],[],color=CG,lw=1.6,marker='o',label='greedy+pp routing'),
    Line2D([],[],color=CG,lw=1.4,ls='--',label='greedy FINAL (deconflicted)'),
    Line2D([],[],color=CB,lw=1.6,marker='o',label='beam+pp routing'),
    Line2D([],[],color=CB,lw=1.4,ls='--',label='beam FINAL (deconflicted)'),
    Line2D([],[],color='#c0392b',marker='o',ms=8,mfc='none',mew=1.8,lw=0,label='beam regresses (M200 K2)'),
],loc='center',frameon=False,fontsize=9,handlelength=2.2,labelspacing=1.3,borderpad=1.0,
   title='within-color gap = deconfliction penalty\n(near-identical for both methods)',title_fontsize=8.5)
fig.suptitle('Greedy vs beam pipeline: routing \u2192 collision-avoided FINAL, per M  '
             '(common 30-instance set, 3 seeds, \u03b4=25 m)',fontsize=13,fontweight='bold',y=0.98)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig('/mnt/user-data/outputs/fig_pipeline_greedy_vs_beam.png',dpi=150,bbox_inches='tight')
print('wrote fig')
