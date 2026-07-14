"""
report_mlp.py  —  Report figures for MLP-REINFORCE
====================================================
Run from the folder containing uav_aoi_solver.py

Reads from:
  results/history_M30.npy
  models/policy_M30.pt

Saves 9 figures to:
  report_figs_mlp/

Usage:
  python report_mlp.py
  python report_mlp.py --n_eval 500
  python report_mlp.py --skip_reeval   # reuse cached data (instant re-run)
"""

import os, sys, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sc
warnings.filterwarnings('ignore')

os.makedirs('report_figs_mlp', exist_ok=True)

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    'Random':           '#AAAAAA',
    'Nearest-Neighbor': '#E69F00',
    'Greedy-Priority':  '#56B4E9',
    'PDR':              '#CC79A7',
    'MLP-REINFORCE':    '#009E73',
}
MK = {'Random':'s','Nearest-Neighbor':'^','Greedy-Priority':'D',
      'PDR':'v','MLP-REINFORCE':'o'}
BASELINES = ['Random','Nearest-Neighbor','Greedy-Priority','PDR']
ALL = BASELINES + ['MLP-REINFORCE']

def smooth(x, k=30): return np.convolve(x, np.ones(min(k,len(x)))/min(k,len(x)), 'same')
def ci95(a): return 1.96*np.std(a)/np.sqrt(len(a))
def pstar(p):
    if p<0.001: return '***'
    if p<0.01:  return '**'
    if p<0.05:  return '*'
    return 'ns'


# ═════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
def collect(n, seed, device):
    import torch, importlib.util
    spec = importlib.util.spec_from_file_location('sol', 'uav_aoi_solver.py')
    sol  = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)

    policy = sol.Policy(hidden=256).to(device)
    ck     = torch.load('models/policy_M30.pt', map_location=device)
    policy.load_state_dict(ck['policy']); policy.eval()
    print(f'  Loaded models/policy_M30.pt')

    rng  = np.random.default_rng(seed)
    data = {m: {'obj':[],'waoi':[],'priority':[],'nodes':[],'reward':[]}
            for m in ALL}

    for i in range(n):
        s = int(rng.integers(0, 10_000_000))
        for bname in BASELINES:
            env  = sol.Env(M=30, seed=s)
            traj = sol.run_baseline(env, sol.BASELINES[bname])
            data[bname]['obj'].append(env.objective(traj))
            data[bname]['waoi'].append(sol.P.theta1 * env.waoi(traj))
            data[bname]['priority'].append(float(sum(env.wi[j] for j in traj)))
            data[bname]['nodes'].append(len(traj))
            data[bname]['reward'].append(env.reward(traj))
        with torch.no_grad():
            env2  = sol.Env(M=30, seed=s)
            traj2, *_ = sol.rollout(policy, env2, device, greedy=True)
        data['MLP-REINFORCE']['obj'].append(env2.objective(traj2))
        data['MLP-REINFORCE']['waoi'].append(sol.P.theta1 * env2.waoi(traj2))
        data['MLP-REINFORCE']['priority'].append(float(sum(env2.wi[j] for j in traj2)))
        data['MLP-REINFORCE']['nodes'].append(len(traj2))
        data['MLP-REINFORCE']['reward'].append(env2.reward(traj2))
        if (i+1)%50==0: print(f'  {i+1}/{n}')
    return data


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
def gm(data, m, k): return float(np.mean(data[m][k]))
def gv(data, m, k): return np.array(data[m][k])
OUT = 'report_figs_mlp'
MODEL = 'MLP-REINFORCE'
MC    = C[MODEL]

# ── Fig 1: Training convergence ───────────────────────────────────────────────
def fig1_convergence(history):
    if history is None: return
    print('  Fig 1: convergence'); fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    fig.suptitle('MLP-REINFORCE — Training Convergence (M=30)',fontsize=13,fontweight='bold')
    for ax,(key,lbl,ref) in zip(axes,[
        ('reward','Episode Reward',None),
        ('nodes','Nodes Visited',None),
        ('entropy','Policy Entropy H',None)]):
        if key not in history: continue
        y=np.array(history[key]); ep=np.arange(1,len(y)+1)
        ax.plot(ep,y,color=MC,alpha=0.2,lw=0.7)
        ax.plot(ep,smooth(y),color=MC,lw=2.3,label='MLP-REINFORCE')
        if key=='nodes':
            ax.axhline(13.7,color=C['Random'],ls=':',lw=1.2,label='Random (13.7)')
            ax.axhline(13.6,color=C['Greedy-Priority'],ls=':',lw=1.2,label='Greedy-Priority (13.6)')
        ax.set_xlabel('Epoch',fontsize=10); ax.set_ylabel(lbl,fontsize=10)
        ax.set_title(lbl,fontsize=11); ax.grid(alpha=0.3)
        if key=='nodes': ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig1_convergence.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 2: Performance bars ───────────────────────────────────────────────────
def fig2_bars(data, n):
    print('  Fig 2: performance bars'); fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    fig.suptitle(f'MLP-REINFORCE vs Baselines — M=30, {n} instances',fontsize=13,fontweight='bold')
    for ax,(key,title,ylabel) in zip(axes,[
        ('waoi','(a) Weighted AoI  ↓ lower=better','θ₁·WAoI'),
        ('priority','(b) Priority Collected  ↑ higher=better','Σwᵢ'),
        ('obj','(c) Composite Objective  ↓ lower=better','θ₁WAoI−θ₂Σwᵢ')]):
        vals=[gm(data,m,key) for m in ALL]
        errs=[ci95(gv(data,m,key)) for m in ALL]
        x=np.arange(len(ALL)); cols=[C[m] for m in ALL]
        bars=ax.bar(x,vals,yerr=errs,capsize=4,color=cols,edgecolor='black',lw=0.7,
                    error_kw=dict(elinewidth=1.5,ecolor='#444'))
        bars[-1].set_linewidth(2.5)
        ax.set_title(title,fontsize=10); ax.set_xticks(x)
        ax.set_xticklabels([m.replace('-','-\n') if len(m)>14 else m for m in ALL],
                           rotation=30,ha='right',fontsize=9)
        ax.set_ylabel(ylabel,fontsize=10); ax.axhline(0,color='black',lw=0.5)
        ax.grid(axis='y',alpha=0.3)
        for bar,v,e in zip(bars,vals,errs):
            rng_span=abs(max(vals)-min(vals))
            ax.text(bar.get_x()+bar.get_width()/2,v+(e+rng_span*0.015)*(1 if v>=0 else -3),
                    f'{v:.1f}',ha='center',fontsize=8,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig2_performance_bars.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 3: Box plots ──────────────────────────────────────────────────────────
def fig3_boxplots(data):
    print('  Fig 3: box plots'); fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    fig.suptitle('MLP-REINFORCE — Result Distribution (whiskers=5th/95th pct)',fontsize=13,fontweight='bold')
    for ax,(key,ylabel,title) in zip(axes,[
        ('obj','Composite Objective','(a) Objective  ↓ lower=better'),
        ('nodes','Nodes Visited','(b) Nodes Visited per Episode')]):
        box_data=[gv(data,m,key) for m in ALL]
        bp=ax.boxplot(box_data,patch_artist=True,notch=False,whis=[5,95],
                      medianprops=dict(color='black',lw=2.5),
                      whiskerprops=dict(lw=1.4),capprops=dict(lw=1.4),
                      flierprops=dict(marker='o',ms=3,alpha=0.3))
        for patch,m in zip(bp['boxes'],ALL): patch.set_facecolor(C[m]); patch.set_alpha(0.8)
        means=[float(np.mean(gv(data,m,key))) for m in ALL]
        ax.scatter(range(1,len(ALL)+1),means,marker='D',color='white',edgecolors='black',s=60,zorder=5,label='Mean')
        ax.set_xticks(range(1,len(ALL)+1)); ax.set_xticklabels(ALL,rotation=20,ha='right',fontsize=10)
        ax.set_ylabel(ylabel,fontsize=11); ax.set_title(title,fontsize=11)
        ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig3_boxplots.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 4: WAoI reduction ─────────────────────────────────────────────────────
def fig4_waoi_reduction(data):
    print('  Fig 4: WAoI reduction'); fig,ax=plt.subplots(figsize=(9,5))
    ref=gm(data,'Random','waoi')
    methods=BASELINES[1:]+[MODEL]; reds=[100*(1-gm(data,m,'waoi')/ref) for m in methods]
    cols=[C[m] for m in methods]
    bars=ax.bar(range(len(methods)),reds,color=cols,edgecolor='black',lw=0.7)
    bars[-1].set_linewidth(2.5)
    ax.axhline(66,color='red',ls='--',lw=2,label='Paper D3QN claims >66%')
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods,rotation=25,ha='right',fontsize=10)
    ax.set_ylabel('WAoI Reduction vs Random (%)',fontsize=11)
    ax.set_title('MLP-REINFORCE — WAoI Reduction\nvs All Baselines (M=30)',fontsize=12,fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y',alpha=0.3)
    for bar,v in zip(bars,reds):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.5,f'{v:.1f}%',ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig4_waoi_reduction.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 5: Statistical significance ──────────────────────────────────────────
def fig5_significance(data):
    print('  Fig 5: significance'); fig,axes=plt.subplots(1,2,figsize=(13,5))
    fig.suptitle('MLP-REINFORCE — Statistical Evidence vs Every Baseline',fontsize=13,fontweight='bold')
    our=gv(data,MODEL,'obj'); t_s,p_v,eff_d=[],[],[]
    for b in BASELINES:
        b_arr=gv(data,b,'obj'); t,p=sc.ttest_ind(our,b_arr); t_s.append(t); p_v.append(p)
        pool=np.sqrt((our.std()**2+b_arr.std()**2)/2)
        eff_d.append((our.mean()-b_arr.mean())/(pool+1e-9))
    y=np.arange(len(BASELINES)); cols=[C[b] for b in BASELINES]
    # p-value
    ax=axes[0]
    ax.barh(y,[-np.log10(p+1e-300) for p in p_v],color=cols,edgecolor='black',lw=0.7,alpha=0.85)
    ax.axvline(-np.log10(0.05), color='orange',ls='--',lw=1.5,label='p=0.05')
    ax.axvline(-np.log10(0.001),color='red',   ls='--',lw=1.5,label='p=0.001')
    ax.set_yticks(y); ax.set_yticklabels([f'{b}  {pstar(p)}' for b,p in zip(BASELINES,p_v)],fontsize=10)
    ax.set_xlabel('−log₁₀(p-value)',fontsize=10); ax.set_title("(a) Welch's t-test",fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis='x',alpha=0.3)
    # Cohen's d
    ax2=axes[1]
    ax2.barh(y,eff_d,color=cols,edgecolor='black',lw=0.7,alpha=0.85)
    ax2.axvline(0,   color='black', lw=0.8)
    ax2.axvline(-0.5,color='orange',ls='--',lw=1.2,label='Medium |d|=0.5')
    ax2.axvline(-0.8,color='red',   ls='--',lw=1.2,label='Large  |d|=0.8')
    ax2.set_yticks(y); ax2.set_yticklabels(BASELINES,fontsize=10)
    ax2.set_xlabel("Cohen's d  (negative = MLP lower/better objective)",fontsize=10)
    ax2.set_title("(b) Effect Size",fontsize=11); ax2.legend(fontsize=9); ax2.grid(axis='x',alpha=0.3)
    for yi,d in zip(y,eff_d): ax2.text(d-0.05,yi,f'{d:.2f}',va='center',ha='right',fontsize=9)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig5_significance.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 6: Pareto scatter ─────────────────────────────────────────────────────
def fig6_pareto(data):
    print('  Fig 6: pareto scatter'); fig,ax=plt.subplots(figsize=(9,7))
    ax.set_title('MLP-REINFORCE — WAoI vs Priority Trade-off\nIdeal corner: bottom-right',fontsize=12,fontweight='bold')
    for m in ALL:
        xs=gv(data,m,'waoi'); ys=gv(data,m,'priority')
        ax.scatter(xs,ys,color=C[m],alpha=0.15,s=14,marker=MK[m])
        lw=2.5 if m==MODEL else 1.2; sz=200 if m==MODEL else 120
        ax.scatter(xs.mean(),ys.mean(),color=C[m],s=sz,marker=MK[m],
                   edgecolors='black',lw=lw,zorder=6,
                   label=f'{m}  (W={xs.mean():.1f}, P={ys.mean():.1f})')
    ax.annotate('',xy=(0.10,0.90),xytext=(0.22,0.76),xycoords='axes fraction',
                arrowprops=dict(arrowstyle='->',color='green',lw=2.5))
    ax.text(0.08,0.92,'Ideal',color='green',fontsize=11,transform=ax.transAxes,fontweight='bold')
    ax.set_xlabel('Weighted AoI (θ₁·WAoI)  ← lower better',fontsize=11)
    ax.set_ylabel('Priority Collected  higher better →',fontsize=11)
    ax.legend(fontsize=9,loc='upper right'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig6_pareto.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 7: Improvement over Random ───────────────────────────────────────────
def fig7_improvement(data):
    print('  Fig 7: improvement'); fig,ax=plt.subplots(figsize=(9,5))
    ref=float(gv(data,'Random','obj').mean())
    methods=BASELINES[1:]+[MODEL]; improv=[]; ci_err=[]; cols=[]
    for m in methods:
        arr=gv(data,m,'obj'); improv.append(ref-arr.mean()); ci_err.append(ci95(arr)); cols.append(C[m])
    y=np.arange(len(methods))
    bars=ax.barh(y,improv,xerr=ci_err,capsize=4,color=cols,edgecolor='black',lw=0.7,
                 error_kw=dict(elinewidth=1.5,ecolor='#444'))
    bars[-1].set_linewidth(2.5)
    ax.axvline(0,color='black',lw=1); ax.set_yticks(y); ax.set_yticklabels(methods,fontsize=11)
    ax.set_xlabel('Objective Improvement over Random\n(positive = better than Random)',fontsize=10)
    ax.set_title('MLP-REINFORCE — How Much Better Than Random?',fontsize=12,fontweight='bold')
    ax.grid(axis='x',alpha=0.3)
    for bar,v,e in zip(bars,improv,ci_err):
        ax.text(max(v,0)+e+0.3,bar.get_y()+bar.get_height()/2,f'{v:+.1f}',va='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig7_improvement.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 8: Selection quality ──────────────────────────────────────────────────
def fig8_quality(data):
    print('  Fig 8: selection quality'); fig,ax=plt.subplots(figsize=(9,5))
    ppn={}
    for m in ALL:
        p=gv(data,m,'priority'); n=np.maximum(gv(data,m,'nodes'),1); ppn[m]=p/n
    vals=[ppn[m].mean() for m in ALL]; errs=[ci95(ppn[m]) for m in ALL]; cols=[C[m] for m in ALL]
    bars=ax.bar(range(len(ALL)),vals,yerr=errs,capsize=4,color=cols,edgecolor='black',lw=0.7,
                error_kw=dict(elinewidth=1.5,ecolor='#444'))
    bars[-1].set_linewidth(2.5)
    ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL,rotation=20,ha='right',fontsize=10)
    ax.set_ylabel('Mean Priority per Node Visited',fontsize=11)
    ax.set_title('MLP-REINFORCE — Node Selection Quality\nHigher = picks more important nodes',fontsize=12,fontweight='bold')
    ax.grid(axis='y',alpha=0.3)
    for bar,v in zip(bars,vals): ax.text(bar.get_x()+bar.get_width()/2,v+0.02,f'{v:.2f}',ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig8_selection_quality.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 9: Summary table ──────────────────────────────────────────────────────
def fig9_table(data, n):
    print('  Fig 9: summary table'); rows=[]; row_clr=[]
    ref_waoi=gm(data,'Random','waoi')
    for m in ALL:
        obj_a=gv(data,m,'obj'); waoi_a=gv(data,m,'waoi')
        red=100*(1-waoi_a.mean()/ref_waoi)
        sig='—' if m in BASELINES else pstar(sc.ttest_ind(obj_a,gv(data,'Random','obj'))[1])
        typ='★ Ours' if m==MODEL else 'Baseline'
        rows.append([m,
                     f'{waoi_a.mean():.1f}±{waoi_a.std():.1f}',
                     f'{gm(data,m,"priority"):.1f}',
                     f'{obj_a.mean():.1f}±{obj_a.std():.1f}',
                     f'{gm(data,m,"nodes"):.1f}',
                     f'{red:.1f}%', sig, typ])
        row_clr.append('#D5F5E3' if m==MODEL else 'white')
    cols=['Method','WAoI (θ₁·W)','Priority','Objective','Nodes','WAoI Red.%','vs Rand','Type']
    fig,ax=plt.subplots(figsize=(16,4))
    ax.axis('off')
    tbl=ax.table(cellText=rows,colLabels=cols,cellLoc='center',loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.9)
    for j in range(len(cols)):
        tbl[(0,j)].set_facecolor('#1A5276'); tbl[(0,j)].set_text_props(color='white',fontweight='bold')
    for i,clr in enumerate(row_clr):
        for j in range(len(cols)): tbl[(i+1,j)].set_facecolor(clr)
    ax.set_title(f'MLP-REINFORCE — Results Summary  (M=30, {n} instances)\n*** p<0.001  ** p<0.01  * p<0.05',
                 fontsize=11,fontweight='bold',pad=12)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig9_summary_table.png',dpi=150,bbox_inches='tight'); plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n_eval',type=int,default=200)
    ap.add_argument('--skip_reeval',action='store_true')
    ap.add_argument('--device',default='')
    args=ap.parse_args()
    import torch
    device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print('='*55); print('  MLP-REINFORCE Report Analysis')
    print(f'  device={device}  n_eval={args.n_eval}'); print('='*55)

    CACHE='report_figs_mlp/raw_data.npy'
    if args.skip_reeval and os.path.exists(CACHE):
        data=np.load(CACHE,allow_pickle=True).item(); print(f'Loaded cache: {CACHE}')
    else:
        print(f'\nRunning {args.n_eval} evaluation instances...')
        data=collect(args.n_eval,42,device); np.save(CACHE,data)

    history=None
    if os.path.exists('results/history_M30.npy'):
        history=np.load('results/history_M30.npy',allow_pickle=True).item()
        print('Loaded results/history_M30.npy')
    else:
        print('No history file found — skipping Fig 1')

    print('\nGenerating figures...')
    fig1_convergence(history)
    fig2_bars(data,args.n_eval)
    fig3_boxplots(data)
    fig4_waoi_reduction(data)
    fig5_significance(data)
    fig6_pareto(data)
    fig7_improvement(data)
    fig8_quality(data)
    fig9_table(data,args.n_eval)

    print('\n'+'='*55)
    r=gm(data,MODEL,'waoi'); ref=gm(data,'Random','waoi')
    print(f'  WAoI       = {r:.1f}  (Random={ref:.1f})')
    print(f'  Objective  = {gm(data,MODEL,"obj"):.1f}  (Random={gm(data,"Random","obj"):.1f})')
    print(f'  Nodes      = {gm(data,MODEL,"nodes"):.1f}')
    print(f'  WAoI red.  = {100*(1-r/ref):.1f}%  (paper claims >66%)')
    print(f'\n  9 figures saved to report_figs_mlp/')
    print('='*55)
