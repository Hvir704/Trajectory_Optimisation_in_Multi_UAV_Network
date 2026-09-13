"""plot_snapshot_pair.py -- Figure 1: two-panel reach comparison, sized for a single IEEE column.
Strips in-figure titles, footers and per-sensor labels; everything explanatory goes in the caption.
usage: python plot_snapshot_pair.py M Emax K_law K_over seed t0_h out.pdf [layout]
"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
M=int(sys.argv[1]); Emax=float(sys.argv[2]); KA=int(sys.argv[3]); KB=int(sys.argv[4])
seed=int(sys.argv[5]); t0=float(sys.argv[6])*3600; out=sys.argv[7]
layout=sys.argv[8] if len(sys.argv)>8 else "paper"
fig,axes=plt.subplots(1,2,figsize=(7.16,3.7))   # full IEEE text width
info=[]
for ax,K in zip(axes,(KA,KB)):
    p=DynParams(M=M,K=K,Emax=Emax,layout=layout,T_horizon=t0+3*3600,T_burnin=t0)
    sim=DynSim(p,build_sa_planner(iters=1200,seed_base=seed),seed=seed); m=sim.run()
    F=sim.field; pos=F.pos/1000; home=p.home/1000
    picked={}
    for rec in sorted(sim.records,key=lambda r:r.t_launch):
        if rec.t_launch>=t0 and rec.drone not in picked and rec.visited: picked[rec.drone]=rec
    r_reach=p.v*(p.E_usable-p.Ph*p.B_bits/p.R)/(2*p.Pf)/1000
    unreached=np.linalg.norm(F.pos-p.home,axis=1)/1000 > r_reach
    ax.scatter(pos[~unreached,0],pos[~unreached,1],s=6+9*F.wi_base[~unreached],
               facecolor="#d8d8d8",edgecolor="#9a9a9a",lw=.4,zorder=2)
    ax.scatter(pos[unreached,0],pos[unreached,1],s=6+9*F.wi_base[unreached],
               facecolor="none",edgecolor="crimson",lw=1.1,zorder=3)
    cols=plt.cm.tab10(np.arange(10))
    for n,(k,rec) in enumerate(sorted(picked.items())):
        idx=list(rec.visited); pts=np.vstack([home,pos[idx],home]); c=cols[n%10]
        ax.plot(pts[:,0],pts[:,1],"-",color=c,lw=1.3,zorder=4)
        ax.scatter(pos[idx,0],pos[idx,1],s=16,color=c,zorder=5,edgecolor="k",lw=.3)
    ax.scatter(*home,marker="*",s=200,c="gold",edgecolor="k",lw=.6,zorder=6)
    ax.add_patch(plt.Circle(home,r_reach,fill=False,ls="--",color="crimson",lw=1.3,zorder=1))
    ax.set_xlim(-.3,p.L/1000+.3); ax.set_ylim(-.3,p.L/1000+.3); ax.set_aspect("equal")
    ax.set_xticks([0,4,8,12]); ax.set_yticks([0,4,8,12])
    ax.tick_params(labelsize=8); ax.set_xlabel("x (km)",fontsize=9)
    ax.set_title(rf"$K={K}$   $r_{{\rm reach}}={r_reach:.1f}$ km   unserved: {m['n_never_visited']}",fontsize=9.5)
    info.append((K,r_reach,m['n_never_visited'],m['J_timeavg'],len(picked)))
axes[0].set_ylabel("y (km)",fontsize=9)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker="*",color="w",markerfacecolor="gold",markeredgecolor="k",markersize=12,label="depot"),
   Line2D([],[],ls="--",color="crimson",label=r"$r_{\rm reach}(K)$"),
   Line2D([],[],marker="o",color="w",markerfacecolor="none",markeredgecolor="crimson",markersize=7,label="unreachable sensor"),
   Line2D([],[],color=plt.cm.tab10(0),lw=1.4,label="one sortie per UAV")]
fig.legend(handles=h,loc="lower center",ncol=4,fontsize=8.5,frameon=False,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(out)
print("wrote",out); [print(f"  K={k}: r_reach={r:.1f} km, unserved={u}, J={j:.3e}, UAVs shown={n}") for k,r,u,j,n in info]