"""plot_routes.py -- fleet snapshot in the style of the static trajectory figures.
usage: python plot_routes.py M Emax K seed t0_h out.pdf [layout=paper]
Shows ONE sortie per UAV (the first launched at or after t0), with sensors drawn as rings
scaled by priority (label = w_i), served sensors filled, the depot, and r_reach(K).
"""
import sys, math, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dyn_env import DynParams, DynSim
from sa_sortie import build_sa_planner
M=int(sys.argv[1]); Emax=float(sys.argv[2]); K=int(sys.argv[3]); seed=int(sys.argv[4]); t0=float(sys.argv[5])*3600
out=sys.argv[6]; layout=sys.argv[7] if len(sys.argv)>7 else "paper"
p=DynParams(M=M,K=K,Emax=Emax,layout=layout,T_horizon=t0+3*3600,T_burnin=t0)
sim=DynSim(p,build_sa_planner(iters=1200,seed_base=seed),seed=seed); m=sim.run()
F=sim.field; pos=F.pos/1000; home=p.home/1000
picked={}
for rec in sorted(sim.records,key=lambda r:r.t_launch):
    if rec.t_launch>=t0 and rec.drone not in picked and rec.visited: picked[rec.drone]=rec
served=set(j for r in picked.values() for j in r.visited)
fig,ax=plt.subplots(figsize=(7.2,7.4))
w=F.wi_base; s=18+22*w
ax.scatter(pos[:,0],pos[:,1],s=s,facecolor="#cfcfcf",edgecolor="#9a9a9a",lw=.6,zorder=2)
for i in range(M): ax.annotate(f"{int(round(w[i]))}",(pos[i,0],pos[i,1]),xytext=(3,3),textcoords="offset points",fontsize=6,color="#555")
cols=plt.cm.Dark2(np.arange(8))
for n,(k,rec) in enumerate(sorted(picked.items())):
    idx=list(rec.visited); pts=np.vstack([home,pos[idx],home]); c=cols[n%8]
    ax.plot(pts[:,0],pts[:,1],"-",color=c,lw=1.6,zorder=3,label=f"UAV {k+1} ({len(idx)} nodes, {(rec.t_land-rec.t_launch)/60:.0f} min)")
    ax.scatter(pos[idx,0],pos[idx,1],s=s[idx],facecolor="none",edgecolor="k",lw=.9,zorder=4)
    ax.scatter(pos[idx,0],pos[idx,1],s=26,color=c,zorder=5)
ax.scatter(*home,marker="*",s=420,c="gold",edgecolor="k",zorder=6)
r_reach=p.v*(p.E_usable-p.Ph*p.B_bits/p.R)/(2*p.Pf)/1000
ax.add_patch(plt.Circle(home,r_reach,fill=False,ls="--",color="crimson",lw=1.2,zorder=1))
Kr=m["K_cov_instance"]
ax.set_xlim(0,p.L/1000); ax.set_ylim(0,p.L/1000); ax.set_aspect("equal"); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
ax.set_title(f"Fleet snapshot  M={M}, K={K} (law: $\\lfloor K_{{reach}}\\rfloor$={math.floor(Kr)}), served={len(served)} of {M}\n"
             f"$J$={m['J_timeavg']:.3e}  ($J_{{age}}$={m['J_age']:.2e}, $J_{{event}}$={m['J_event']:.2e});  never-visited={m['n_never_visited']}",fontsize=11)
ax.legend(fontsize=8,loc="upper right",framealpha=.92)
fig.text(0.5,0.01,f"ring size $\\propto$ priority $w_i$ (label = $w_i$);  $E_{{max}}$={Emax:,.0f} J, {p.E_usable:,.0f} J usable per UAV;  "
         f"dashed: $r_{{reach}}(K)$={r_reach:.1f} km;  seed {seed}, 1200 SA iters, coordinated, one sortie per UAV from t={t0/3600:.0f} h",
         ha="center",fontsize=7.5,color="#444")
fig.tight_layout(rect=(0,0.03,1,1)); fig.savefig(out); print("wrote",out)