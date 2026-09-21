import sys, csv, math, collections, os, numpy as np
CORRECTED = os.environ.get('P6_ALGEBRA','corrected') == 'corrected'
sys.path.insert(0,".")
from dyn_env import DynParams, SensorField
rows=[r for r in csv.DictReader(open(sys.argv[1] if len(sys.argv)>1 else 'grid.csv')) if r["coord"]=="exclude" and r["replan"]=="launch" and float(r.get("Th",0))==43200]
cells=collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows: cells[(r["layout"],int(float(r["M"])),float(r["Emax"]),float(r.get("L",12600)))][int(float(r["seed"]))][int(float(r["K"]))]=r
print(f"{'cell':34}{'argmin':>7}{'old':>7}{'P6':>7}{'old=':>6}{'P6=':>6}{'old±1':>7}{'P6±1':>6}")
tot=collections.Counter()
for key in sorted(cells):
    lay,M,E,L=key; inst=cells[key]
    p0=DynParams(M=M,Emax=E,layout=lay,L=L); U=(1-p0.rho)*E; Tb,T=p0.T_burnin,p0.T_horizon-p0.T_burnin
    A=[];O=[];P=[]
    for s,byK in sorted(inst.items()):
        F=SensorField(DynParams(M=M,Emax=E,layout=lay,L=L), np.random.default_rng(s))
        r=np.linalg.norm(F.pos-p0.home,axis=1); w=F.wi_base; c=2*p0.Pf*r/p0.v
        Ks=sorted(byK); r0=byK[Ks[0]]; rc=float(r0["r_c"]); Pb=float(r0["P_bar"]); tc=2*rc/p0.v
        Phi=lambda K:(K*(U-K*p0.Pf*tc)/(U+K*tc*(Pb-p0.Pf))) if (U+K*tc*(Pb-p0.Pf))>0 else -1
        reach=lambda K: p0.v*(U/K-p0.Ph*p0.B_bits/p0.R)/(2*p0.Pf)
        Kc=U/(tc*(p0.Pf+(p0.Pf*Pb)**.5)); Kr=p0.v*U/(2*p0.Pf*r.max()+p0.v*p0.Ph*p0.B_bits/p0.R)
        old=min(math.floor(Kr),round(Kc))
        Kp=1
        for K in range(1,max(Ks)+1):
            if Phi(K+1)<=0: break
            S=r<=reach(K); Sn=r<=reach(K+1)
            QK=np.sqrt(w[S]*c[S]).sum(); QK1=np.sqrt(w[Sn]*c[Sn]).sum()
            gain=(QK**2/(2*Pb*Phi(K))-QK1**2/(2*Pb*Phi(K+1))) if CORRECTED else QK**2/(2*Pb)*(1/Phi(K)-1/Phi(K+1))
            loss=(Tb+T/2)*w[S&~Sn].sum()
            if gain>loss: Kp=K+1
            else: break
        Kp=min(Kp,round(Kc))                      # commute branch still caps it
        am=min(Ks,key=lambda K: float(byK[K]["J"]))
        A.append(am);O.append(old);P.append(Kp)
    A,O,P=np.array(A),np.array(O),np.array(P)
    lab=f"{lay} M={M} E={E:.0e}"+(f" L={L/1000:.0f}" if L!=12600 else "")
    print(f"{lab:34}{A.mean():7.2f}{O.mean():7.2f}{P.mean():7.2f}{np.mean(A==O):6.2f}{np.mean(A==P):6.2f}{np.mean(abs(A-O)<=1):7.2f}{np.mean(abs(A-P)<=1):6.2f}")
    tot['n']+=len(A); tot['o']+=np.sum(A==O); tot['p']+=np.sum(A==P); tot['o1']+=np.sum(abs(A-O)<=1); tot['p1']+=np.sum(abs(A-P)<=1)
n=tot['n']
print(f"\nOVERALL n={n}:  exact  old {tot['o']/n:.2f}  Prop6 {tot['p']/n:.2f}   within1  old {tot['o1']/n:.2f}  Prop6 {tot['p1']/n:.2f}")