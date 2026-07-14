"""
compare_baseline.py  —  STRONG competing baseline for the fleet objective.
==========================================================================
A simulated-annealing / large-neighbourhood search that optimises the EXACT
stage-weighted fleet objective  J = theta1*WAoI - theta2*priority  under the
per-UAV split-battery budget (Emax/K) and the partition constraint. Unlike the
greedy heuristics, this is a genuinely strong, policy-free competitor: it is the
fair "is your RL near-optimal?" reviewer baseline.

It reports, per (M,K) cell, the SA objective vs the RL policy objective read from
eval_table_split.csv, on the SAME shared-seed (2025) instances.

FAIRNESS NOTE: SA is given `--iters` local-search steps; the RL policy uses a
single construction + post-process. This is an ACHIEVABILITY probe (how good a
feasible solution exists), not yet an equal-wall-clock comparison. Use --iters to
control budget; report runtime alongside.

Run:
    python compare_baseline.py --M 50 100 200 --K 1 2 3 4 5 6 --iters 4000 --instances 30 \
        --eval-table eval_table_split.csv --out compare_baseline.csv
"""
import argparse, csv, time
from collections import defaultdict
import numpy as np

AREA,V,PH,PF,EMAX=1000.,20.,200.,150.,50000.
Ps,k0,s2,Wbw=0.1,1e-3,1e-14,1e6; Dlo,Dhi,wlo,whi=.5e6,5e6,1.,10.
TH1,TH2=.01,1.; R=Wbw*np.log2(1+k0*Ps/(100.**2*s2)); HOME=np.array([500.,500.])
INSTANCE_SEED=2025

def gen(M,seed):
    rng=np.random.default_rng(seed)
    return rng.uniform(0,AREA,(M,2)),rng.uniform(wlo,whi,M),rng.uniform(Dlo,Dhi,M)/R
def tf(a,b): return float(np.linalg.norm(a-b))/V
def chain_waoi(t,pos,wi,tcd):
    W=0.;val=0.
    for k,j in enumerate(t):
        W+=wi[j]; nxt=pos[t[k+1]] if k<len(t)-1 else HOME
        val+=W*(tcd[j]+tf(pos[j],nxt))
    return val
def chain_energy(t,pos,tcd):
    E=0.;prev=HOME
    for j in t: E+=PF*tf(prev,pos[j])+PH*tcd[j]; prev=pos[j]
    return E+PF*tf(prev,HOME)
def fleet_obj(trajs,pos,wi,tcd):
    return TH1*sum(chain_waoi(t,pos,wi,tcd) for t in trajs)-TH2*sum(wi[j] for t in trajs for j in t)
def feasible(trajs,pos,tcd,Ee):
    return all(chain_energy(t,pos,tcd)<=Ee+1e-6 for t in trajs)

def greedy_init(pos,wi,tcd,K,Ee,M):
    trajs=[[] for _ in range(K)]; served=set(); improved=True
    while improved:
        improved=False; best=None
        for k in range(K):
            for j in range(M):
                if j in served: continue
                for p in range(len(trajs[k])+1):
                    cand=trajs[k][:p]+[j]+trajs[k][p:]
                    if chain_energy(cand,pos,tcd)>Ee: continue
                    g=(TH1*chain_waoi(trajs[k],pos,wi,tcd)-TH2*sum(wi[x] for x in trajs[k]))-\
                      (TH1*chain_waoi(cand,pos,wi,tcd)-TH2*sum(wi[x] for x in cand))
                    if best is None or g>best[0]: best=(g,k,cand,j)
        if best and best[0]>1e-9:
            _,k,cand,j=best; trajs[k]=cand; served.add(j); improved=True
    return trajs,served

def sa(pos,wi,tcd,K,Ee,M,iters,seed):
    rng=np.random.default_rng(seed)
    trajs,served=greedy_init(pos,wi,tcd,K,Ee,M)
    cur=fleet_obj(trajs,pos,wi,tcd); best=cur; best_tr=[t[:] for t in trajs]
    T0,T1=abs(cur)*0.05+1e-3,1e-4
    for it in range(iters):
        T=T0*(T1/T0)**(it/max(iters-1,1))
        nt=[t[:] for t in trajs]; op=rng.integers(0,5)
        uns=[j for j in range(M) if j not in served]
        if op==0 and uns:
            j=int(rng.choice(uns)); k=int(rng.integers(0,K)); p=int(rng.integers(0,len(nt[k])+1)); nt[k]=nt[k][:p]+[j]+nt[k][p:]
        elif op==1 and any(nt):
            k=int(rng.choice([i for i in range(K) if nt[i]])); i=int(rng.integers(0,len(nt[k]))); del nt[k][i]
        elif op==2 and any(nt):
            k=int(rng.choice([i for i in range(K) if nt[i]])); i=int(rng.integers(0,len(nt[k]))); j=nt[k][i]; del nt[k][i]
            k2=int(rng.integers(0,K)); p=int(rng.integers(0,len(nt[k2])+1)); nt[k2]=nt[k2][:p]+[j]+nt[k2][p:]
        elif op==3 and any(len(t)>=2 for t in nt):
            k=int(rng.choice([i for i in range(K) if len(nt[i])>=2])); a,b=sorted(rng.choice(len(nt[k]),2,replace=False)); nt[k][a:b+1]=nt[k][a:b+1][::-1]
        elif op==4 and uns and any(nt):
            k=int(rng.choice([i for i in range(K) if nt[i]])); i=int(rng.integers(0,len(nt[k]))); nt[k][i]=int(rng.choice(uns))
        if not feasible(nt,pos,tcd,Ee): continue
        o=fleet_obj(nt,pos,wi,tcd); d=o-cur
        if d<0 or rng.random()<np.exp(-d/max(T,1e-9)):
            trajs=nt; cur=o; served=set(x for t in nt for x in t)
            if o<best: best=o; best_tr=[t[:] for t in nt]
    return best

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--M',type=int,nargs='+',default=[50,100,200])
    ap.add_argument('--K',type=int,nargs='+',default=[1,2,3,4,5,6])
    ap.add_argument('--iters',type=int,default=4000)
    ap.add_argument('--instances',type=int,default=30)
    ap.add_argument('--eval-table',default='eval_table_split.csv')
    ap.add_argument('--out',default='compare_baseline.csv')
    a=ap.parse_args()
    rl=defaultdict(lambda:defaultdict(list))
    for r in csv.DictReader(open(a.eval_table)):
        rl[int(r['M'])][int(r['K'])].append(float(r['obj']))
    rl={M:{K:float(np.mean(v)) for K,v in d.items()} for M,d in rl.items()}
    rows=[]
    print(f'{"M":>4} {"K":>2} {"RL_obj":>9} {"SA_obj":>9} {"gap%":>6} {"winner":>7} {"s/inst":>7}')
    for M in a.M:
        for K in a.K:
            Ee=EMAX/K; rng=np.random.default_rng(INSTANCE_SEED)
            seeds=[int(rng.integers(0,10_000_000)) for _ in range(a.instances)]
            t0=time.time(); vals=[sa(*gen(M,s),K,Ee,M,a.iters,si) for si,s in enumerate(seeds)]
            sa_obj=float(np.mean(vals)); dt=(time.time()-t0)/a.instances
            rlv=rl.get(M,{}).get(K,float('nan'))
            gap=100*(sa_obj-rlv)/abs(rlv) if rlv==rlv else float('nan')
            win='SA' if sa_obj<rlv-1e-6 else ('RL' if rlv<sa_obj-1e-6 else 'tie')
            rows.append(dict(M=M,K=K,RL_obj=rlv,SA_obj=sa_obj,gap_pct=gap,winner=win,instances=a.instances,iters=a.iters))
            print(f'{M:>4} {K:>2} {rlv:>9.2f} {sa_obj:>9.2f} {gap:>6.1f} {win:>7} {dt:>7.1f}')
    with open(a.out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f'\nWrote {a.out}')

if __name__=='__main__': main()
