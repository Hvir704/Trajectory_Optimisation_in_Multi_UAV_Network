import csv, collections, numpy as np, pickle, os, sys, math
sys.path.insert(0, ".")
import theory_v3
from theory_v3 import field, surrogate_rule, old_rule
from exante_rc import exante
from run_grid import k_range
if os.path.exists("tour_cache_v3.pkl"): theory_v3.TOUR_CACHE.update(pickle.load(open("tour_cache_v3.pkl","rb")))
def load(path):
    D=collections.defaultdict(dict)
    for r in csv.DictReader(open(path)):
        if float(r.get('Th',43200))!=43200 or float(r.get('L',12600))!=12600: continue
        if r.get('coord','exclude')!='exclude' or r.get('replan','launch')!='launch': continue
        D[(r.get('planner','sa'),r['layout'],int(float(r['M'])),float(r['Emax']),int(float(r['seed'])))][int(float(r['K']))]=(float(r['J']),r.get('K_ref'))
    return D
A=os.environ.get('V3_ASSEMBLED', 'assembled') + '/'
R={}
for f in ['grid_final.csv','rr_tour_win.csv','eval_all_v2.csv','cluster_win.csv','gzz_eval.csv','asghar_eval.csv','liu_all.csv']: R.update(load(A+f))
H=load(os.environ.get('V3_RESULTS', 'results') + '/heldout_v3.csv')
def preds(D, kref_fn):
    P={}
    for k in sorted({k[1:] for k in D if k[0]=='sa'}):
        lay,M,E,s=k; p,F=field(lay,M,E,12600.0,s); r=np.linalg.norm(F.pos-p.home,axis=1)
        rc,Pb,_=exante(p,r,np.full(M,p.lam_bits),F.pos,kref_fn(k)); key=(lay,M,12600.0,s)
        P[k]=dict(old=old_rule(p,r,rc,Pb), v3=surrogate_rule(p,F,rc,Pb,key=key)[0], v33=surrogate_rule(p,F,rc,Pb,key=key,gray="exclude")[0])
    return P
def kref_retro(k):
    c=R[('sa',)+k]; v=next(iter(c.values()))[1]; return int(float(v)) if v not in (None,'') else min(c)
PR=preds(R,kref_retro); PH=preds(H,lambda k: min(k_range(k[0],k[1],k[2],k[3],12600.0)))
pickle.dump(theory_v3.TOUR_CACHE, open("tour_cache_v3.pkl","wb"))
def nr90(x): x=np.sort(x); return x[max(0,math.ceil(.9*len(x))-1)]
def report(D,P,label,planners,seeds=None):
    for pl in planners:
        for fam in ('core','paper','ring','all'):
            S=collections.defaultdict(list)
            for k,pr in P.items():
                if seeds and k[3] not in seeds: continue
                if fam!='all' and k[0]!=fam: continue
                c=D.get((pl,)+k)
                if not c: continue
                J={K:v[0] for K,v in c.items()}; Ks=sorted(J); am=min(J,key=J.get)
                for nm in ('old','v3','v33'):
                    K=pr[nm]; S[nm+'w'].append(abs(am-K)<=1); S[nm+'g'].append(J[min(Ks,key=lambda x:abs(x-K))]/J[am]-1)
            if not S['oldw']: continue
            line=f"{label:7s} {pl:15s} {fam:5s} n={len(S['oldw']):3d}"
            for nm in ('old','v3','v33'):
                g=np.array(S[nm+'g']); line+=f" | {nm:3s} w1 {np.mean(S[nm+'w']):.2f} mean {100*g.mean():5.1f} p90nr {100*nr90(g):5.1f} max {100*g.max():5.1f}"
            print(line)
report(R,PR,'retro',['sa','rr_tour','cyc','cluster_patrol','liu_mpga','gzz_rev','asghar_rev'])
report(H,PH,'held1',['sa','rr_tour','cyc'])
report(H,PH,'h1clean',['sa','rr_tour','cyc'],seeds=range(16,21))
ch=[(k,v) for k,v in list(PR.items())+list(PH.items()) if v['v3']!=v['v33']]
print('v3 != v3.3 in',len(ch),'instances; layouts',collections.Counter(k[0] for k,v in ch))
