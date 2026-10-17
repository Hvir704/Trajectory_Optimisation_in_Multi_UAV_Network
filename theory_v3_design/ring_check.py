import os
import sys, numpy as np, collections, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "v34_eval.py")).read().split("def load(")[0])
R = collections.defaultdict(dict)
for r in csv.DictReader(open(os.environ.get('V3_ASSEMBLED', 'assembled') + '/grid_final.csv')):
    if r['layout']=='ring' and float(r['Th'])==43200 and float(r['L'])==12600 and r.get('planner','sa')=='sa' and r['coord']=='exclude' and r['replan']=='launch':
        R[(int(r['M']),float(r['Emax']),int(r['seed']))][int(r['K'])]=(float(r['J']),r['K_ref'])
cell = collections.defaultdict(list)
for (M,E,s),c in sorted(R.items()):
    p,F = field('ring',M,E,12600.0,s); r=np.linalg.norm(F.pos-p.home,axis=1)
    K0=int(float(next(iter(c.values()))[1])); rc,Pb,_=exante(p,r,np.full(M,p.lam_bits),F.pos,K0)
    J={K:v[0] for K,v in c.items()}; am=min(J,key=J.get)
    cell[(M,E)].append((am, old_rule(p,r,rc,Pb), rule(p,F,rc,Pb,('ring',M,12600.0,s),False), rule(p,F,rc,Pb,('ring',M,12600.0,s),True)))
for k,v in sorted(cell.items()):
    a=np.array(v); print(f"ring M={k[0]} {k[1]/1e6:.1f} MJ: measured {a[:,0].mean():.2f}  old {a[:,1].mean():.2f}  v3.3 {a[:,2].mean():.2f}  v3.4 {a[:,3].mean():.2f}")
