import time, statistics as st
from dyn_env import *
from sa_sortie import *
for M in (50,100,200,400):
    ts=[]
    for rep in range(1,11):
        p=DynParams(M=M,K=4); s=DynSim(p,build_sa_planner(iters=1200,seed_base=rep),seed=rep)
        s.field.advance(0,3600)
        t=time.perf_counter(); s._launch(0,3600.0); ts.append(time.perf_counter()-t)
    print(M, 'median %.3f s   min %.3f   max %.3f   (n=10)' % (st.median(ts), min(ts), max(ts)))
