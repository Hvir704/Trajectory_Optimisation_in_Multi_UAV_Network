import glob, re, numpy as np
from dyn_env import DynParams
p=DynParams()
byK={}
for f in sorted(glob.glob('nodes_paper_M100_*_K*_s1_exclude_launch.npz')):
    K=int(re.search(r'_K(\d+)_',f).group(1)); d=np.load(f)
    c=2*p.Pf*d['r']/p.v
    served=d['visits']>0
    prof=c[served]/c[served].mean()
    byK[K]=(prof, served)
K0=min(byK)
for K in sorted(byK):
    both=byK[K][1]&byK[K0][1]
    a=(2*p.Pf*np.load(sorted(glob.glob('nodes_paper_M100_*_K%d_s1_exclude_launch.npz'%K))[0])['r']/p.v)[both]
    b=(2*p.Pf*np.load(sorted(glob.glob('nodes_paper_M100_*_K%d_s1_exclude_launch.npz'%K0))[0])['r']/p.v)[both]
    a=a/a.mean(); b=b/b.mean()
    print(K, 'rel-cost drift vs K=%d:'%K0, round(float(np.abs(a-b).mean()),4))
