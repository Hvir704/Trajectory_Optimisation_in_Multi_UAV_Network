import csv, collections, numpy as np
rows=[r for r in csv.DictReader(open("grid.csv")) if float(r.get("L",12600) or 12600)==12600.0]
by=collections.defaultdict(list)
for r in rows:
    by[(r["layout"], int(float(r["M"])), float(r["Emax"]))].append((int(float(r["K"])), float(r["r_c"])))
print("%-6s %4s %7s  %s" % ("layout","M","Emax","r_c(K)"))
for key in sorted(by):
    d=collections.defaultdict(list)
    for K,rc in by[key]: d[K].append(rc)
    Ks=sorted(d); v=[np.mean(d[K]) for K in Ks]
    ch=100*(v[-1]-v[0])/v[0]
    tag="FALLS" if ch<-5 else ("flat" if abs(ch)<=5 else "RISES")
    print("%-6s %4d %5.1fMJ  K%-2d:%5d -> K%-2d:%5d  %+6.1f%%  %s" %
          (key[0],key[1],key[2]/1e6,Ks[0],round(v[0]),Ks[-1],round(v[-1]),ch,tag))
