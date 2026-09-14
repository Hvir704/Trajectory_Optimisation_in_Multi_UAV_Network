import csv, collections
rows=[r for r in csv.DictReader(open("grid.csv")) if float(r.get("L",12600) or 12600)==12600.0]
cells=collections.defaultdict(lambda: collections.defaultdict(dict))
for r in rows:
    cells[(r["layout"],int(float(r["M"])),float(r["Emax"]))][int(float(r["seed"]))][int(float(r["K"]))]=float(r["J"])
print("%-6s %4s %8s  %-8s %-8s %s" % ("layout","M","Emax","at Kmax","at Kmin","verdict"))
for key in sorted(cells):
    hi=lo=n=0
    for seed,byK in cells[key].items():
        Ks=sorted(byK); am=min(Ks,key=lambda K:byK[K]); n+=1
        hi+=(am==Ks[-1]); lo+=(am==Ks[0])
    v="CENSORED HIGH" if hi/n>0.25 else ("censored low" if lo/n>0.25 else "interior")
    print("%-6s %4d %6.1fMJ  %2d/%-3d   %2d/%-3d  %s" % (key[0],key[1],key[2]/1e6,hi,n,lo,n,v))
