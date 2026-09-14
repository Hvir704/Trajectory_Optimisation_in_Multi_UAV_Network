import csv, collections, numpy as np
rows=[r for r in csv.DictReader(open("grid.csv"))
      if r["layout"]=="ring" and float(r["Emax"])==3e6 and float(r.get("L",12600) or 12600)==12600.0]
by=collections.defaultdict(list)
for r in rows: by[(int(float(r["M"])), int(float(r["K"])))].append(float(r["r_c"]))
print("measured r_c (m) -- ring, 3 MJ.  Your r_c(K) fixed point must reproduce these curves.")
for M in sorted({m for m,_ in by}):
    Ks=sorted(k for m,k in by if m==M)
    print(" M=%-4d" % M, "  ".join("K%d:%d" % (K, round(np.mean(by[(M,K)]))) for K in Ks))
