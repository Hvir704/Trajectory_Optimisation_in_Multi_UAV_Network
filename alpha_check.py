import csv, collections, numpy as np
print("measured mean_n -- ring, 3 MJ.  Model must PREDICT these, not fit alpha to them.")
for M in (50,100,200):
    rows=[r for r in csv.DictReader(open("grid.csv"))
          if r["layout"]=="ring" and float(r["Emax"])==3e6 and int(float(r["M"]))==M
          and float(r.get("L",12600) or 12600)==12600.0]
    by=collections.defaultdict(list)
    for r in rows: by[int(float(r["K"]))].append(float(r["mean_n"]))
    print(" M=%-4d" % M, "  ".join("K%d:%.1f" % (K, np.mean(by[K])) for K in sorted(by)))
