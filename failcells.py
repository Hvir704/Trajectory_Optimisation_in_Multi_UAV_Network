import csv, collections, numpy as np
for lay,M,E in (("ring",50,3e6),("ring",100,3e6),("core",200,1.5e6),("core",200,3e6)):
    rows=[r for r in csv.DictReader(open("grid.csv"))
          if r["layout"]==lay and int(float(r["M"]))==M and float(r["Emax"])==E
          and float(r.get("L",12600) or 12600)==12600.0]
    by=collections.defaultdict(list)
    for r in rows: by[int(float(r["K"]))].append(r)
    print("\n%s M=%d %.1fMJ" % (lay,M,E/1e6))
    print("  K    J        J_age    J_event  n_never share  mean_n")
    for K in sorted(by):
        v=by[K]; f=lambda c: np.mean([float(r[c]) for r in v])
        print("  %-4d %.2e %.2e %.2e %6.1f %6.3f %6.1f" %
              (K,f("J"),f("J_age"),f("J_event"),f("n_never"),f("share_never"),f("mean_n")))
