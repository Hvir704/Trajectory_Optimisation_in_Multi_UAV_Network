"""ablation.py -- paired comparison of planner variants at fixed K.  usage: python ablation.py ablation.csv [K=4]"""
import sys, csv, math, statistics as st, collections

K = sys.argv[2] if len(sys.argv) > 2 else "4"

def _i(v, default=0):
    """CSV field -> int. fix_schema.py backfills columns with float defaults, so an int column can
    come back as '4.0' or ''; comparing those as strings silently matches nothing."""
    try: return int(float(v))
    except (TypeError, ValueError): return int(default)

rows = [r for r in csv.DictReader(open(sys.argv[1])) if _i(r["K"], -1) == _i(K, -2)]

# divert MUST be in the key: (sa, exclude, per_leg_sa) is produced both by "--planner sa --replan
# per_leg_sa" and by "--replan per_leg_sa --divert 1" (--planner defaults to sa), and without it the
# second silently overwrites the first for every seed.
cfg = lambda r: (r["planner"], r["coord"], r["replan"], str(_i(r.get("divert", 0))))
ref = ("sa", "exclude", "launch", "0")

d = collections.defaultdict(dict)
dups = collections.Counter()
for r in rows:
    c, s = cfg(r), int(r["seed"])
    if s in d[c]: dups[c] += 1
    d[c][s] = r
for c, k in sorted(dups.items()):
    print(f"WARNING: {k} duplicate (config, seed) row(s) for {c} -- last kept", file=sys.stderr)

if ref not in d:
    print("reference config (sa, exclude, launch, divert=0) missing"); sys.exit()

def sign(x):
    """Two-sided sign test with ties dropped. Returns (#positive, #non-tied pairs, p)."""
    nz = [v for v in x if v != 0.0]
    n = len(nz); s = sum(1 for v in nz if v > 0)
    if n == 0: return 0, 0, 1.0
    return s, n, min(1.0, 2 * sum(math.comb(n, k) for k in range(max(s, n - s), n + 1)) / 2 ** n)

order = sorted(d, key=lambda c: (c != ref, c))          # reference row first, rest alphabetical
print(f"{'planner':16}{'coord':9}{'replan':13}{'div':>4}{'J mean':>11}{'dJ vs full':>12}"
      f"{'worse in':>10}{'p':>8}{'seeds':>7}{'trunc':>7}{'dup':>6}")
for c in order:
    ks = sorted(set(d[c]) & set(d[ref]))
    if not ks: continue
    J  = [float(d[c][s]["J"])   for s in ks]
    Jr = [float(d[ref][s]["J"]) for s in ks]
    diff = [a - b for a, b in zip(J, Jr)]
    s_, n, p = sign(diff)
    tr = st.mean(float(d[c][s]["trunc_frac"]) for s in ks)
    du = st.mean(float(d[c][s]["dup_frac"])   for s in ks)
    print(f"{c[0]:16}{c[1]:9}{c[2]:13}{c[3]:>4}{st.mean(J):11.3e}"
          f"{100 * st.mean(diff) / st.mean(Jr):+11.1f}%{s_:>6}/{n:<3}{p:8.4f}"
          f"{len(ks):>7}{tr:7.2f}{du:6.2f}")