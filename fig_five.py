"""fig_five.py -- J(K) for every compared planner on two held-out deployments, rule's l marked.

Deployments are fixed by a rule applied before any Gao/Asghar curve was seen (selected.json):
for the paper and ring families at M=100, 1.5 MJ, the deployment whose C4-vs-cluster own-optimum
ratio is closest to that family's median (paper seed 10, ring seed 12).
Each curve is J(K) divided by SA's minimum on the same deployment (same convention as fig_planners);
markers at each planner's own optimum; dashed line at the rule's l (from SA's smallest-K row).

usage: python fig_five.py out.pdf --sa sa_rerun.csv --eval eval_all_v2.csv [--extra FILE ...]
--extra: any run_grid-schema CSV with planner names gzz_rev / asghar_rev (held-out seeds 1-12).
Planners missing from the inputs are listed and skipped, never invented."""
import sys, csv, json, math, argparse
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--sa", required=True)
ap.add_argument("--eval", required=True); ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()
sel = json.load(open("selected.json"))
NAMES = [("sa", "SA (ours)", "#1f4e79", "o"), ("cyc", "C4 (ours)", "#7030a0", "D"),
         ("cluster_patrol", "cluster patrol (Rahimi & S.)", "#c55a11", "s"),
         ("gzz_rev", "Gao et al.", "#548235", "^"), ("asghar_rev", "Asghar et al.", "#808080", "v")]
data = defaultdict(lambda: defaultdict(dict)); rule = {}
def take(path, rename=None):
    for r in csv.DictReader(open(path)):
        if int(float(r["M"])) != 100 or float(r["Emax"]) != 1.5e6 or float(r.get("Th", 43200)) != 43200: continue
        lay, s = r["layout"], int(float(r["seed"]))
        if lay not in sel or s != sel[lay]: continue
        pl = rename or r.get("planner", "sa")
        data[lay][pl][int(float(r["K"]))] = float(r["J"])
        if pl == "sa" and "K_reach_i" in r:
            k = int(float(r["K"]))
            if lay not in rule or k < rule[lay][0]: rule[lay] = (k, r)
take(a.sa, "sa"); take(a.eval)
for f in a.extra: take(f)
missing = [lab for key, lab, *_ in NAMES if not all(key in data[l] for l in sel)]
if missing: print("not in the inputs (skipped):", ", ".join(missing))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
for ax, lay, title in zip(axes, ("paper", "ring"), ("clustered (paper), seed %d" % sel["paper"], "ring, seed %d" % sel["ring"])):
    sa_min = min(data[lay]["sa"].values())
    for key, lab, col, mk in NAMES:
        if key not in data[lay]: continue
        Ks = sorted(data[lay][key]); y = [data[lay][key][K] / sa_min for K in Ks]
        ax.plot(Ks, y, color=col, lw=1.1, label=lab)
        kb = min(data[lay][key], key=data[lay][key].get)
        ax.plot([kb], [data[lay][key][kb] / sa_min], marker=mk, color=col, ms=6, mec="white", mew=0.8, zorder=5)
    r0 = rule[lay][1]; kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"])
    law = min(kr, kc); l = int(law) if law == kr else int(round(law))
    ax.axvline(l, color="k", ls="--", lw=0.8, label=r"rule's $\ell$")
    ax.set_yscale("log"); ax.set_title(title, fontsize=8); ax.set_xlabel("fleet size $K$")
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
    allK = [K for v in data[lay].values() for K in v]; ax.set_xlim(min(allK) - 0.5, max(allK) + 0.5)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 20, 50]
    lo, hi = ax.get_ylim(); ax.set_yticks([t for t in ticks if lo <= t <= hi]); ax.set_yticklabels([f"{t:g}" for t in ticks if lo <= t <= hi]); ax.minorticks_off()
    print(f"{lay}: rule l = {l}; own optima:", {k: min(v, key=v.get) for k, v in data[lay].items()})
axes[0].set_ylabel(r"$J(K)\,/\,\min_K J_{\mathrm{SA}}$"); axes[1].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(a.out, bbox_inches="tight"); fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", a.out)
