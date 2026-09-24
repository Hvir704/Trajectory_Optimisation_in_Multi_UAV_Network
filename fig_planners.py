"""fig_planners.py -- the planner-robustness result as a figure.

For each of SA, the tour patrol and CP-SAT at M=40, 1.5 MJ: J(K) divided by SA's minimum over K on the
SAME instance, median over 12 seeds with an interquartile band. Normalising by SA's minimum (not each
planner's own) keeps the level differences visible, which is the point: the curves sit at different
heights but bottom out at the same fleet size. Markers: each planner's per-instance argmin, mean over
seeds, placed on its median curve. Dashed line: mean point prediction of the rule.

usage: python fig_planners.py m40_sa.csv m40_rr.csv m40_cp_final.csv out.pdf
"""
import sys, csv, math
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[(r["layout"], int(r["seed"]))][int(r["K"])] = r
    return d

sa, rr, cp = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
PL = [("SA (ours)", sa, "#1f4e79", "o"), ("tour patrol (age-blind)", rr, "#c55a11", "s"),
      ("CP-SAT (per-decision optimal)", cp, "#548235", "^")]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=False)
for ax, lay, title in zip(axes, ("paper", "ring"), ("clustered (reach-bound)", "ring (commute-bound)")):
    seeds = sorted(s for (l, s) in sa if l == lay)
    Ks = sorted(set.intersection(*[set(d[(lay, s)]) for _, d, _, _ in PL for s in seeds]))
    rule_vals = []
    for s in seeds:
        r0 = sa[(lay, s)][min(sa[(lay, s)])]
        kr = math.floor(float(r0["K_reach_i"])); kc = float(r0["K_commute_i"]); law = min(kr, kc)
        rule_vals.append(law if law == kr else round(law))
    for name, d, col, mk in PL:
        norm = np.array([[float(d[(lay, s)][K]["J"]) / min(float(sa[(lay, s)][k]["J"]) for k in Ks) for K in Ks] for s in seeds])
        med = np.median(norm, axis=0); q1, q3 = np.quantile(norm, [.25, .75], axis=0)
        ax.plot(Ks, med, color=col, lw=1.2, label=name)
        ax.fill_between(Ks, q1, q3, color=col, alpha=0.15, lw=0)
        am = np.mean([Ks[int(np.argmin(row))] for row in norm])
        ax.plot([am], [np.interp(am, Ks, med)], marker=mk, color=col, ms=6, mec="white", mew=0.8, zorder=5)
    ax.axvline(np.mean(rule_vals), color="k", ls="--", lw=0.8, label="rule (mean prediction)")
    ax.set_yscale("log"); ax.set_xlabel("fleet size $K$"); ax.set_title(title, fontsize=8)
    ax.set_xticks(Ks)
    top = {"paper": 6.0, "ring": 4.0}[lay]
    ax.set_ylim(0.85, top)
    ticks = [t for t in (1, 1.25, 1.5, 2, 3, 4, 6) if t <= top]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:g}" for t in ticks]); ax.minorticks_off()
    ax.grid(True, which="both", lw=0.3, alpha=0.4)
axes[0].set_ylabel(r"$J(K)\,/\,\min_K J_{\mathrm{SA}}$")
axes[1].legend(loc="upper right", fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[4], bbox_inches="tight")
fig.savefig(sys.argv[4].replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print("saved", sys.argv[4])
