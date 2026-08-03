#!/usr/bin/env python
"""
make_figures.py -- the five paper figures, read from CSVs that already exist.

Runs NO solver and no new experiment: every number plotted is read from disk.

    python make_figures.py --figure all --out-dir figs
    python make_figures.py --figure kstar --out figs/fig1_kstar_surface.png

Figures:
    kstar    K* surface over (M, Emax) with the Emax/12,500 law overlaid,
             1-SEM bands from Kstar_lo/Kstar_hi drawn as vertical extent.
    reach    reach law r(e): the linear branch a*(e-e_min) and the r_cap
             plateau, measured points from primitives_raw.csv.
    phi      deconfliction penalty Phi as % of |J_route| vs K, one line per
             separation radius delta.
    pareto   speed-quality operating points (MLP / greedy-ish SA0 / SA2000 /
             SA8000) as RATIOS, never raw ms.
    predict  predicted vs measured K*, held-out cells marked separately.

Colours come from plot_trajectory.UAV_COLORS when that module is importable,
so the whole figure set matches the trajectory plot.
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:  # keep the palette consistent with the trajectory figure
    from plot_trajectory import UAV_COLORS
except Exception:
    UAV_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
                  "#8c564b", "#17becf", "#e377c2", "#7f7f7f", "#bcbd22"]

LAW_E_STAR = 12500.0
GREY = "#8a8a8a"
LAWC = "#111111"


# --------------------------------------------------------------------------
def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    """Float from a csv cell; '' and 'nan' -> default."""
    v = row.get(key, "")
    if v is None:
        return default
    v = str(v).strip()
    if v == "" or v.lower() == "nan":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def need(path, what):
    rows = read_csv(path)
    if rows is None:
        print("  [skip] %s not found -- cannot draw %s" % (path, what))
    return rows


def finish(fig, ax_or_none, out, dpi, note=None):
    if note:
        fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#555555")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  wrote %s" % out)


def color_for(i):
    return UAV_COLORS[i % len(UAV_COLORS)]


# --------------------------------------------------------------------------
# Figure 1 -- K* surface
# --------------------------------------------------------------------------
def fig_kstar(args):
    rows = need(os.path.join(args.data_root, "kstar_sa", "kstar_sa.csv"),
                "the K* surface")
    if not rows:
        return
    Ms = sorted({int(fnum(r, "M")) for r in rows})
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    Es_all = sorted({fnum(r, "Emax") for r in rows})
    law_x = np.linspace(min(Es_all) * 0.92, max(Es_all) * 1.05, 200)
    ax.plot(law_x, law_x / LAW_E_STAR, "--", color=LAWC, lw=1.6, zorder=2,
            label="law  $K^*=E_{\\max}/12{,}500$")

    wide = []
    for i, M in enumerate(Ms):
        sub = sorted([r for r in rows if int(fnum(r, "M")) == M],
                     key=lambda r: fnum(r, "Emax"))
        x = np.array([fnum(r, "Emax") for r in sub])
        y = np.array([fnum(r, "Kstar") for r in sub])
        lo = np.array([fnum(r, "Kstar_lo", float("nan")) for r in sub])
        hi = np.array([fnum(r, "Kstar_hi", float("nan")) for r in sub])
        nb = np.array([fnum(r, "n_in_band", 1.0) for r in sub])
        c = color_for(i)
        yerr = np.vstack([np.nan_to_num(y - lo, nan=0.0),
                          np.nan_to_num(hi - y, nan=0.0)])
        ax.errorbar(x, y, yerr=yerr, fmt="o-", color=c, lw=1.4, ms=5.5,
                    capsize=3.5, elinewidth=1.2, zorder=3,
                    label="M=%d" % M)
        for xi, yi, n in zip(x, y, nb):
            if n > 1:  # ambiguous cell: band holds more than one K
                ax.scatter([xi], [yi], s=190, facecolors="none",
                           edgecolors=c, lw=1.4, zorder=4)
                wide.append((M, int(xi), int(n)))

    ax.set_xlabel("total energy budget $E_{\\max}$ (J)")
    ax.set_ylabel("optimal fleet size $K^*$")
    ax.set_title("Optimal fleet size is set by the energy budget, not by $M$",
                 fontsize=12)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    note = ("bars span the 1-SEM band [$K^*_{lo}$, $K^*_{hi}$]; "
            "hollow rings mark cells where the band holds >1 fleet size")
    if wide:
        note += "  (%s)" % ", ".join("M=%d/%dk: %d" % (m, e // 1000, n)
                                     for m, e, n in wide)
    finish(fig, ax, args.out or os.path.join(args.out_dir,
                                             "fig1_kstar_surface.png"),
           args.dpi, note)


# --------------------------------------------------------------------------
# Figure 2 -- reach law r(e)
# --------------------------------------------------------------------------
def fig_reach(args):
    raw = need(os.path.join(args.data_root, "kstar_primitives",
                            "primitives_raw.csv"), "the reach law")
    if not raw:
        return
    pred = read_csv(os.path.join(args.data_root, "kstar_predict",
                                 "kstar_predicted.csv")) or []
    fit = read_csv(os.path.join(args.data_root, "kstar_primitives",
                                "primitives_fit.csv")) or []
    pred_by_M = {int(fnum(r, "M")): r for r in pred}
    fit_by_M = {int(fnum(r, "M")): r for r in fit}

    Ms = sorted({int(fnum(r, "M")) for r in raw})
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    disagree = []

    for i, M in enumerate(Ms):
        sub = sorted([r for r in raw if int(fnum(r, "M")) == M],
                     key=lambda r: fnum(r, "e"))
        e = np.array([fnum(r, "e") for r in sub])
        r = np.array([fnum(r, "r") for r in sub])
        sd = np.array([fnum(r_, "r_std", 0.0) for r_ in sub])
        c = color_for(i)
        ax.errorbar(e, r, yerr=sd, fmt="o", color=c, ms=5, capsize=3,
                    elinewidth=1.0, alpha=0.9, zorder=3, label="M=%d" % M)

        p = pred_by_M.get(M)
        if p is not None:
            a = fnum(p, "a")
            emin = fnum(p, "e_min")
            rcap = fnum(p, "r_cap")
            xs = np.linspace(e.min(), e.max(), 300)
            ys = np.clip(a * (xs - emin), 0, rcap)
            ax.plot(xs, ys, "-", color=c, lw=1.5, alpha=0.85, zorder=2)
            ax.axhline(rcap, color=c, ls=":", lw=1.0, alpha=0.55, zorder=1)
            ax.annotate("$r_{cap}$=%.1f" % rcap, (e.max(), rcap),
                        textcoords="offset points", xytext=(-4, 4),
                        ha="right", fontsize=8, color=c)
            f = fit_by_M.get(M)
            if f is not None:
                a2, em2 = fnum(f, "a"), fnum(f, "e_min")
                if a2 == a2 and (abs(a2 - a) / max(a, 1e-12) > 0.25
                                 or em2 < 0 <= emin):
                    disagree.append((M, a, a2, emin, em2, fnum(f, "R2")))

    ax.set_xlabel("per-UAV energy above the depot commute, $e$ (J)")
    ax.set_ylabel("reach $r(e)$  (nodes served)")
    ax.set_title("Reach grows linearly in energy, then saturates at $r_{cap}$",
                 fontsize=12)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    note = ("lines: $r(e)=\\min(a(e-e_{min}),\\,r_{cap})$ "
            "from kstar_predict/kstar_predicted.csv")
    if disagree:
        note += ("   [!] primitives_fit.csv disagrees for M=%s -- see console"
                 % ", ".join(str(d[0]) for d in disagree))
        print("\n  [!] the two fit files do not agree on the reach law:")
        for M, a1, a2, em1, em2, r2 in disagree:
            print("      M=%-4d kstar_predict a=%.3e e_min=%+9.1f | "
                  "primitives_fit a=%.3e e_min=%+9.1f (R2=%.3f)"
                  % (M, a1, em1, a2, em2, r2))
        print("      primitives_fit has a NEGATIVE e_min and a low R2, which "
              "is not physical (it implies reach>0 at zero energy).")
        print("      The figure plots the kstar_predict fit. Decide which is "
              "the fit of record before this goes in the paper.\n")
    finish(fig, ax, args.out or os.path.join(args.out_dir,
                                             "fig2_reach_law.png"),
           args.dpi, note)


# --------------------------------------------------------------------------
# Figure 3 -- Phi vs K
# --------------------------------------------------------------------------
def fig_phi(args):
    rows = need(os.path.join(args.data_root, "phi_measure", "phi_measure.csv"),
                "the Phi figure")
    if not rows:
        return
    Ms = sorted({int(fnum(r, "M")) for r in rows})
    deltas = sorted({fnum(r, "delta") for r in rows})
    fig, axes = plt.subplots(1, len(Ms), figsize=(4.6 * len(Ms), 4.6),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, M in zip(axes, Ms):
        for i, d in enumerate(deltas):
            sub = sorted([r for r in rows if int(fnum(r, "M")) == M
                          and fnum(r, "delta") == d],
                         key=lambda r: fnum(r, "K"))
            if not sub:
                continue
            K = [fnum(r, "K") for r in sub]
            pct = [fnum(r, "Phi_pct") for r in sub]
            ax.plot(K, pct, "o-", color=color_for(i), lw=1.5, ms=5,
                    label="$\\delta$=%g m" % d)
        ax.set_title("M=%d" % M, fontsize=11)
        ax.set_xlabel("fleet size $K$")
        ax.grid(True, lw=0.3, alpha=0.35)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[0].set_ylabel("deconfliction penalty $\\Phi$  (% of $|J_{route}|$)")
    axes[-1].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Deconfliction cost grows with fleet size and separation "
                 "radius", fontsize=12, y=1.0)
    finish(fig, None, args.out or os.path.join(args.out_dir,
                                               "fig3_phi_vs_K.png"),
           args.dpi,
           "geometry-bound: $\\Phi$ is set by fleet density, not by routing")


# --------------------------------------------------------------------------
# Figure 4 -- speed / quality Pareto (ratios only)
# --------------------------------------------------------------------------
def fig_pareto(args):
    rows = need(os.path.join(args.data_root, "bench_speed",
                             "bench_speed.csv"), "the Pareto figure")
    if not rows:
        return
    pts = {"MLP (rollout+pp)": [], "SA (0 iters, greedy init)": [],
           "SA (2000 iters)": [], "SA (8000 iters)": []}
    for r in rows:
        ref_obj = fnum(r, "sa_ref_obj")
        ref_s = fnum(r, "sa_ref_s")
        if not (ref_obj == ref_obj and ref_s == ref_s) or ref_obj == 0:
            continue
        rl_s = fnum(r, "rl_total_ms") / 1000.0
        for name, obj, sec in (
                ("MLP (rollout+pp)", fnum(r, "rl_obj"), rl_s),
                ("SA (0 iters, greedy init)", fnum(r, "sa0_obj"),
                 fnum(r, "sa0_s")),
                ("SA (2000 iters)", fnum(r, "sa2000_obj"),
                 fnum(r, "sa2000_s")),
                ("SA (8000 iters)", fnum(r, "sa8000_obj"),
                 fnum(r, "sa8000_s"))):
            if obj == obj and sec == sec and sec > 0:
                pts[name].append((sec / ref_s, obj / ref_obj))

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    marks = ["s", "^", "D", "o"]
    for i, (name, v) in enumerate(pts.items()):
        if not v:
            continue
        v = np.array(v)
        ax.scatter(v[:, 0], v[:, 1], s=54, marker=marks[i % len(marks)],
                   color=color_for(i), edgecolors="white", lw=0.5,
                   alpha=0.9, zorder=3, label="%s  (n=%d)" % (name, len(v)))
        ax.scatter([v[:, 0].mean()], [v[:, 1].mean()], s=210,
                   marker=marks[i % len(marks)], color=color_for(i),
                   edgecolors="black", lw=1.1, zorder=4)

    ax.axhline(1.0, color=GREY, ls="--", lw=1.0, zorder=1)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.0, zorder=1)
    ax.annotate("SA reference (8000 iters)", (1.0, 1.0),
                textcoords="offset points", xytext=(-8, 8), ha="right",
                fontsize=8.5, color=GREY)
    ax.set_xscale("log")
    ax.set_xlabel("wall-clock, relative to the SA reference  (log scale)")
    ax.set_ylabel("solution quality, relative to the SA reference\n"
                  "(1.0 = matches reference; lower = worse)")
    ax.set_title("Speed-quality operating points", fontsize=12)
    ax.grid(True, lw=0.3, alpha=0.35, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    dom = {r.get("dominance", "") for r in rows}
    note = "large outlined markers are per-method means over all (M,K) cells"
    if any("NO low-time dominance" in d for d in dom):
        note += ";  every cell reports NO low-time dominance for the policy"
    finish(fig, ax, args.out or os.path.join(args.out_dir,
                                             "fig4_speed_quality.png"),
           args.dpi, note)


# --------------------------------------------------------------------------
# Figure 5 -- predicted vs measured K*
# --------------------------------------------------------------------------
def fig_predict(args):
    meas_rows = need(os.path.join(args.data_root, "kstar_sa", "kstar_sa.csv"),
                     "the prediction scatter")
    if not meas_rows:
        return
    meas = {(int(fnum(r, "M")), int(fnum(r, "Emax"))): r for r in meas_rows}

    sets = [("fit", "kstar_predict"), ("held-out", "kstar_predict_heldout")]
    loaded = []
    for label, d in sets:
        rows = read_csv(os.path.join(args.data_root, d, "kstar_predicted.csv"))
        if rows:
            loaded.append((label, d, rows))
    if not loaded:
        print("  [skip] no kstar_predicted.csv found")
        return

    # are the fit and held-out files actually different?
    if len(loaded) == 2:
        a = [tuple(sorted(r.items())) for r in loaded[0][2]]
        b = [tuple(sorted(r.items())) for r in loaded[1][2]]
        if a == b:
            print("  [!] %s/kstar_predicted.csv is byte-identical to %s's."
                  % (loaded[1][1], loaded[0][1]))
            print("      Held-out cells cannot be marked as held out, because "
                  "nothing distinguishes them. Either the held-out run wrote "
                  "to the wrong directory or it re-fit on all cells.")
            print("      The figure is drawn WITHOUT a held-out series.")
            loaded = loaded[:1]

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    lim_hi = 0.0
    exact = tot = 0
    for si, (label, d, rows) in enumerate(loaded):
        xs, ys, ms = [], [], []
        for r in rows:
            M = int(fnum(r, "M"))
            for col, E in (("Kstar_E25000", 25000), ("Kstar_E50000", 50000),
                           ("Kstar_E100000", 100000)):
                p = fnum(r, col)
                mm = meas.get((M, E))
                if p != p or mm is None:
                    continue
                k = fnum(mm, "Kstar")
                xs.append(p)
                ys.append(k)
                ms.append(M)
                tot += 1
                if round(p) == k:
                    exact += 1
        if not xs:
            continue
        lim_hi = max(lim_hi, max(xs), max(ys))
        for i, M in enumerate(sorted(set(ms))):
            idx = [j for j, m in enumerate(ms) if m == M]
            ax.scatter([xs[j] for j in idx], [ys[j] for j in idx], s=66,
                       marker="o" if si == 0 else "^", color=color_for(i),
                       edgecolors="white" if si == 0 else "black",
                       lw=0.6 if si == 0 else 1.2, zorder=3,
                       label="M=%d%s" % (M, "" if si == 0 else " (held-out)"))

    hi = math.ceil(lim_hi) + 1
    ax.plot([0, hi], [0, hi], "--", color=LAWC, lw=1.3, zorder=2,
            label="perfect prediction")
    for off, st in ((0.5, ":"),):
        ax.plot([0, hi], [off, hi + off], st, color=GREY, lw=1.0, zorder=1)
        ax.plot([0, hi], [-off, hi - off], st, color=GREY, lw=1.0, zorder=1)

    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("predicted $K^*$ (derived, continuous)")
    ax.set_ylabel("measured $K^*$ (SA argmin, integer)")
    ax.set_title("Derived $K^*$ vs measured $K^*$", fontsize=12)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    finish(fig, ax, args.out or os.path.join(args.out_dir,
                                             "fig5_predicted_vs_measured.png"),
           args.dpi,
           "dotted lines: $\\pm$0.5, the rounding band; %d/%d cells round to "
           "the measured $K^*$" % (exact, tot))


# --------------------------------------------------------------------------
FIGURES = {"kstar": fig_kstar, "reach": fig_reach, "phi": fig_phi,
           "pareto": fig_pareto, "predict": fig_predict}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figure", default="all",
                   choices=list(FIGURES) + ["all"])
    p.add_argument("--data-root", default=".",
                   help="directory holding kstar_sa/, phi_measure/, ...")
    p.add_argument("--out", default=None,
                   help="output path (single figure only)")
    p.add_argument("--out-dir", default="figs")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if args.figure == "all":
        if args.out:
            sys.exit("--out is for a single figure; use --out-dir with 'all'")
        for name, fn in FIGURES.items():
            print("[%s]" % name)
            fn(args)
    else:
        FIGURES[args.figure](args)


if __name__ == "__main__":
    main()