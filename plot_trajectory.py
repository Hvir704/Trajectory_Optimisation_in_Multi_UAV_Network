#!/usr/bin/env python
"""
plot_trajectory.py -- Figure: one solved instance, K UAV routes over M sensors.

Renders the 1000x1000 sensor field, the central depot, the M ground sensors
(marker size proportional to priority weight w_i in [1,10]), and the K UAV
routes as coloured polylines depot -> visit order -> depot.

Routes come from sa_routes.sa_with_routes (bit-identical to compare_baseline.sa).
This script runs no new experiment and changes no solver state. Every number it
prints or draws comes from compare_baseline's OWN functions (chain_waoi,
chain_energy, fleet_obj) -- nothing about the objective is re-implemented here.

Typical use (a K* cell, so the figure shows the OPTIMAL fleet):
    python plot_trajectory.py --M 50 --K 4 --Emax 50000 --seed 2025 --sa-seed 0 \
        --out figs/traj_M50_K4_E50000.png

Settle the SA settings against a banked number first:
    python plot_trajectory.py --selftest

SEEDS -- compare_baseline.py:111-113 does NOT call gen(M, 2025). It seeds an
RNG with INSTANCE_SEED=2025 and draws `--instances` instance seeds from it,
then uses the enumerate index as the SA seed:

    rng=np.random.default_rng(INSTANCE_SEED)
    seeds=[int(rng.integers(0,10_000_000)) for _ in range(a.instances)]
    vals=[sa(*gen(M,s),K,Ee,M,a.iters,si) for si,s in enumerate(seeds)]

So gen(M, 2025) is NOT a member of the evaluation population and will never
match a table number. This script therefore selects instances by INDEX into
that drawn list:
    --instance-index I   plot instance I of the 30 (SA seed = I). default 0
    --meta-seed          the INSTANCE_SEED that generates the list. default 2025
    --raw-instance-seed  escape hatch: call gen(M, S) directly for an ad-hoc
                         instance that is NOT in the evaluation population

CONSISTENCY CHECKS run on every invocation:
    (a) fleet_obj(trajs) must equal the objective sa_with_routes returned
        -- catches routes that don't correspond to the reported objective;
    (b) sum of per-chain J_k must equal fleet_obj(trajs)
        -- guaranteed by fleet_obj's structure, verified anyway;
    (c) chain_energy(t) <= Ee for every chain (compare_baseline.feasible).
Any failure is printed loudly and the figure is still written, so you can look
at what went wrong.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Shared palette. Assign UAV colours ONCE here; later figure scripts should
# import UAV_COLORS from this module so the colour scheme is consistent.
# --------------------------------------------------------------------------
UAV_COLORS = [
    "#1b9e77",  # teal
    "#d95f02",  # orange
    "#7570b3",  # purple
    "#e7298a",  # magenta
    "#66a61e",  # green
    "#e6ab02",  # gold
    "#a6761d",  # brown
    "#666666",  # grey
    "#1f78b4",  # blue
    "#b2182b",  # dark red
]
UNSERVED_COLOR = "#c8c8c8"
DEPOT_COLOR = "#ffd21f"

BANKED = {  # (M, K, Emax) -> 30-instance MEAN objective, CONTEXT_00_SHARED_FACTS
    (100, 4, 50000): -235.45,
    (100, 1, 50000): -89.56,
    (200, 4, 50000): -348.39,
}
TOL = 5e-3  # banked-number match tolerance


# --------------------------------------------------------------------------
# plumbing -- compare_baseline is the single source of truth
# --------------------------------------------------------------------------
def cb():
    import compare_baseline
    return compare_baseline


def require(mod, names):
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        sys.exit("FATAL: compare_baseline lacks %s -- this script calls the "
                 "module's own objective functions and will not guess at "
                 "substitutes." % ", ".join(missing))


def instance_seeds(meta_seed, n):
    """Exactly compare_baseline.py:111-112."""
    rng = np.random.default_rng(meta_seed)
    return [int(rng.integers(0, 10_000_000)) for _ in range(n)]


def resolve_seeds(args):
    """-> (gen_seed, sa_seed, label). Table mode unless --raw-instance-seed."""
    if args.raw_instance_seed is not None:
        return (args.raw_instance_seed, args.sa_seed,
                "ad-hoc instance gen(M,%d) -- NOT in the evaluation population"
                % args.raw_instance_seed)
    n = max(args.instances, args.instance_index + 1)
    seeds = instance_seeds(args.meta_seed, n)
    s = seeds[args.instance_index]
    return (s, args.instance_index,
            "instance %d of %d from meta-seed %d -> gen(M,%d), SA seed %d"
            % (args.instance_index, args.instances, args.meta_seed, s,
               args.instance_index))


def load_instance(M, seed):
    pos, wi, tcd = cb().gen(M, seed)
    return (np.asarray(pos, float), np.asarray(wi, float).ravel(),
            np.asarray(tcd, float).ravel())


def solve(pos, wi, tcd, K, Ee, M, iters, sa_seed):
    from sa_routes import sa_with_routes
    obj, trajs = sa_with_routes(pos, wi, tcd, K, Ee, M, iters, sa_seed)
    return float(obj), [list(map(int, t)) for t in trajs]


def chain_J(chain, pos, wi, tcd):
    """The single-chain term of fleet_obj (compare_baseline.py:46)."""
    m = cb()
    return (m.TH1 * m.chain_waoi(chain, pos, wi, tcd)
            - m.TH2 * float(sum(wi[j] for j in chain)))


def strip_depot_indices(trajs, pos, home, tol=1e-9):
    depot_idx = {i for i in range(pos.shape[0])
                 if np.linalg.norm(pos[i] - home) < tol}
    if not depot_idx:
        return trajs, 0
    cleaned = [[i for i in t if i not in depot_idx] for t in trajs]
    return cleaned, sum(len(a) - len(b) for a, b in zip(trajs, cleaned))


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def ring_sizes(w, smin=70.0, smax=330.0, wlo=1.0, whi=10.0):
    """Outer priority ring area, scaled by weight w_i."""
    w = np.clip(np.asarray(w, float), wlo, whi)
    return smin + (w - wlo) / (whi - wlo) * (smax - smin)


def plot(pos, wi, trajs, obj, chain_vals, chain_e, args, Ee, out_path):
    m = cb()
    home = np.asarray(m.HOME, float).reshape(2)
    L = float(m.AREA)
    M = pos.shape[0]

    served = np.zeros(M, dtype=bool)
    for chain in trajs:
        for i in chain:
            served[i] = True
    rings = ring_sizes(wi)
    waoi = float(sum(m.chain_waoi(t, pos, wi, tcd_of(args), ) for t in [])) \
        if False else None  # placeholder replaced below

    fig, ax = plt.subplots(figsize=(9.0, 8.2))

    # --- unserved sensors: grey ring + grey fill ---
    if (~served).any():
        idx = np.where(~served)[0]
        ax.scatter(pos[idx, 0], pos[idx, 1], s=rings[idx],
                   facecolors=UNSERVED_COLOR, edgecolors="#8f8f8f",
                   linewidths=0.9, zorder=2)
        if args.labels:
            for i in idx:
                ax.annotate("%d" % int(round(wi[i])),
                            (pos[i, 0], pos[i, 1]),
                            textcoords="offset points", xytext=(7, 6),
                            fontsize=7, color="#7a7a7a", zorder=2)

    # --- routes and served sensors ---
    handles = []
    for k, chain in enumerate(trajs):
        col = UAV_COLORS[k % len(UAV_COLORS)]
        if chain:
            xs = [home[0]] + [pos[i, 0] for i in chain] + [home[0]]
            ys = [home[1]] + [pos[i, 1] for i in chain] + [home[1]]
            ax.plot(xs, ys, "-", color=col, lw=1.6, alpha=0.95, zorder=3,
                    solid_capstyle="round")
            idx = np.asarray(chain, dtype=int)
            # hollow priority ring
            ax.scatter(pos[idx, 0], pos[idx, 1], s=rings[idx],
                       facecolors="none", edgecolors="#3a3a3a",
                       linewidths=0.9, zorder=4)
            # solid centre dot in the UAV colour
            ax.scatter(pos[idx, 0], pos[idx, 1], s=34, c=col,
                       edgecolors=col, linewidths=0.4, zorder=5)
            if args.labels:
                for i in idx:
                    ax.annotate("%d" % int(round(wi[i])),
                                (pos[i, 0], pos[i, 1]),
                                textcoords="offset points", xytext=(7, 6),
                                fontsize=7, color="#2a2a2a", zorder=6)
        handles.append(Line2D([0], [0], color=col, lw=2.0, marker="o",
                              markersize=6, markerfacecolor=col,
                              markeredgecolor=col,
                              label="UAV %d (%d nodes)" % (k + 1, len(chain))))

    # --- depot ---
    ax.scatter([home[0]], [home[1]], s=520, marker="*", c=DEPOT_COLOR,
               edgecolors="#000000", linewidths=1.1, zorder=7)

    ax.set_xlim(-30, L + 30)
    ax.set_ylim(-30, L + 30)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=11)
    ax.set_ylabel("y (m)", fontsize=11)
    ax.grid(True, lw=0.5, alpha=0.4, color="#cccccc")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444444")

    n_served = int(served.sum())
    W = float(sum(m.chain_waoi(t, pos, wi, args._tcd) for t in trajs))
    ax.set_title("Fleet trajectory  M=%d, K=%d, served=%d\n"
                 "Fleet WAoI=%.1f  Obj=%.2f" % (M, args.K, n_served, W, obj),
                 fontsize=13)
    ax.legend(handles=handles, loc="upper right", frameon=True,
              framealpha=0.95, edgecolor="#999999", fontsize=9.5)

    if args.footnote:
        fig.text(0.5, 0.045,
                 "ring size $\\propto$ priority $w_i$ (label = $w_i$);  "
                 "$E_{\\max}$=%s J, %s J per UAV;  instance seed %d, "
                 "%d SA iters"
                 % ("{:,}".format(int(args.Emax)),
                    "{:,}".format(int(round(Ee))), args.seed, args.iters),
                 ha="center", fontsize=8.5, color="#555555")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".",
                exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return M - n_served


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def run(args, make_figure=True, quiet=False):
    _p = (lambda *a, **k: None) if quiet else print
    m = cb()
    require(m, ["gen", "chain_waoi", "chain_energy", "fleet_obj",
                "TH1", "TH2", "HOME", "AREA"])
    home = np.asarray(m.HOME, float).reshape(2)
    Ee = float(args.Emax) if args.ee_is_total else float(args.Emax) / args.K
    gen_seed, sa_seed, label = resolve_seeds(args)
    args.seed, args.sa_seed = gen_seed, sa_seed
    _p("  %s" % label)

    pos, wi, tcd = load_instance(args.M, args.seed)
    if pos.shape[0] != args.M:
        _p("  [warn] gen(%d,%d) returned %d positions"
              % (args.M, args.seed, pos.shape[0]))

    args._tcd = tcd
    obj, trajs = solve(pos, wi, tcd, args.K, Ee, args.M, args.iters,
                       args.sa_seed)
    trajs, dropped = strip_depot_indices(trajs, pos, home)
    if dropped:
        _p("  [info] stripped %d depot index/indices from routes" % dropped)

    # (a) do the routes reproduce the reported objective?
    recomputed = float(m.fleet_obj(trajs, pos, wi, tcd))
    d_a = abs(recomputed - obj)
    # (b) does the per-chain decomposition sum to it?
    chain_vals = [chain_J(t, pos, wi, tcd) for t in trajs]
    d_b = abs(sum(chain_vals) - recomputed)
    # (c) feasibility
    chain_e = [float(m.chain_energy(t, pos, tcd)) for t in trajs]
    infeasible = [k for k, e in enumerate(chain_e) if e > Ee + 1e-6]
    # (d) did SA actually improve on its own starting point?
    g_obj = None
    if not args.skip_greedy_check and hasattr(m, "greedy_init"):
        g_trajs, _ = m.greedy_init(pos, wi, tcd, args.K, Ee, args.M)
        g_obj = float(m.fleet_obj(g_trajs, pos, wi, tcd))

    _p("  SA objective:            %.4f" % obj)
    _p("  fleet_obj(trajs):        %.4f   (|d|=%.2e %s)"
          % (recomputed, d_a, "OK" if d_a < 1e-6 else "MISMATCH"))
    _p("  sum of per-chain J_k:    %.4f   (|d|=%.2e %s)"
          % (sum(chain_vals), d_b, "OK" if d_b < 1e-6 else "MISMATCH"))
    if d_a >= 1e-6:
        _p("  [warn] the returned routes do not reproduce the returned "
              "objective. The figure would show a different solution than the "
              "number in its title -- investigate before using it.")
    if infeasible:
        _p("  [warn] chains over the %.0f J budget: %s"
              % (Ee, ", ".join("UAV %d" % (k + 1) for k in infeasible)))
    if g_obj is not None:
        imp = g_obj - obj
        _p("  greedy_init objective:   %.4f   (SA improved by %.4f, %.2f%%)"
              % (g_obj, imp, 100.0 * imp / abs(g_obj) if g_obj else 0.0))
        if abs(imp) < 1e-9:
            _p("  [warn] SA returned greedy_init's solution unchanged -- it "
                  "accepted no improving move in %d iters. Chains near the "
                  "energy cap reject most moves; raise --iters before trusting "
                  "this as an SA-quality solution." % args.iters)

    key = (args.M, args.K, int(args.Emax))
    if key in BANKED:
        _p("  note: the banked %s value %.2f is a MEAN over instances; a "
              "single instance is not expected to match it. Use --selftest."
              % (str(key), BANKED[key]))

    for k, t in enumerate(trajs):
        _p("  UAV %d: %3d nodes, energy %8.1f J (%5.1f%% of Ee), J_k=%.3f"
              % (k + 1, len(t), chain_e[k], 100.0 * chain_e[k] / Ee,
                 chain_vals[k]))

    if make_figure:
        n_un = plot(pos, wi, trajs, obj, chain_vals, chain_e, args, Ee,
                    args.out)
        _p("  served %d / %d nodes; wrote %s at %d dpi"
              % (args.M - n_un, args.M, args.out, args.dpi))
    return obj


def selftest(args):
    """Reproduce a banked TABLE MEAN, replicating compare_baseline.py:111-114."""
    m = cb()
    from sa_routes import sa_with_routes
    key = (args.M, args.K, int(args.Emax))
    target = BANKED.get(key)
    if target is None:
        print("  note: no banked mean for %s -- running as a DIAGNOSTIC "
              "(mean + freeze rate reported, no pass/fail)." % str(key))
    Ee = float(args.Emax) / args.K
    seeds = instance_seeds(args.meta_seed, args.instances)
    print("SELFTEST: M=%d K=%d Emax=%d, mean over %d instances, iters=%d%s\n"
          % (args.M, args.K, int(args.Emax), args.instances, args.iters,
             ("\n          expecting %.2f" % target) if target is not None
             else ""))

    vals, gvals = [], []
    for si, sd in enumerate(seeds):
        pos, wi, tcd = load_instance(args.M, sd)
        obj, _ = sa_with_routes(pos, wi, tcd, args.K, Ee, args.M, args.iters, si)
        vals.append(float(obj))
        line = "  [%2d/%2d] gen(%d,%d) sa_seed=%d  J=%9.4f  running mean %9.4f" \
               % (si + 1, len(seeds), args.M, sd, si, obj, np.mean(vals))
        if args.with_greedy:
            gt, _ = m.greedy_init(pos, wi, tcd, args.K, Ee, args.M)
            g = float(m.fleet_obj(gt, pos, wi, tcd))
            gvals.append(g)
            line += "  greedy %9.4f (SA better by %.2f%%)" % (
                g, 100.0 * (g - obj) / abs(g) if g else 0.0)
        print(line)

    mean = float(np.mean(vals))
    print("\n  SA mean over %d instances: %.4f" % (len(vals), mean))
    if target is not None:
        d = abs(mean - target)
        print("  banked value:              %.4f   (|d|=%.4f)" % (target, d))
    if args.with_greedy:
        gm = float(np.mean(gvals))
        inert = sum(1 for g, v in zip(gvals, vals) if abs(g - v) < 1e-9)
        print("  greedy_init mean:          %.4f   (SA better by %.2f%%)"
              % (gm, 100.0 * (gm - mean) / abs(gm) if gm else 0.0))
        print("  instances where SA found no improvement: %d / %d"
              % (inert, len(vals)))
    if target is None:
        print("\nDIAGNOSTIC COMPLETE -- no banked value for this cell.")
        return 0
    if abs(mean - target) < TOL:
        print("\nSELFTEST PASSED -- pipeline reproduces the banked mean. Use "
              "--iters %d and index into the same seed list for figures."
              % args.iters)
        return 0
    print("\nSELFTEST FAILED -- treat as a discrepancy to investigate, not a "
          "number to overwrite. Check --instances and --iters first.")
    return 1


def eval_cell(args, K, Ee, seeds, m, sa_with_routes):
    """Mean SA (and greedy) objective over the instance population for one K."""
    vals, gvals = [], []
    for si, sd in enumerate(seeds):
        pos, wi, tcd = load_instance(args.M, sd)
        obj, _ = sa_with_routes(pos, wi, tcd, K, Ee, args.M, args.iters, si)
        vals.append(float(obj))
        if args.with_greedy:
            gt, _ = m.greedy_init(pos, wi, tcd, K, Ee, args.M)
            gvals.append(float(m.fleet_obj(gt, pos, wi, tcd)))
    frozen = (sum(1 for g, v in zip(gvals, vals) if abs(g - v) < 1e-9)
              if gvals else None)
    return (float(np.mean(vals)),
            float(np.mean(gvals)) if gvals else None, frozen)


def sweep_k(args):
    """Locate K* = argmin_K mean J, and report whether SA and greedy agree."""
    m = cb()
    from sa_routes import sa_with_routes
    seeds = instance_seeds(args.meta_seed, args.instances)
    print("K-SWEEP: M=%d Emax=%d, %d instances, iters=%d\n"
          "         law predicts K* = Emax/12500 = %.2f -> %d\n"
          % (args.M, int(args.Emax), args.instances, args.iters,
             args.Emax / 12500.0, int(round(args.Emax / 12500.0))))
    rows = []
    for K in args.sweep_k:
        Ee = float(args.Emax) / K
        sa_m, g_m, frozen = eval_cell(args, K, Ee, seeds, m, sa_with_routes)
        rows.append((K, sa_m, g_m, frozen))
        print("  K=%d  Ee=%8.1f  SA mean %10.4f%s%s"
              % (K, Ee, sa_m,
                 ("  greedy %10.4f" % g_m) if g_m is not None else "",
                 ("  frozen %2d/%d" % (frozen, len(seeds)))
                 if frozen is not None else ""))

    sa_star = min(rows, key=lambda r: r[1])[0]
    print("\n  %-6s %-13s %-13s %-9s %s"
          % ("K", "SA mean", "greedy mean", "frozen", ""))
    for K, sa_m, g_m, fr in rows:
        mark = "  <-- K* (SA)" if K == sa_star else ""
        print("  %-6d %-13.4f %-13s %-9s%s"
              % (K, sa_m, "%.4f" % g_m if g_m is not None else "-",
                 "%d/%d" % (fr, len(seeds)) if fr is not None else "-", mark))
    print("\n  K* from SA:     %d" % sa_star)
    if args.with_greedy:
        g_star = min(rows, key=lambda r: r[2])[0]
        print("  K* from greedy: %d" % g_star)
        print("  -> solvers %s on K*"
              % ("AGREE" if g_star == sa_star else "DISAGREE"))
    pred = int(round(args.Emax / 12500.0))
    print("  K* from law:    %d   -> law %s at iters=%d"
          % (pred, "HOLDS" if pred == sa_star else "FAILS", args.iters))
    return 0


def pair_k(args):
    """Paired per-instance comparison of two K values, with a CI on the gap."""
    from sa_routes import sa_with_routes
    K1, K2 = args.pair_k[0], args.pair_k[1]
    seeds = instance_seeds(args.meta_seed, args.instances)
    print("PAIRED K=%d vs K=%d: M=%d Emax=%d, %d instances x %d SA seeds, "
          "iters=%d\n" % (K1, K2, args.M, int(args.Emax), args.instances,
                          args.reps, args.iters))
    print("  %-5s %-10s %-11s %-11s %-11s %-11s %s"
          % ("inst", "gen seed", "K%d best" % K1, "K%d best" % K2,
             "K%d mean" % K1, "K%d mean" % K2, "diff(best)"))

    d_best, d_mean = [], []
    for si, sd in enumerate(seeds):
        pos, wi, tcd = load_instance(args.M, sd)
        got = {}
        for K in (K1, K2):
            Ee = float(args.Emax) / K
            runs = []
            for r in range(args.reps):
                obj, _ = sa_with_routes(pos, wi, tcd, K, Ee, args.M,
                                        args.iters, si + 1000 * r)
                runs.append(float(obj))
            got[K] = (min(runs), float(np.mean(runs)))
        db = got[K1][0] - got[K2][0]
        dm = got[K1][1] - got[K2][1]
        d_best.append(db)
        d_mean.append(dm)
        print("  %-5d %-10d %-11.4f %-11.4f %-11.4f %-11.4f %+.4f"
              % (si, sd, got[K1][0], got[K2][0], got[K1][1], got[K2][1], db))

    for label, d in (("best-of-%d" % args.reps, d_best),
                     ("mean-of-%d" % args.reps, d_mean)):
        a = np.asarray(d, float)
        n = len(a)
        mean = float(a.mean())
        sd_ = float(a.std(ddof=1)) if n > 1 else 0.0
        se = sd_ / np.sqrt(n) if n > 1 else 0.0
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        wins = int((a < 0).sum())
        print("\n  --- %s ---" % label)
        print("  paired mean diff J(K=%d) - J(K=%d): %+.4f  (SD %.4f, SE %.4f)"
              % (K1, K2, mean, sd_, se))
        print("  95%% CI: [%+.4f, %+.4f]" % (lo, hi))
        print("  instances where K=%d is better: %d/%d" % (K1, wins, n))
        if lo < 0 < hi:
            print("  VERDICT: NOT RESOLVED -- CI contains 0, the two fleet "
                  "sizes are statistically indistinguishable here.")
        elif hi < 0:
            print("  VERDICT: K=%d better, CI excludes 0." % K1)
        else:
            print("  VERDICT: K=%d better, CI excludes 0." % K2)
    return 0


def batch(args):
    """Grid of trajectory figures over M x K, one PNG per cell + contact sheet."""
    import time as _time
    Ms, Ks = args.batch_m, args.batch_k
    os.makedirs(args.out_dir, exist_ok=True)
    print("BATCH: %d M-values x %d K-values = %d figures, Emax=%d, iters=%d, "
          "instance-index %d\n  out-dir %s\n"
          % (len(Ms), len(Ks), len(Ms) * len(Ks), int(args.Emax), args.iters,
             args.instance_index, args.out_dir))

    grid, done, skipped, failed = {}, 0, 0, []
    t_all = _time.time()
    for M in Ms:
        for K in Ks:
            out = os.path.join(
                args.out_dir, "traj_M%d_K%d_E%d_i%d.png"
                % (M, K, int(args.Emax), args.instance_index))
            grid[(M, K)] = out
            if os.path.exists(out) and not args.overwrite:
                print("  [skip] M=%-4d K=%-2d exists" % (M, K))
                skipped += 1
                continue
            sub = argparse.Namespace(**vars(args))
            sub.M, sub.K, sub.out = M, K, out
            t0 = _time.time()
            try:
                run(sub, make_figure=True, quiet=True)
                dt = _time.time() - t0
                done += 1
                print("  [ ok ] M=%-4d K=%-2d  %6.1fs  -> %s"
                      % (M, K, dt, os.path.basename(out)))
            except Exception as exc:  # keep the batch going
                failed.append((M, K, repr(exc)))
                print("  [FAIL] M=%-4d K=%-2d  %s" % (M, K, exc))

    print("\n  %d written, %d skipped, %d failed in %.1f s"
          % (done, skipped, len(failed), _time.time() - t_all))
    for M, K, e in failed:
        print("    M=%d K=%d: %s" % (M, K, e))

    if args.contact_sheet:
        sheet = os.path.join(args.out_dir, "contact_sheet_E%d.png"
                             % int(args.Emax))
        make_contact_sheet(Ms, Ks, grid, sheet, args)
    return 1 if failed else 0


def make_contact_sheet(Ms, Ks, grid, out, args):
    """Tile the per-cell PNGs into one M x K overview image."""
    import matplotlib.image as mpimg
    have = [(mk, p) for mk, p in grid.items() if os.path.exists(p)]
    if not have:
        print("  [skip] contact sheet: no figures on disk")
        return
    fig, axes = plt.subplots(len(Ms), len(Ks),
                             figsize=(2.5 * len(Ks), 2.6 * len(Ms)),
                             squeeze=False)
    for i, M in enumerate(Ms):
        for j, K in enumerate(Ks):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor("#dddddd")
            p = grid.get((M, K))
            if p and os.path.exists(p):
                ax.imshow(mpimg.imread(p))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        fontsize=8, color="#bbbbbb", transform=ax.transAxes)
            if i == 0:
                ax.set_title("K=%d" % K, fontsize=11)
            if j == 0:
                ax.set_ylabel("M=%d" % M, fontsize=11)
    fig.suptitle("Example trajectories, $E_{\\max}$=%s J, instance %d, "
                 "%d SA iters" % ("{:,}".format(int(args.Emax)),
                                  args.instance_index, args.iters),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=args.sheet_dpi, bbox_inches="tight")
    plt.close(fig)
    print("  wrote contact sheet %s" % out)


DEFAULT_OUT = "figs/traj.png"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--M", type=int, default=50)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--Emax", type=float, default=None,
                   help="TOTAL energy budget in J (default: compare_baseline.EMAX)")
    p.add_argument("--meta-seed", type=int, default=None,
                   help="INSTANCE_SEED that generates the instance-seed list "
                        "(default: compare_baseline.INSTANCE_SEED)")
    p.add_argument("--instances", type=int, default=30,
                   help="size of the evaluation population (compare_baseline "
                        "default is 30)")
    p.add_argument("--instance-index", type=int, default=0,
                   help="which instance of the population to plot; SA seed is "
                        "set to this index, matching enumerate()")
    p.add_argument("--raw-instance-seed", type=int, default=None,
                   help="escape hatch: gen(M,S) directly, OFF-population")
    p.add_argument("--sa-seed", type=int, default=0,
                   help="SA seed, only used with --raw-instance-seed")
    p.add_argument("--with-greedy", action="store_true",
                   help="selftest: also report the greedy_init mean")
    p.add_argument("--iters", type=int, default=4000,
                   help="SA iterations (compare_baseline default is 4000)")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--ee-is-total", action="store_true",
                   help="pass Emax rather than Emax/K as Ee")
    p.add_argument("--no-labels", dest="labels", action="store_false",
                   help="omit the per-node priority-weight labels")
    p.add_argument("--no-footnote", dest="footnote", action="store_false",
                   help="omit the settings footnote under the plot")
    p.add_argument("--arrows", action="store_true",
                   help="draw direction arrowheads on route segments")
    p.add_argument("--skip-greedy-check", action="store_true",
                   help="skip the greedy_init comparison (it is slow at large M)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--batch-m", type=int, nargs="+", default=None,
                   metavar="M", help="batch mode: M values for the grid")
    p.add_argument("--batch-k", type=int, nargs="+", default=None,
                   metavar="K", help="batch mode: K values for the grid")
    p.add_argument("--out-dir", default="figs",
                   help="batch mode: directory for the per-cell PNGs")
    p.add_argument("--overwrite", action="store_true",
                   help="batch mode: redraw cells whose PNG already exists")
    p.add_argument("--contact-sheet", action="store_true",
                   help="batch mode: also tile the cells into one overview PNG")
    p.add_argument("--sheet-dpi", type=int, default=150,
                   help="dpi for the contact sheet")
    p.add_argument("--pair-k", type=int, nargs=2, default=None,
                   metavar=("K1", "K2"),
                   help="paired per-instance comparison of two fleet sizes")
    p.add_argument("--reps", type=int, default=5,
                   help="SA seeds per instance for --pair-k")
    p.add_argument("--sweep-k", type=int, nargs="+", default=None,
                   help="locate K* = argmin_K mean J over the population")
    args = p.parse_args()

    m = cb()
    print("  plot_trajectory.py rev9 (selftest NameError fixed)")
    if args.Emax is None:
        args.Emax = float(getattr(m, "EMAX", 50000.0))
    if args.meta_seed is None:
        args.meta_seed = int(getattr(m, "INSTANCE_SEED", 2025))
    args.seed = args.meta_seed  # placeholder; resolve_seeds() overwrites it
    print("  constants: TH1=%g TH2=%g V=%g AREA=%g PF=%g PH=%g HOME=(%g,%g)"
          % (m.TH1, m.TH2, m.V, m.AREA, m.PF, m.PH,
             np.asarray(m.HOME).ravel()[0], np.asarray(m.HOME).ravel()[1]))

    if args.batch_m or args.batch_k:
        if not (args.batch_m and args.batch_k):
            sys.exit("batch mode needs BOTH --batch-m and --batch-k")
        sys.exit(batch(args))
    if args.pair_k:
        sys.exit(pair_k(args))
    if args.sweep_k:
        sys.exit(sweep_k(args))
    if args.selftest:
        sys.exit(selftest(args))
    run(args)


if __name__ == "__main__":
    main()