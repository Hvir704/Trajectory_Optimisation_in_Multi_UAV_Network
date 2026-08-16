#!/usr/bin/env python
"""
make_all_figures.py -- every figure for the paper, from one file.  [rev2]

Three sections, each independent. A section that cannot run (missing module,
missing CSV) reports why and the others still run.

  traj   M x K grid of example-trajectory figures + a contact sheet.
         Needs: compare_baseline.py, sa_routes.py
  fleet  (a) objective vs K and (b) fleet coverage vs K, one two-panel figure
         per M, plus a stacked grid. Needs: multi_uav_solver.py
  paper  the five CSV-driven figures: K* surface, reach law, Phi vs K,
         speed-quality Pareto, predicted-vs-measured K*.
         Needs: the CSVs under --data-root (no solver, no experiment)

Run everything:
    python make_all_figures.py --all

Run one section:
    python make_all_figures.py --only paper
    python make_all_figures.py --only traj --traj-m 50 100 --traj-k 1 2 4

ENERGY CONVENTION for the fleet section -- pick deliberately:
    --energy split  E_each = MP.Emax / K   (K drones share a FIXED total
                    budget; the convention the K* law is stated in)
    --energy fixed  E_each = MP.Emax_each  (every drone gets a full battery,
                    so total energy GROWS with K; this is the silent default
                    inside eval_fleet_baselines)

Solver diagnostics (--selftest, --sweep-k, --pair-k) are NOT here; they live in
plot_trajectory.py. This file only draws.
"""

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# One palette for every figure in the set (Dark2).
# --------------------------------------------------------------------------
UAV_COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2182b",
]
UNSERVED_COLOR = "#c8c8c8"
DEPOT_COLOR = "#ffd21f"

LAW_E_STAR = 12500.0
GREY = "#8a8a8a"
LAWC = "#111111"

POLICY_LABEL = "Fleet MLP (ours)"
POLICY_COLOR = "#009E73"
ALL_NODES_COLOR = "#d62728"
FALLBACK_BASE_COLORS = {
    "Multi-Random": "#888888",
    "Multi-NearestNeighbor": "#E69F00",
    "Multi-GreedyPriority": "#56B4E9",
    "Multi-PDR": "#CC79A7",
}
POLICY_FN_NAMES = ["eval_fleet_policy", "eval_fleet", "eval_fleet_mlp",
                   "eval_policy_fleet", "eval_fleet_rollout"]

BANKED = {  # (M, K, Emax) -> 12-instance MEAN objective
    (100, 4, 50000): -235.45,
    (100, 1, 50000): -89.56,
    (200, 4, 50000): -348.39,
}
TOL = 5e-3


# ==========================================================================
# SECTION 1 -- example trajectories
# ==========================================================================
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




# ==========================================================================
# SECTION 2 -- objective and coverage vs fleet size
# ==========================================================================
def load_solver():
    try:
        import multi_uav_solver as ms
    except Exception as exc:
        sys.exit("FATAL: cannot import multi_uav_solver (%r).\n"
                 "Run this from the repo root with the venv active." % exc)
    for req in ("FLEET_BASELINES", "eval_fleet_baselines", "MP"):
        if not hasattr(ms, req):
            sys.exit("FATAL: multi_uav_solver has no %s" % req)
    return ms


def find_policy_fn(ms, override):
    names = ([override] if override else []) + POLICY_FN_NAMES
    for n in names:
        fn = getattr(ms, n, None)
        if callable(fn):
            return n, fn
    return None, None


def ckpt_path(a, M, K):
    return os.path.join(a.models_dir,
                        a.ckpt_pattern.format(M=M, K=K, seed=a.model_seed))


def load_policy(ms, a, M, K, cache):
    """Load the (M,K) checkpoint into a MultiUAVPolicy. None if unavailable."""
    key = (M, K)
    if key in cache:
        return cache[key]
    path = ckpt_path(a, M, K)
    if not os.path.exists(path):
        cache[key] = None
        return None
    try:
        import torch
        cls = getattr(ms, "MultiUAVPolicy", None)
        if cls is None:
            raise RuntimeError("multi_uav_solver has no MultiUAVPolicy")
        pol = cls()
        ck = torch.load(path, map_location=a.device, weights_only=False)
        state = ck.get("policy", ck) if isinstance(ck, dict) else ck
        pol.load_state_dict(state)
        pol.eval()
        pol.to(a.device)
        cache[key] = pol
    except Exception as exc:
        print("      [ckpt FAIL] %s: %r" % (os.path.basename(path), exc))
        cache[key] = None
    return cache[key]


def ckpt_inventory(a, Ms, Ks):
    have = [(M, K) for M in Ms for K in Ks
            if os.path.exists(ckpt_path(a, M, K))]
    total = len(Ms) * len(Ks)
    print("  checkpoints: %d/%d cells found under %s (pattern %s, seed %d)"
          % (len(have), total, a.models_dir, a.ckpt_pattern, a.model_seed))
    if not have:
        print("      none found -- the MLP series will be OMITTED and the "
              "figures will show baselines only.")
    elif len(have) < total:
        missing_M = sorted({M for M in Ms
                            if not any(m == M for m, _ in have)})
        if missing_M:
            print("      no checkpoint at all for M=%s"
                  % ", ".join(str(m) for m in missing_M))
        print("      cells without a checkpoint are skipped for the MLP line "
              "only; baselines are unaffected.")
    return set(have)


def base_colors(ms):
    c = dict(FALLBACK_BASE_COLORS)
    c.update(getattr(ms, "_C_BASE", {}) or {})
    return c


def emax_each_for(ms, K, mode):
    if mode == "split":
        total = float(getattr(ms.MP, "Emax", 50000.0))
        return total / K
    return float(getattr(ms.MP, "Emax_each", 50000.0))


# --------------------------------------------------------------------------
def collect(ms, args):
    """-> rows: list of dicts, one per (M, K, method)."""
    pol_name, pol_fn = find_policy_fn(ms, args.policy_fn)
    if pol_fn is None:
        print("  [warn] no fleet-policy evaluator found (tried: %s); "
              "baselines only." % ", ".join(POLICY_FN_NAMES))
    else:
        print("  policy evaluator: multi_uav_solver.%s" % pol_name)
    have = ckpt_inventory(args, args.M, args.K) if pol_fn else set()
    cache = {}

    rows = []
    for M in args.M:
        for K in args.K:
            Ee = emax_each_for(ms, K, args.energy)
            print("  M=%-4d K=%-2d Emax_each=%9.1f" % (M, K, Ee), end="",
                  flush=True)
            bl = ms.eval_fleet_baselines(M, K, n=args.instances,
                                         seed=args.seed, Emax_each=Ee)
            for name, d in bl.items():
                rows.append(dict(M=M, K=K, Emax_each=Ee, method=name,
                                 obj=float(d["obj"]),
                                 nodes=float(d.get("nodes", float("nan")))))
            if pol_fn is not None and (M, K) in have:
                pol = load_policy(ms, args, M, K, cache)
                if pol is not None:
                    try:
                        p = pol_fn(pol, M, K, n=args.instances,
                                   seed=args.seed, device=args.device,
                                   Emax_each=Ee,
                                   use_postprocess=args.use_postprocess)
                        rows.append(dict(M=M, K=K, Emax_each=Ee,
                                         method=POLICY_LABEL,
                                         obj=float(p["obj"]),
                                         nodes=float(p.get("nodes",
                                                           float("nan")))))
                        print("  +MLP", end="")
                    except Exception as exc:
                        print("  [policy FAILED: %s]" % exc, end="")
            print("  done")
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "K", "Emax_each", "method",
                                          "obj", "nodes"])
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s" % path)


# --------------------------------------------------------------------------
def series(rows, M, method, key):
    sub = sorted([r for r in rows if r["M"] == M and r["method"] == method],
                 key=lambda r: r["K"])
    return ([r["K"] for r in sub], [r[key] for r in sub])


def panel_pair(rows, M, ms, args, ax_a, ax_b):
    cols = base_colors(ms)
    methods = [m for m in ms.FLEET_BASELINES]
    have_policy = any(r["method"] == POLICY_LABEL and r["M"] == M
                      for r in rows)

    # (a) objective vs K
    for name in methods:
        K, y = series(rows, M, name, "obj")
        if K:
            ax_a.plot(K, y, "s--", color=cols.get(name, "#999999"), lw=1.4,
                      ms=5, label=name, zorder=2)
    if have_policy:
        K, y = series(rows, M, POLICY_LABEL, "obj")
        ax_a.plot(K, y, "o-", color=POLICY_COLOR, lw=2.4, ms=8,
                  label=POLICY_LABEL, zorder=4)
        best = int(K[int(np.argmin(y))])
        ax_a.axvline(best, color=POLICY_COLOR, ls=":", lw=1.0, alpha=0.5,
                     zorder=1)
        ax_a.annotate("argmin K=%d" % best, (best, max(y)),
                      textcoords="offset points", xytext=(5, -10),
                      fontsize=8, color=POLICY_COLOR)
    ax_a.axhline(0, color="#333333", lw=0.9, zorder=1)
    ax_a.set_xlabel("Number of UAVs K")
    ax_a.set_ylabel("Fleet composite objective (lower=better)")
    ax_a.set_title("(a) Objective vs fleet size  (M=%d)" % M, fontsize=11)
    ax_a.grid(True, lw=0.4, alpha=0.35)
    ax_a.set_axisbelow(True)
    ax_a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_a.legend(fontsize=8, frameon=True, framealpha=0.9,
                edgecolor="#bbbbbb")

    # (b) coverage vs K
    if have_policy:
        K, y = series(rows, M, POLICY_LABEL, "nodes")
        ax_b.plot(K, y, "o-", color=POLICY_COLOR, lw=2.4, ms=8,
                  label=POLICY_LABEL, zorder=3)
    if args.coverage_baselines:
        for name in methods:
            K, y = series(rows, M, name, "nodes")
            if K:
                ax_b.plot(K, y, "s--", color=cols.get(name, "#999999"),
                          lw=1.2, ms=4, alpha=0.85, label=name, zorder=2)
    ax_b.axhline(M, color=ALL_NODES_COLOR, ls=":", lw=1.3,
                 label="All M=%d nodes" % M, zorder=1)
    ax_b.set_xlabel("Number of UAVs K")
    ax_b.set_ylabel("Total nodes served by fleet")
    ax_b.set_title("(b) Fleet coverage vs K  (M=%d)" % M, fontsize=11)
    ax_b.grid(True, lw=0.4, alpha=0.35)
    ax_b.set_axisbelow(True)
    ax_b.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax_b.legend(fontsize=8, frameon=True, framealpha=0.9,
                edgecolor="#bbbbbb", loc="lower left")


def subtitle(args, ms):
    if args.energy == "split":
        return ("split battery: $E_{each}=E_{\\max}/K$ = %s/K J, "
                "%d instances/cell"
                % ("{:,}".format(int(getattr(ms.MP, "Emax", 50000))),
                   args.instances))
    return ("fixed battery: $E_{each}$ = %s J per UAV (total energy GROWS "
            "with K), %d instances/cell"
            % ("{:,}".format(int(getattr(ms.MP, "Emax_each", 50000))),
               args.instances))


def draw(rows, ms, args):
    os.makedirs(args.out_dir, exist_ok=True)
    sub = subtitle(args, ms)

    for M in args.M:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.2))
        panel_pair(rows, M, ms, args, ax_a, ax_b)
        fig.suptitle(sub, fontsize=9, color="#555555", y=0.005)
        fig.tight_layout()
        out = os.path.join(args.out_dir, "fleet_vs_K_M%d_%s.png"
                           % (M, args.energy))
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  wrote %s" % out)

    n = len(args.M)
    fig, axes = plt.subplots(n, 2, figsize=(13.5, 5.0 * n), squeeze=False)
    for i, M in enumerate(args.M):
        panel_pair(rows, M, ms, args, axes[i][0], axes[i][1])
    fig.suptitle("Objective and coverage vs fleet size\n%s" % sub,
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = os.path.join(args.out_dir, "fleet_vs_K_ALL_%s.png" % args.energy)
    fig.savefig(out, dpi=args.grid_dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("  wrote %s" % out)


# --------------------------------------------------------------------------


# ==========================================================================
# SECTION 3 -- the five CSV-driven paper figures
# ==========================================================================
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


PAPER_FIGURES = {"kstar": fig_kstar, "reach": fig_reach, "phi": fig_phi,
                 "pareto": fig_pareto, "predict": fig_predict}


# ==========================================================================
# driver
# ==========================================================================
def ns(**kw):
    return argparse.Namespace(**kw)


def section_traj(a):
    print("\n" + "=" * 72)
    print("SECTION 1/3  example trajectories  (%d M x %d K = %d figures)"
          % (len(a.traj_m), len(a.traj_k), len(a.traj_m) * len(a.traj_k)))
    print("=" * 72)
    try:
        import compare_baseline  # noqa: F401
        import sa_routes  # noqa: F401
    except Exception as exc:
        print("  [SKIP] cannot import the solver modules (%r)." % exc)
        print("         Run from the repo root with the venv active.")
        return False
    sub = ns(batch_m=a.traj_m, batch_k=a.traj_k, Emax=a.Emax, iters=a.iters,
             instance_index=a.instance_index, meta_seed=a.meta_seed,
             instances=a.traj_instances, raw_instance_seed=None, sa_seed=0,
             out_dir=os.path.join(a.out_dir, a.traj_subdir),
             overwrite=a.overwrite, contact_sheet=not a.no_contact_sheet,
             sheet_dpi=a.sheet_dpi, dpi=a.dpi, ee_is_total=False,
             arrows=False, labels=not a.no_labels,
             footnote=not a.no_footnote,
             skip_greedy_check=not a.traj_greedy_check, seed=a.meta_seed,
             out=None, _tcd=None)
    batch(sub)
    return True


def section_fleet(a):
    print("\n" + "=" * 72)
    print("SECTION 2/3  objective + coverage vs K  (%d M x %d K, energy=%s)"
          % (len(a.fleet_m), len(a.fleet_k), a.energy))
    print("=" * 72)
    try:
        ms = load_solver()
    except SystemExit as exc:
        print("  [SKIP] %s" % exc)
        return False
    out_dir = os.path.join(a.out_dir, a.fleet_subdir)
    sub = ns(M=a.fleet_m, K=a.fleet_k, energy=a.energy,
             instances=a.fleet_instances, seed=a.fleet_seed,
             policy_fn=a.policy_fn,
             coverage_baselines=a.coverage_baselines, out_dir=out_dir,
             csv=None, dpi=a.dpi, grid_dpi=a.grid_dpi,
             models_dir=a.models_dir, ckpt_pattern=a.ckpt_pattern,
             model_seed=a.model_seed, device=a.device,
             use_postprocess=not a.no_postprocess)
    print("  energy convention: %s" % a.energy.upper())
    rows = collect(ms, sub)
    if not rows:
        print("  [SKIP] no results collected")
        return False
    write_csv(rows, os.path.join(out_dir, "fleet_vs_K_%s.csv" % a.energy))
    draw(rows, ms, sub)
    return True


def section_paper(a):
    print("\n" + "=" * 72)
    print("SECTION 3/3  paper figures from existing CSVs")
    print("=" * 72)
    out_dir = os.path.join(a.out_dir, a.paper_subdir)
    sub = ns(data_root=a.data_root, out=None, out_dir=out_dir, dpi=a.dpi)
    any_ok = False
    for name, fn in PAPER_FIGURES.items():
        print("[%s]" % name)
        try:
            fn(sub)
            any_ok = True
        except Exception as exc:
            print("  [FAIL] %s: %r" % (name, exc))
    return any_ok


SECTIONS = {"traj": section_traj, "fleet": section_fleet,
            "paper": section_paper}


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="run every section (same as --only traj fleet paper)")
    p.add_argument("--only", nargs="+", choices=list(SECTIONS), default=None)
    p.add_argument("--out-dir", default="figs", help="root output directory")
    p.add_argument("--data-root", default=".",
                   help="directory holding kstar_sa/, phi_measure/, ...")
    p.add_argument("--dpi", type=int, default=300)

    g = p.add_argument_group("section 1: trajectories")
    g.add_argument("--traj-m", type=int, nargs="+",
                   default=[50, 60, 80, 100, 120, 150, 200])
    g.add_argument("--traj-k", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 8])
    g.add_argument("--Emax", type=float, default=50000.0)
    g.add_argument("--iters", type=int, default=2000)
    g.add_argument("--instance-index", type=int, default=0)
    g.add_argument("--meta-seed", type=int, default=2025)
    g.add_argument("--traj-instances", type=int, default=30,
                   help="population size the instance index refers to")
    g.add_argument("--traj-subdir", default="trajectories")
    g.add_argument("--overwrite", action="store_true",
                   help="redraw trajectory cells whose PNG already exists")
    g.add_argument("--no-labels", action="store_true",
                   help="omit per-node priority-weight labels")
    g.add_argument("--no-footnote", action="store_true")
    g.add_argument("--no-contact-sheet", action="store_true")
    g.add_argument("--sheet-dpi", type=int, default=150)
    g.add_argument("--traj-greedy-check", action="store_true",
                   help="also compare each cell against greedy_init (SLOW)")

    g = p.add_argument_group("section 2: fleet vs K")
    g.add_argument("--fleet-m", type=int, nargs="+",
                   default=[50, 60, 80, 100, 120, 150, 200])
    g.add_argument("--fleet-k", type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 8])
    g.add_argument("--energy", choices=["split", "fixed"], default="split")
    g.add_argument("--fleet-instances", type=int, default=200)
    g.add_argument("--fleet-seed", type=int, default=42)
    g.add_argument("--policy-fn", default=None)
    g.add_argument("--coverage-baselines", action="store_true")
    g.add_argument("--models-dir", default="models_multi_uav")
    g.add_argument("--ckpt-pattern",
                   default="fleet_M{M}_K{K}_split_seed{seed}.pt")
    g.add_argument("--model-seed", type=int, default=42,
                   help="trained-model seed; only 7, 42, 123 exist")
    g.add_argument("--device", default="cpu",
                   help="cpu or cuda, for the policy rollout")
    g.add_argument("--no-postprocess", action="store_true",
                   help="evaluate the raw MLP rollout without post-processing "
                        "(default matches the banked rollout+pp numbers)")
    g.add_argument("--fleet-subdir", default="fleet_vs_K")
    g.add_argument("--grid-dpi", type=int, default=140)

    g = p.add_argument_group("section 3: paper figures")
    g.add_argument("--paper-subdir", default="paper")

    a = p.parse_args()
    order = ["traj", "fleet", "paper"]
    want = order if (a.all or not a.only) else [s for s in order
                                                if s in a.only]
    if not a.all and not a.only:
        print("  (no --all or --only given: running every section)")

    t0 = time.time()
    status = {}
    for name in want:
        try:
            status[name] = bool(SECTIONS[name](a))
        except KeyboardInterrupt:
            print("\n  interrupted during section '%s'" % name)
            status[name] = False
            break
        except Exception as exc:
            print("  [FAIL] section '%s': %r" % (name, exc))
            status[name] = False

    print("\n" + "=" * 72)
    print("SUMMARY  (%.1f s)" % (time.time() - t0))
    for name in want:
        print("  %-6s %s" % (name, "ok" if status.get(name) else "FAILED/SKIPPED"))
    print("  output root: %s" % os.path.abspath(a.out_dir))
    print("=" * 72)
    sys.exit(0 if all(status.get(n) for n in want) else 1)


if __name__ == "__main__":
    main()