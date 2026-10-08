"""liu_mpga.py -- baseline: the MPGA of Liu, Guo, Li, Song, "AoI-Minimal Task Assignment and Trajectory
Optimization in Multi-UAV-Assisted IoT Networks", IEEE IoT-J 9(21), 2022, adapted to this paper's model.

What is kept from Liu (Sec. III-B, Alg. 3): the chromosome (one permutation of all sensors cut into one
segment per UAV by breakpoints), N_p populations of N_0 individuals run for N_iter generations, the
normalised fitness F with psi, roulette selection plus selection of the population / global best as a
parent ("exploration-exploitation"), the sequential (order, OX1) crossover of their Fig. 2, the swap /
inversion / slide mutations, the balancing breakpoint-update operator, per-population p_cr ~ U[0.5,0.95]
and p_mu ~ U[0.05,0.3], and their improved global K-means as the seed individual.

What is changed (see README.md for the reasons): the IPT, data sharing and user-distribution stage are
dropped; the fitness is this paper's weighted age-at-collection J of the resulting repeating patrol,
computed by the planner-side surrogate of cyclic_sched (Ctx.uav_J), instead of their one-mission eq. (22);
the energy constraint (17) is enforced by cutting each UAV's segment into energy-feasible sorties
(cyclic_sched.cut_sortie, the same rule as cluster_patrol / C4) instead of by fitness = 0 or task reduction.

Deployment is the cluster_patrol / C4 pattern: segment k is UAV k's territory and visit sequence; at each
launch the UAV flies the next contiguous run of its sequence that fits U/K.
"""
from __future__ import annotations
import os, sys; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", d) for d in ("sim",)]
import time
from typing import List
import numpy as np
from rr_planner import tour_order
from cyclic_sched import Ctx, cut_sortie

# Liu Sec. IV: N_p = 8, N_0 = 60, N_iter = 100, p_cr ~ U[0.5,0.95], p_mu ~ U[0.05,0.3].
# The rest are NOT given in the paper; values agreed before implementation, listed in README.md.
CFG = dict(Np=8, N0=60, Niter=100, pcr=(0.5, 0.95), pmu=(0.05, 0.3),
           p_local_best=0.1, p_global_best=0.1, psi=1e-9, bp_shift=(1, 3), seed=0)


# ------------------------------------------------------------------------------------------------
# seed individual: Liu's improved global K-means (steps a-g), IPT replaced by the depot
# ------------------------------------------------------------------------------------------------
def improved_kmeans(pos, home, K, iters=100):
    """Labels in 0..K-1. Liu's Dr_i as printed sums d(i,j)/sum_l d(i,l), which is identically 1
    (a typo); we use Dr_i = sum_j d(i,j), the reading under which step b picks the most central
    target (any normalisation leaves the argmin unchanged). After the K seeds, Lloyd iterations
    with their depot-pulled centre update until the labels stop changing (not stated in the paper)."""
    X = np.asarray(pos, float); home = np.asarray(home, float)
    M = len(X); K = min(K, M)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    Dr = D.sum(1)

    def centre(members):
        return (X[members].sum(0) + home) / (len(members) + 1)        # Cc = (sum x + l_s)/(N+1)

    chosen = [int(np.argmin(Dr))]                                      # step b: most central target seeds
    lab = np.zeros(M, int); C = [centre(np.arange(M))]                 # cluster 1, which takes every target
    while len(C) < K:                                                  # steps c-g
        Dc = Dr / np.linalg.norm(X[:, None] - np.array(C)[None], axis=2).sum(1)
        Dc[chosen] = np.inf
        i = int(np.argmin(Dc)); chosen.append(i); C.append(X[i].copy())
        lab = np.argmin(np.linalg.norm(X[:, None] - np.array(C)[None], axis=2), axis=1)
        C = [centre(np.where(lab == k)[0]) if np.any(lab == k) else C[k] for k in range(len(C))]
    for _ in range(iters):
        new = np.argmin(np.linalg.norm(X[:, None] - np.array(C)[None], axis=2), axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        C = [centre(np.where(lab == k)[0]) if np.any(lab == k) else C[k] for k in range(K)]
    return lab


def seed_individual(ctx: Ctx):
    lab = improved_kmeans(ctx.pos, ctx.home, ctx.K)
    perm, cuts = [], []
    for k in range(ctx.K):
        idx = np.where(lab == k)[0]
        perm += [int(x) for x in (idx[tour_order(ctx.pos[idx], ctx.home)] if len(idx) else idx)]
        cuts.append(len(perm))
    return perm, cuts[:-1]


# ------------------------------------------------------------------------------------------------
# chromosome = (perm: list of all M sensor ids, cuts: K-1 non-decreasing breakpoints in [0, M])
# ------------------------------------------------------------------------------------------------
def segments(perm, cuts):
    b = [0] + list(cuts) + [len(perm)]
    return [perm[b[k]:b[k + 1]] for k in range(len(b) - 1)]


def ox1(p1, p2, i, j):
    """Order crossover (Liu Fig. 2): child keeps p1[i..j], the other positions are filled from
    p2 read cyclically from position j+1, skipping genes already present, starting at j+1."""
    n = len(p1); child = [None] * n
    child[i:j + 1] = p1[i:j + 1]
    keep = set(p1[i:j + 1])
    fill = [p2[(j + 1 + t) % n] for t in range(n) if p2[(j + 1 + t) % n] not in keep]
    for t, g in enumerate(fill):
        child[(j + 1 + t) % n] = g
    return child


def mutate(perm, rng):
    """One of Liu's three mutations, chosen uniformly. 'Slide translation' = move a random block
    to a random new position (the paper names the operator without defining it)."""
    n = len(perm); perm = list(perm)
    if n < 2:
        return perm
    op = rng.integers(3)
    a, b = sorted(rng.choice(n, 2, replace=False))
    if op == 0:
        perm[a], perm[b] = perm[b], perm[a]
    elif op == 1:
        perm[a:b + 1] = perm[a:b + 1][::-1]
    else:
        blk = perm[a:b + 1]; rest = perm[:a] + perm[b + 1:]
        at = int(rng.integers(len(rest) + 1))
        perm = rest[:at] + blk + rest[at:]
    return perm


def tour_len(seq, ctx: Ctx):
    if not seq:
        return 0.0
    D = ctx.D; M = ctx.M
    return D[M][seq[0]] + sum(D[a][b] for a, b in zip(seq, seq[1:])) + D[seq[-1]][M]


def breakpoint_update(perm, cuts, ctx: Ctx, rng, shift=(1, 3)):
    """Liu's balancing operator (form not given): move one random breakpoint by 1-3 genes so the
    segment with the longer closed depot tour hands genes to its neighbour."""
    if not cuts:
        return list(cuts)
    cuts = list(cuts); b = int(rng.integers(len(cuts)))
    segs = segments(perm, cuts)
    d = int(rng.integers(shift[0], shift[1] + 1))
    lo = cuts[b - 1] if b > 0 else 0
    hi = cuts[b + 1] if b + 1 < len(cuts) else len(perm)
    if tour_len(segs[b], ctx) > tour_len(segs[b + 1], ctx):
        cuts[b] = max(lo, cuts[b] - d)
    else:
        cuts[b] = min(hi, cuts[b] + d)
    return cuts


# ------------------------------------------------------------------------------------------------
# fitness: surrogate time-average weighted age of the repeating patrol (cyclic_sched.Ctx.uav_J)
# ------------------------------------------------------------------------------------------------
class Evaluator:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx; self.cache = {}; self.n_sim = 0

    def __call__(self, perm, cuts):
        J = 0.0
        for k, seg in enumerate(segments(perm, cuts)):
            key = (k, tuple(seg))
            if key not in self.cache:            # a segment's J depends only on (k, sequence)
                self.cache[key] = self.ctx.uav_J(list(seg), list(seg), k); self.n_sim += 1
            J += self.cache[key]
        return J / (self.ctx.H - self.ctx.burn)


def mpga(ctx: Ctx, cfg=CFG, log=None):
    """Returns (best_perm, best_cuts, best_J, info). best_J <= J(seed) by elitism."""
    rng = np.random.default_rng(cfg["seed"])
    M, K = ctx.M, ctx.K
    ev = Evaluator(ctx)

    def rand_ind():
        return [int(x) for x in rng.permutation(M)], sorted(int(x) for x in rng.integers(0, M + 1, K - 1))

    pops = [[rand_ind() for _ in range(cfg["N0"])] for _ in range(cfg["Np"])]
    seed = seed_individual(ctx)
    pops[0][0] = seed
    pcr = rng.uniform(*cfg["pcr"], cfg["Np"]); pmu = rng.uniform(*cfg["pmu"], cfg["Np"])
    J_seed = ev(*seed)
    fit = [[ev(*ind) for ind in pop] for pop in pops]
    best_loc = [min(zip(f, pop), key=lambda t: t[0]) for f, pop in zip(fit, pops)]
    best_glob = min(best_loc, key=lambda t: t[0])
    hist = [best_glob[0]]
    for gen in range(cfg["Niter"]):
        for i in range(cfg["Np"]):
            f = np.array(fit[i]); pop = pops[i]
            Fn = (f.max() - f + cfg["psi"]) / (f.max() - f.min() + cfg["psi"])      # Liu's F(.)
            pr = Fn / Fn.sum()

            def parent():
                u = rng.random()
                if u < cfg["p_local_best"]:
                    return best_loc[i][1]
                if u < cfg["p_local_best"] + cfg["p_global_best"]:
                    return best_glob[1]
                return pop[int(rng.choice(len(pop), p=pr))]

            new = [best_loc[i][1]]                                                   # elitism
            while len(new) < cfg["N0"]:
                (p1, c1), (p2, _) = parent(), parent()
                if rng.random() < pcr[i] and M > 1:
                    a, b = sorted(rng.choice(M, 2, replace=False))
                    child = ox1(p1, p2, int(a), int(b))
                else:
                    child = list(p1)
                cuts = list(c1)
                if rng.random() < pmu[i]:
                    child = mutate(child, rng)
                if rng.random() < pmu[i]:
                    cuts = breakpoint_update(child, cuts, ctx, rng, cfg["bp_shift"])
                new.append((child, cuts))
            pops[i] = new
            fit[i] = [ev(*ind) for ind in new]
            j = int(np.argmin(fit[i]))
            if fit[i][j] < best_loc[i][0]:
                best_loc[i] = (fit[i][j], new[j])
        best_glob = min(best_loc + [best_glob], key=lambda t: t[0])
        hist.append(best_glob[0])
        if log and (gen % 10 == 0 or gen == cfg["Niter"] - 1):
            log(f"    gen {gen}: J_sur={best_glob[0]:.4e} seed={J_seed:.4e} sims={ev.n_sim}")
    Jb, (perm, cuts) = best_glob
    return perm, cuts, Jb, dict(J_seed=J_seed, J_best=Jb, n_sim=ev.n_sim, hist=hist)


# ------------------------------------------------------------------------------------------------
# runtime planner (same bind_sim / cut_sortie pattern as cluster_patrol and C4)
# ------------------------------------------------------------------------------------------------
def build_liu_planner(cfg=CFG, seqs=None, verbose=False):
    """seqs: fixed per-UAV sequences instead of running MPGA (used by the harness self-test)."""
    st = {"seq": None, "ptr": None, "k": 0, "info": None}

    def bind_sim(sim):
        orig = sim._plan
        def _plan(k, *a, **kw):
            st["k"] = k
            return orig(k, *a, **kw)
        sim._plan = _plan
        st["K"] = sim.p.K

    def planner(req) -> List[int]:
        if req.start is not None:
            raise ValueError("liu_mpga is launch-time only (use --replan launch)")
        p = req.p
        if st["seq"] is None:
            t = time.time()
            ctx = Ctx(req.pos, req.home, req.weight_est, p, st["K"])
            if seqs is not None:
                st["seq"] = [list(s) for s in seqs]; info = {}
            else:
                perm, cuts, _, info = mpga(ctx, cfg, log=(print if verbose else None))
                st["seq"] = segments(perm, cuts)
            st["ctx"] = ctx; st["ptr"] = [0] * st["K"]
            st["info"] = dict(info, build_s=time.time() - t)
        ctx = st["ctx"]; k = st["k"] % st["K"]
        hov = p.e_hover(req.dwell_est)
        alone = p.e_fly(2.0 * np.asarray(ctx.dh)) + hov
        excl = np.zeros(ctx.M, bool) if req.excluded is None else np.asarray(req.excluded, bool)
        route, st["ptr"][k] = cut_sortie(st["seq"][k], st["ptr"][k], ctx.D, ctx.dh,
                                         hov.tolist(), alone.tolist(), excl.tolist(), req.E_usable, ctx.Pf_v)
        return route

    planner.bind_sim = bind_sim
    planner.state = st
    return planner
