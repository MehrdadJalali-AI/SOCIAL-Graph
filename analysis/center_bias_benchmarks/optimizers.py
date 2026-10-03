"""Continuous optimizers on the unit box with a strict evaluation budget (Problem raises BudgetExhausted).

SOCIAL follows Algorithm 1 and Table 2 of Jalali et al., Appl. Soft Comput. 2026, 194, 114914, unchanged:
Watts-Strogatz agent graph (K = 4, p = 0.3), betweenness centrality, log-ratio influence, schedules
alpha_t, beta_t decreasing and gamma_t, delta_t increasing, synchronization with the population mean
(omega = 0.05), elite memory, mutation of worse-than-median agents (p_m = 0.1, s_t = 0.1(1 - t/T) + 0.01) and the
periodic perturbation (every 10 iterations, probability 0.05, U(-0.5, 0.5)^D). Only the population size is set
for the budget (all population methods use the same size).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import igraph as ig
import numpy as np

class BudgetExhausted(Exception):
    pass


Problem = object  # any object with .dim, .budget and __call__(x) on the unit box


def _evaluate(prob: Problem, X: np.ndarray) -> np.ndarray:
    return np.array([prob(x) for x in X])


def social(prob: Problem, seed: int, pop: int = 20, alpha=0.4, beta=0.4, gamma=0.2, delta=0.2, p_m=0.1,
           omega=0.05, K=4, p_rewire=0.3, use_centrality=True, use_neighbors=True) -> None:
    rng = np.random.default_rng(seed)
    D = prob.dim
    T = prob.budget // pop
    ig.set_random_number_generator(random.Random(seed))
    g = ig.Graph.Watts_Strogatz(dim=1, size=pop, nei=K // 2, p=p_rewire)
    nbrs = [np.array(g.neighbors(i)) for i in range(pop)]
    bc = np.asarray(g.betweenness(), dtype=float)
    c = bc / bc.max() if bc.max() > 0 else np.zeros(pop)
    if not use_centrality:
        c = np.zeros(pop)
    x = rng.random((pop, D))
    try:
        f = _evaluate(prob, x)
        elite_x, elite_f = x[np.argmin(f)].copy(), f.min()
        for t in range(1, T):
            tau = t / T
            a_t, b_t, g_t, d_t = alpha * (1 - tau), beta * (1 - tau), gamma * tau, delta * tau
            gbest = x[np.argmin(f)].copy()
            xmean = x.mean(axis=0)
            infl = 1.0 - np.log1p(np.abs(f)) / (np.log1p(np.max(f)) + 1e-6)
            med = np.median(f)
            new = np.empty_like(x)
            for i in range(pop):
                nb = nbrs[i]
                if use_neighbors and len(nb):
                    w = a_t * c[nb] + b_t * infl[nb]
                    w = w / w.sum() if w.sum() > 0 else np.full(len(nb), 1 / len(nb))
                    xn = w @ x[nb]
                    xi = (1 - a_t - b_t - g_t - d_t) * x[i] + (a_t + b_t) * xn + g_t * gbest + d_t * elite_x
                else:
                    xi = (1 - g_t - d_t) * x[i] + g_t * gbest + d_t * elite_x
                xi = (1 - omega) * xi + omega * xmean
                xi = np.clip(xi, 0, 1)
                if f[i] > med and rng.random() < p_m:
                    s_t = 0.1 * (1 - tau) + 0.01
                    xi = np.clip(xi + rng.uniform(-s_t, s_t, D), 0, 1)
                if t % 10 == 0 and rng.random() < 0.05:
                    xi = np.clip(xi + rng.uniform(-0.5, 0.5, D), 0, 1)
                new[i] = xi
            x = new
            f = _evaluate(prob, x)
            if f.min() < elite_f:
                elite_f, elite_x = f.min(), x[np.argmin(f)].copy()
    except BudgetExhausted:
        pass


def de(prob: Problem, seed: int, pop: int = 20, F=0.5, CR=0.9) -> None:
    rng = np.random.default_rng(seed)
    D = prob.dim
    x = rng.random((pop, D))
    try:
        f = _evaluate(prob, x)
        while True:
            for i in range(pop):
                r1, r2, r3 = rng.choice([j for j in range(pop) if j != i], 3, replace=False)
                v = np.clip(x[r1] + F * (x[r2] - x[r3]), 0, 1)
                cross = rng.random(D) < CR
                cross[rng.integers(D)] = True
                u = np.where(cross, v, x[i])
                fu = prob(u)
                if fu <= f[i]:
                    x[i], f[i] = u, fu
    except BudgetExhausted:
        pass


def pso(prob: Problem, seed: int, pop: int = 20, w=0.729, c1=1.494, c2=1.494) -> None:
    rng = np.random.default_rng(seed)
    D = prob.dim
    x = rng.random((pop, D))
    v = np.zeros_like(x)
    try:
        f = _evaluate(prob, x)
        pb, pf = x.copy(), f.copy()
        while True:
            gb = pb[np.argmin(pf)]
            for i in range(pop):
                v[i] = w * v[i] + c1 * rng.random(D) * (pb[i] - x[i]) + c2 * rng.random(D) * (gb - x[i])
                x[i] = np.clip(x[i] + v[i], 0, 1)
                fi = prob(x[i])
                if fi < pf[i]:
                    pb[i], pf[i] = x[i].copy(), fi
    except BudgetExhausted:
        pass


def cmaes(prob: Problem, seed: int, sigma0: float = 0.3) -> None:
    import cma

    es = cma.CMAEvolutionStrategy(np.full(prob.dim, 0.5), sigma0,
                                  {"bounds": [0, 1], "seed": seed + 1, "verbose": -9, "maxfevals": prob.budget})
    try:
        while not es.stop():
            X = es.ask()
            es.tell(X, [prob(xx) for xx in X])
        while True:  # restart with a larger population if CMA-ES stops early (IPOP-style), within the budget
            es = cma.CMAEvolutionStrategy(np.random.default_rng(seed + prob.evals).random(prob.dim), sigma0,
                                          {"bounds": [0, 1], "seed": seed + prob.evals, "verbose": -9,
                                           "popsize": 2 * es.popsize})
            while not es.stop():
                X = es.ask()
                es.tell(X, [prob(xx) for xx in X])
    except BudgetExhausted:
        pass


def random_search(prob: Problem, seed: int) -> None:
    rng = np.random.default_rng(seed)
    try:
        while True:
            prob(rng.random(prob.dim))
    except BudgetExhausted:
        pass


METHODS = {"SOCIAL": social, "CMA-ES": cmaes, "DE": de, "PSO": pso, "random": random_search}


SOCIAL_REPO = Path(__file__).resolve().parents[2] / "data" / "raw" / "SOCIAL-OPTIMIZATION"


def social_repo(prob: Problem, seed: int, num_nodes: int | None = None) -> None:
    """The authors' published SOCIAL implementation (SOCIAL-OPTIMIZATION, social/optimizer.py), run unchanged with
    its default configuration (preset 'SOCIAL_balanced'); only the budget and, optionally, the population size are
    set. Bounds are the unit box."""
    import contextlib
    import io

    if str(SOCIAL_REPO) not in sys.path:
        sys.path.insert(0, str(SOCIAL_REPO))
    from social.budget import BudgetedObjective
    from social.config import Config
    from social.optimizer import SOCIALOptimizer

    cfg = Config(MAX_EVALS=prob.budget, SHOW_PROGRESS=False)
    if num_nodes is not None:
        cfg.NUM_NODES = num_nodes
        cfg.SEED_SET_SIZE = max(1, int(0.1 * num_nodes))
    opt = SOCIALOptimizer(cfg, rng=np.random.default_rng(seed))
    obj = BudgetedObjective(lambda x: prob(np.asarray(x)), prob.budget)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            opt.optimize(obj, [0.0, 1.0], prob.dim, seed=seed)
    except (BudgetExhausted, RuntimeError):
        pass


METHODS["SOCIAL (repo, 60 agents)"] = social_repo
METHODS["SOCIAL (repo, 20 agents)"] = lambda prob, seed: social_repo(prob, seed, num_nodes=20)
METHODS["SOCIAL (paper Alg. 1)"] = METHODS.pop("SOCIAL")
