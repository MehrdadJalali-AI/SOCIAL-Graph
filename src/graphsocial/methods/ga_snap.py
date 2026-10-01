"""Genetic algorithm with the snap operator: binary tournament selection, uniform crossover, Gaussian
mutation (per-gene probability 1/D, sigma = 0.1 * range), generational replacement with one elite."""

from __future__ import annotations

import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng, spend_leftover


class GASnap(Method):
    name = "ga"
    population = True

    def __init__(self, tournament: int = 2, sigma: float = 0.1, **params):
        super().__init__(tournament=tournament, sigma=sigma, **params)

    def _select(self, f: np.ndarray, rng: np.random.Generator) -> int:
        cand = rng.choice(len(f), size=self.params["tournament"], replace=False)
        return int(cand[np.argmin(f[cand])])

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        snap = Snapper(prob.X)
        lo, hi = snap.bounds()
        span = hi - lo
        P, D = len(init), prob.X.shape[1]
        T = budget // P
        cur = init.astype(np.int64).copy()
        f = evaluate_initial(oracle, cur)
        x = prob.X[cur].copy()
        best_x, best_f = x[np.argmin(f)].copy(), f.min()
        for t in range(1, T):
            child_x, child_f = np.empty_like(x), np.empty(P)
            for i in range(P):
                a, b = self._select(f, rng), self._select(f, rng)
                mask = rng.random(D) < 0.5
                c = np.where(mask, x[a], x[b])
                mut = rng.random(D) < 1.0 / D
                c = np.clip(c + mut * rng.normal(0.0, self.params["sigma"] * span), lo, hi)
                m = snap.nearest(c, oracle.evaluated)
                child_f[i] = oracle(m, step=t, agent=i, kind="update")
                child_x[i] = prob.X[m]
            elite = int(np.argmin(f))
            worst_child = int(np.argmax(child_f))
            if f[elite] < child_f[worst_child]:
                child_x[worst_child], child_f[worst_child] = x[elite], f[elite]
            x, f = child_x, child_f
            if f.min() < best_f:
                best_f, best_x = f.min(), x[np.argmin(f)].copy()
        spend_leftover(oracle, snap, best_x, T)
        return oracle
