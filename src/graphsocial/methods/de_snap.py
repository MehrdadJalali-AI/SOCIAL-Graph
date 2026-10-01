"""Differential evolution (DE/rand/1/bin, F=0.5, CR=0.9) in the embedding with the snap operator.

Population P from the shared initial design; T = floor(budget/P) generations including the initial one;
each trial vector snaps to the nearest unevaluated MOF and replaces its parent if not worse. Leftover
evaluations go to the MOFs nearest the best-so-far.
"""

from __future__ import annotations

import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng, spend_leftover


class DESnap(Method):
    name = "de"
    population = True

    def __init__(self, F: float = 0.5, CR: float = 0.9, **params):
        super().__init__(F=F, CR=CR, **params)

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        snap = Snapper(prob.X)
        lo, hi = snap.bounds()
        P, D = len(init), prob.X.shape[1]
        T = budget // P
        cur = init.astype(np.int64).copy()
        f = evaluate_initial(oracle, cur)
        x = prob.X[cur].copy()
        best_x, best_f = x[np.argmin(f)].copy(), f.min()
        for t in range(1, T):
            for i in range(P):
                others = [j for j in range(P) if j != i]
                r1, r2, r3 = rng.choice(others, size=3, replace=len(others) < 3)
                v = x[r1] + self.params["F"] * (x[r2] - x[r3])
                cross = rng.random(D) < self.params["CR"]
                cross[rng.integers(D)] = True
                u = np.clip(np.where(cross, v, x[i]), lo, hi)
                m = snap.nearest(u, oracle.evaluated)
                fu = oracle(m, step=t, agent=i, kind="update")
                if fu <= f[i]:
                    cur[i], f[i], x[i] = m, fu, prob.X[m]
                if fu < best_f:
                    best_f, best_x = fu, prob.X[m].copy()
        spend_leftover(oracle, snap, best_x, T)
        return oracle
