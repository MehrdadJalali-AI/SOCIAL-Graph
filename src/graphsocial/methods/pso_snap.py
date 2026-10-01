"""Particle swarm optimisation (w=0.729, c1=c2=1.494, global-best topology) with the snap operator.

After each velocity/position update a particle snaps to the nearest unevaluated MOF; its position becomes
that MOF's embedding. Same generation count and leftover rule as the other population methods.
"""

from __future__ import annotations

import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng, spend_leftover


class PSOSnap(Method):
    name = "pso"
    population = True

    def __init__(self, w: float = 0.729, c1: float = 1.494, c2: float = 1.494, **params):
        super().__init__(w=w, c1=c1, c2=c2, **params)

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        p = self.params
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        snap = Snapper(prob.X)
        lo, hi = snap.bounds()
        P, D = len(init), prob.X.shape[1]
        T = budget // P
        cur = init.astype(np.int64).copy()
        f = evaluate_initial(oracle, cur)
        x = prob.X[cur].copy()
        v = np.zeros_like(x)
        pbest_x, pbest_f = x.copy(), f.copy()
        g = int(np.argmin(pbest_f))
        gbest_x, gbest_f = pbest_x[g].copy(), pbest_f[g]
        for t in range(1, T):
            for i in range(P):
                r1, r2 = rng.random(D), rng.random(D)
                v[i] = p["w"] * v[i] + p["c1"] * r1 * (pbest_x[i] - x[i]) + p["c2"] * r2 * (gbest_x - x[i])
                target = np.clip(x[i] + v[i], lo, hi)
                m = snap.nearest(target, oracle.evaluated)
                fi = oracle(m, step=t, agent=i, kind="update")
                x[i] = prob.X[m]
                if fi < pbest_f[i]:
                    pbest_f[i], pbest_x[i] = fi, x[i].copy()
                if fi < gbest_f:
                    gbest_f, gbest_x = fi, x[i].copy()
        spend_leftover(oracle, snap, gbest_x, T)
        return oracle
