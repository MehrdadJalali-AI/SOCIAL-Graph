"""Static diverse sampling: k-center greedy over the embedding, seeded with the shared initial design,
selects the whole budget up front (no adaptation to observed values)."""

from __future__ import annotations

import numpy as np

from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial


def k_center_greedy(X: np.ndarray, seeds: np.ndarray, k: int) -> np.ndarray:
    """Pick ``k`` further points, each maximising its distance to the current set (ties: lowest index)."""
    dist = np.full(len(X), np.inf)
    for s in seeds:
        dist = np.minimum(dist, ((X - X[s]) ** 2).sum(axis=1))
    chosen = []
    for _ in range(k):
        j = int(np.argmax(dist))
        chosen.append(j)
        dist = np.minimum(dist, ((X - X[j]) ** 2).sum(axis=1))
    return np.asarray(chosen, dtype=np.int64)


class StaticDiverse(Method):
    name = "static_diverse"

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        for step, j in enumerate(k_center_greedy(prob.X, init, oracle.remaining), start=1):
            oracle(int(j), step=step, agent=0, kind="acq")
        return oracle
