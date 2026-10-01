"""Random search: after the shared initial design, evaluate uniformly random unevaluated MOFs."""

from __future__ import annotations

import numpy as np

from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng


class RandomSearch(Method):
    name = "random"

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        rest = rng.permutation(oracle.unevaluated())
        for step, i in enumerate(rest[: oracle.remaining], start=1):
            oracle(int(i), step=step, agent=0, kind="acq")
        return oracle
