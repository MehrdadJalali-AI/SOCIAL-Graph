"""Greedy best-first graph walk.

Each step evaluates a random unevaluated neighbour of the best evaluated MOF that still has one
(evaluated MOFs are tried in order of objective value). If no evaluated MOF has an unevaluated
neighbour, it restarts at a uniformly random unevaluated MOF.
"""

from __future__ import annotations

import numpy as np

from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng


class GreedyWalk(Method):
    name = "greedy_walk"

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        exhausted: set[int] = set()  # evaluated MOFs whose neighbourhood is fully evaluated
        step = 0
        while not oracle.exhausted():
            step += 1
            order = np.asarray(oracle.order)[np.lexsort((np.asarray(oracle.order), np.asarray(oracle.values)))]
            choice, kind = None, "restart"
            for b in order:
                if int(b) in exhausted:
                    continue
                nb = prob.topology.neighbors(int(b))
                free = nb[~oracle.evaluated[nb]]
                if len(free):
                    choice, kind = int(rng.choice(free)), "walk"
                    break
                exhausted.add(int(b))
            if choice is None:
                choice = int(rng.choice(oracle.unevaluated()))
            oracle(choice, step=step, agent=0, kind=kind)
        return oracle
