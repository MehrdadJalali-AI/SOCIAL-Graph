"""Shared pieces for search methods: initial design, RNG streams and leftover handling."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem


def initial_design(n: int, size: int, seed: int) -> np.ndarray:
    """The P initial MOFs for a seed: uniform without replacement, identical for every method."""
    return np.random.default_rng([seed, 7919]).choice(n, size=size, replace=False)


def method_rng(seed: int, method: str) -> np.random.Generator:
    """Method-specific stream so the initial design is shared but method randomness is independent."""
    return np.random.default_rng([seed, sum(map(ord, method)), 104729])


def evaluate_initial(oracle: BudgetedOracle, init: np.ndarray) -> np.ndarray:
    vals = np.empty(len(init))
    for a, i in enumerate(init):
        vals[a] = oracle(int(i), step=0, agent=a, kind="init")
    return vals


def spend_leftover(oracle: BudgetedOracle, snap: Snapper, anchor: np.ndarray, step: int) -> None:
    """Remaining evaluations go to the unevaluated MOFs nearest to ``anchor`` (the elite)."""
    while not oracle.exhausted():
        oracle(snap.nearest(anchor, oracle.evaluated), step=step, agent=-1, kind="leftover")


class Method:
    name = "base"
    population = False

    def __init__(self, **params: Any):
        self.params = params

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        raise NotImplementedError
