"""Budgeted lookup oracle (port of SOCIAL-OPTIMIZATION's ``BudgetedObjective`` to a finite MOF set).

Every method gets the same oracle: it counts evaluations, refuses to exceed the budget, forbids
re-evaluating a MOF, and records the evaluation trace with per-evaluation metadata.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np


class BudgetExceeded(RuntimeError):
    pass


class AlreadyEvaluated(RuntimeError):
    pass


class BudgetedOracle:
    def __init__(self, f: np.ndarray, budget: int):
        self.f = np.asarray(f, dtype=np.float64)
        self.budget = int(budget)
        self.evaluated = np.zeros(len(self.f), dtype=bool)
        self.order: list[int] = []
        self.values: list[float] = []
        self.meta: list[dict[str, Any]] = []
        self.lookup_time = 0.0

    def __call__(self, idx: int, **meta: Any) -> float:
        idx = int(idx)
        if len(self.order) >= self.budget:
            raise BudgetExceeded(f"budget {self.budget} exhausted")
        if self.evaluated[idx]:
            raise AlreadyEvaluated(f"MOF {idx} already evaluated")
        t = time.perf_counter()
        val = float(self.f[idx])
        self.lookup_time += time.perf_counter() - t
        self.evaluated[idx] = True
        self.order.append(idx)
        self.values.append(val)
        self.meta.append(meta)
        return val

    @property
    def used(self) -> int:
        return len(self.order)

    @property
    def remaining(self) -> int:
        return self.budget - len(self.order)

    def exhausted(self) -> bool:
        return self.remaining <= 0

    def best(self) -> tuple[int, float]:
        i = int(np.argmin(self.values))
        return self.order[i], self.values[i]

    def unevaluated(self) -> np.ndarray:
        return np.flatnonzero(~self.evaluated)
