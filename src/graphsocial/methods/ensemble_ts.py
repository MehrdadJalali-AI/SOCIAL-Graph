"""Ensemble Thompson sampling with a bootstrap ensemble of 10 random forests on the embedding.

Each step samples one ensemble member, fits it on its own bootstrap resample of the evaluated data, and
evaluates the unevaluated MOF with the best (lowest) predicted objective. Only the sampled member is
needed per step, so only it is fitted; that is equivalent in distribution to refitting all ten.
(Deviation from a GNN ensemble, chosen for CPU feasibility.)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng


class EnsembleTS(Method):
    name = "ensemble_ts"

    def __init__(self, n_members: int = 10, n_estimators: int = 100, **params):
        super().__init__(n_members=n_members, n_estimators=n_estimators, **params)

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        member_seeds = rng.integers(2**31, size=self.params["n_members"])
        step = 0
        while not oracle.exhausted():
            step += 1
            m = int(rng.integers(self.params["n_members"]))
            mrng = np.random.default_rng([int(member_seeds[m]), step])
            n = oracle.used
            boot = mrng.integers(0, n, size=n)
            Xtr = prob.X[np.asarray(oracle.order)[boot]]
            ytr = np.asarray(oracle.values)[boot]
            rf = RandomForestRegressor(n_estimators=self.params["n_estimators"],
                                       random_state=int(mrng.integers(2**31)), n_jobs=1)
            rf.fit(Xtr, ytr)
            cand = oracle.unevaluated()
            pred = rf.predict(prob.X[cand])
            oracle(int(cand[int(np.argmin(pred))]), step=step, agent=m, kind="acq")
        return oracle
