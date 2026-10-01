"""GP-EI: Gaussian process (Matern nu=2.5 + WhiteKernel, normalised y) on the embedding, refit every
step, expected improvement (minimisation) over all unevaluated MOFs."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.0) -> np.ndarray:
    sd = np.maximum(sd, 1e-12)
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)


class GPEI(Method):
    name = "gp_ei"

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-3)
        step = 0
        while not oracle.exhausted():
            step += 1
            Xtr = prob.X[oracle.order]
            ytr = np.asarray(oracle.values)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=0,
                                          random_state=int(rng.integers(2**31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Xtr, ytr)
            cand = oracle.unevaluated()
            mu, sd = gp.predict(prob.X[cand], return_std=True)
            ei = expected_improvement(mu, sd, ytr.min())
            oracle(int(cand[int(np.argmax(ei))]), step=step, agent=0, kind="acq")
        return oracle
