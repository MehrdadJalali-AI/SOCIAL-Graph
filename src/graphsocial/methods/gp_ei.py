"""GP-EI: Gaussian process on the embedding, refit every step, expected improvement over all unevaluated MOFs.

Kernel: Constant x Matern(nu=2.5, ARD: one length scale per embedding dimension) + White; normalised
targets; 5 optimiser restarts; EI for minimisation with xi = 0.01 and the incumbent = best observed f.
Each refit starts the optimiser from the previous step's fitted hyperparameters (plus 5 random restarts).
"""

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


def expected_improvement(mu: np.ndarray, sd: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    """EI for minimisation: E[max(0, best - xi - Y)], Y ~ N(mu, sd^2). Large when mu is below the incumbent."""
    sd = np.maximum(sd, 1e-12)
    z = (best - mu - xi) / sd
    return (best - mu - xi) * norm.cdf(z) + sd * norm.pdf(z)


class GPEI(Method):
    name = "gp_ei"

    def __init__(self, n_restarts: int = 5, xi: float = 0.01, ard: bool = True, **params):
        super().__init__(n_restarts=n_restarts, xi=xi, ard=ard, **params)

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        p = self.params
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        evaluate_initial(oracle, init)
        D = prob.X.shape[1]
        ls = np.ones(D) if p["ard"] else 1.0
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=ls, length_scale_bounds=(1e-2, 1e3), nu=2.5)
                  + WhiteKernel(1e-3, (1e-8, 1e0)))
        step = 0
        while not oracle.exhausted():
            step += 1
            Xtr = prob.X[oracle.order]
            ytr = np.asarray(oracle.values)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=p["n_restarts"],
                                          random_state=int(rng.integers(2**31)))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Xtr, ytr)
            kernel = gp.kernel_  # warm start for the next refit
            cand = oracle.unevaluated()
            mu, sd = gp.predict(prob.X[cand], return_std=True)
            ei = expected_improvement(mu, sd, ytr.min(), p["xi"])
            oracle(int(cand[int(np.argmax(ei))]), step=step, agent=0, kind="acq")
        return oracle
