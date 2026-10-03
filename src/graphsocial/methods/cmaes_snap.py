"""CMA-ES with the snap operator (revision baseline).

Uses the `cma` package with default settings except popsize = P (10) and sigma0 = 0.3 x mean range of the
embedding; bounds are the embedding's bounding box. The first generation is the shared initial design: its
embeddings are injected into CMA-ES and the initial MOFs themselves are evaluated. In later generations each
candidate snaps to the nearest unevaluated MOF, which is evaluated, and CMA-ES is told the snapped positions
(as DE, PSO and GA move to the snapped MOF). T = floor(B / P) generations including the initial one; the
remainder goes to the unevaluated MOFs nearest the best-so-far. If a default CMA-ES stopping criterion fires
before T generations, the remaining budget also goes to the MOFs nearest the best-so-far (counted in the log).
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng, spend_leftover


class CMAESSnap(Method):
    name = "cmaes"
    population = True

    def __init__(self, sigma_frac: float = 0.3, **params):
        super().__init__(sigma_frac=sigma_frac, **params)

    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        import cma

        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        snap = Snapper(prob.X)
        lo, hi = snap.bounds()
        P = len(init)
        T = budget // P
        sigma0 = self.params["sigma_frac"] * float(np.mean(hi - lo))
        X0 = prob.X[init]
        f0 = evaluate_initial(oracle, init)
        best_x, best_f = X0[np.argmin(f0)].copy(), float(f0.min())
        opts = {"popsize": P, "bounds": [lo.tolist(), hi.tolist()], "seed": int(rng.integers(1, 2**31 - 1)),
                "verbose": -9}
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            es = cma.CMAEvolutionStrategy(X0.mean(axis=0), sigma0, opts)
            es.inject(list(X0), force=True)
            es.ask()  # generation 0 = the injected initial design (already evaluated)
            es.tell(list(X0), list(f0))
            stopped_early = False
            for t in range(1, T):
                if es.stop():
                    stopped_early = True
                    break
                cand = es.ask()
                snapped, fs = [], []
                for i, x in enumerate(cand):
                    m = snap.nearest(np.clip(x, lo, hi), oracle.evaluated)
                    fi = oracle(m, step=t, agent=i, kind="update")
                    snapped.append(prob.X[m].copy())
                    fs.append(fi)
                    if fi < best_f:
                        best_f, best_x = fi, prob.X[m].copy()
                es.tell(snapped, fs)
        self.stopped_early = stopped_early
        spend_leftover(oracle, snap, best_x, T)
        return oracle
