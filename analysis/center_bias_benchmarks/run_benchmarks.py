"""Center-bias test on the 23 classic benchmark functions used in the SOCIAL paper (definitions and bounds taken
unchanged from the SOCIAL-OPTIMIZATION repository, social/functions.py).

* F1-F13 (scalable, D = 30): evaluated as published ("original") and with the optimum moved off-center
  ("shifted"): f_s(x) = f(x - o) with o_i = +/-0.6 * half-range (random signs, fixed per function).
* F14-F23 (fixed dimension): evaluated as published; most have off-center optima and act as a natural control.

Methods: SOCIAL (repository optimizer, default preset), SOCIAL (paper Algorithm 1), CMA-ES, DE, PSO, random.
Budget: 30,000 evaluations (D = 30) or 10,000 (fixed dimension); 10 seeds. Each run is cached as JSON, so the
script can be restarted and resumes.

    python analysis/center_bias_benchmarks/run_benchmarks.py [--n-jobs 8] [--seeds 10]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # graph-social repository root
sys.path.insert(0, str(HERE))
from optimizers import METHODS, SOCIAL_REPO, BudgetExhausted  # noqa: E402

sys.path.insert(0, str(SOCIAL_REPO))
from social.functions import BenchmarkFunctions  # noqa: E402

BF = BenchmarkFunctions()
SCALABLE = ["Sphere", "Schwefel_2_22", "Schwefel_1_2", "Schwefel_2_21", "Rosenbrock", "Step", "Quartic",
            "Schwefel_2_26", "Rastrigin", "Ackley", "Griewank", "Penalized", "Penalized2"]
FIXED_DIM = {"Foxholes": 2, "Kowalik": 4, "Camel-Back": 2, "Branin": 2, "Goldstein-Price": 2, "Hartman": 3,
             "Shekel1": 6, "Shekel2": 4, "Shekel3": 4, "Shekel4": 4}
OUT = ROOT / "results" / "runs_bench23"
USED = ["SOCIAL (repo, 60 agents)", "SOCIAL (paper Alg. 1)", "CMA-ES", "DE", "PSO", "random"]


class Bench:
    def __init__(self, name: str, dim: int, shift: bool, budget: int):
        self.fn, b, self.fopt = BF.functions[name]
        self.lo, self.hi = float(b[0]), float(b[1])
        self.dim, self.budget = dim, budget
        half = (self.hi - self.lo) / 2
        rng = np.random.default_rng(abs(hash(name)) % 2**32)
        self.o = (rng.choice([-1.0, 1.0], dim) * 0.6 * half) if shift else np.zeros(dim)
        self.evals, self.best_feasible, self.trace, self.best_x = 0, np.inf, [], None

    def __call__(self, u):
        if self.evals >= self.budget:
            raise BudgetExhausted
        x = self.lo + np.clip(np.asarray(u, float), 0, 1) * (self.hi - self.lo)
        f = float(self.fn(x - self.o))
        self.evals += 1
        if f < self.best_feasible:
            self.best_feasible = f
        return f


def run(name, dim, shift, method, seed, budget):
    f = OUT / f"{name}__{'shifted' if shift else 'original'}__{method.replace(' ', '_')}__s{seed}.json"
    if f.exists():
        return json.loads(f.read_text())
    b = Bench(name, dim, shift, budget)
    METHODS[method](b, seed)
    row = {"function": name, "dim": dim, "version": "shifted" if shift else "original", "method": method,
           "seed": seed, "best": b.best_feasible, "error": b.best_feasible - b.fopt}
    f.write_text(json.dumps(row))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name in SCALABLE:
        for shift in (False, True):
            jobs += [(name, 30, shift, m, s, 30000) for m in USED for s in range(a.seeds)]
    for name, d in FIXED_DIM.items():
        jobs += [(name, d, False, m, s, 10000) for m in USED for s in range(a.seeds)]
    # slowest (repository SOCIAL) first
    jobs.sort(key=lambda j: (not j[3].startswith("SOCIAL (repo"), j[0]))
    rows = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(run)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "bench23_runs.csv", index=False)
    med = df.groupby(["function", "version", "method"]).error.median().unstack("method")[USED]
    rank = med.rank(axis=1, method="average")
    lines = ["# Center-bias test on the 23 classic functions of the SOCIAL paper", "",
             "Median final error (f - f*) over seeds; lower is better. Rank 1 = best of the six methods.", "",
             med.to_markdown(floatfmt=".3g"), "", "## Rank of each method per function and version", "",
             rank.to_markdown(floatfmt=".1f"), ""]
    for v in ("original", "shifted"):
        sub = rank.xs(v, level="version").loc[SCALABLE] if v in rank.index.get_level_values("version") else None
        if sub is not None:
            lines.append(f"- F1-F13, {v}: mean rank " + ", ".join(f"{m} {sub[m].mean():.2f}" for m in USED))
    fixed = rank.loc[[(n, "original") for n in FIXED_DIM]]
    lines.append("- F14-F23 (as published): mean rank " + ", ".join(f"{m} {fixed[m].mean():.2f}" for m in USED))
    (ROOT / "reports" / "BENCH23_CENTER_BIAS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
