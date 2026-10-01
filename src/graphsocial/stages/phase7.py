"""Phase 7: full benchmark, ablations and phi x rho sensitivity.

Every (method, variant, objective, budget, P, seed) run writes its own files to ``results/runs/`` and is
skipped when they exist, so interrupted runs resume. Slow methods (GP-EI, ensemble TS) are scheduled first.
"""

from __future__ import annotations

import time

from .. import config as C
from .. import plan
from ..experiment import run_all
from . import GateResult


def run(cfg: dict, force: bool = False) -> GateResult:
    t0 = time.time()
    reports = C.path(cfg, "reports")
    n_jobs = cfg.get("n_jobs", 1)
    main = plan.main_specs(cfg)
    abl = [s for _, s in plan.ablation_specs(cfg)]
    sens = plan.phase7_sensitivity_specs(cfg)
    run_all(main + abl + sens, cfg, n_jobs, "phase7 (main + ablations + sensitivity)")
    n_unique = len({s.key for s in main + abl + sens})
    objs = plan.objectives_to_run(cfg)
    summary = (f"{len(main)} benchmark runs, {len(abl)} ablation and {len(sens)} sensitivity specs "
               f"({n_unique} unique runs); objectives {objs}")
    lines = [
        "# Phase 7 — Full benchmark, ablations, sensitivity",
        "",
        summary + ".",
        "",
        f"- Graph-SOCIAL topology: `{plan.best_topology(cfg)}` (from Phase 6); reference communities at "
        f"φ* = {plan.phi_star(cfg)}.",
        f"- Budgets: {cfg['phase7']['budgets']} of N (minimum {cfg['experiments']['min_budget']} evaluations); "
        f"seeds {cfg['seeds_full'][0]}–{cfg['seeds_full'][1]}.",
        f"- O4 runs only if the HSE06 subset has ≥ {cfg['objectives']['o4_min_mofs']} MOFs: "
        f"{'yes' if 'O4' in objs else 'no'}.",
        f"- Wall time of this stage: {time.time() - t0:.0f} s with n_jobs = {n_jobs}.",
        "",
        "Analysis, tables and figures are produced by stage 8 (`RESULTS.md`).",
    ]
    (reports / "PHASE7_BENCHMARK.md").write_text("\n".join(lines) + "\n")
    return GateResult(7, True, "n/a", summary)
