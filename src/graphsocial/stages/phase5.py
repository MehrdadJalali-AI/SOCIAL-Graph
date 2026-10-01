"""Phase 5: verify every method on the real problem (invariants + determinism) and time one run each.

Invariants checked per method: no MOF evaluated twice, budget used exactly and never exceeded, the first
P evaluations equal the shared initial design, the snap operator only returns unevaluated MOFs (implied
by the oracle refusing re-evaluations), and two runs with the same seed give identical traces.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C
from .. import methods, metrics, store
from ..experiment import budget_for
from ..store import Store, graph_name
from . import GateResult


def check_method(name: str, prob, budget: int, seed: int, P: int) -> dict:
    init = methods.initial_design(prob.n, P, seed)
    t0 = time.perf_counter()
    o1 = methods.make(name).run(prob, budget, seed, init)
    secs = time.perf_counter() - t0
    o2 = methods.make(name).run(prob, budget, seed, init)
    order = np.asarray(o1.order)
    s = metrics.summarize(order, prob.objective, prob.communities)
    return {
        "method": name,
        "budget": budget,
        "evaluations": o1.used,
        "unique": len(set(o1.order)) == len(o1.order),
        "budget_exact": o1.used == budget,
        "initial_design_first": list(o1.order[:P]) == [int(i) for i in init],
        "deterministic": o1.order == o2.order,
        "final_recall": s["final_recall"],
        "seconds": round(secs, 2),
    }


def run(cfg: dict, force: bool = False) -> GateResult:
    st = Store(cfg)
    tables, reports = C.path(cfg, "tables"), C.path(cfg, "reports")
    phi = st.read_choices().get("phi_star", cfg["phase3"]["phis"][-1])
    prob = store.problem(cfg, "O2", graph_name(phi), phi)
    P = cfg["experiments"]["default_P"]
    budget = budget_for(cfg["phase6"]["budget_frac"], prob.n, cfg["experiments"]["min_budget"])
    names = list(methods.REGISTRY) + list(methods.PRESETS)
    rows = [check_method(n, prob, budget, seed=0, P=P) for n in names]
    tab = pd.DataFrame(rows)
    tab.to_csv(tables / "phase5_method_checks.csv", index=False)
    ok = bool(tab[["unique", "budget_exact", "initial_design_first", "deterministic"]].all().all())
    gate = "PASS" if ok else "FAIL"
    summary = f"{len(tab)} methods; invariants {'hold' if ok else 'VIOLATED'} (O2, budget {budget}, φ={phi})"
    lines = [
        "# Phase 5 — Graph-SOCIAL and baselines",
        "",
        f"**Checks: {gate}** — {summary}",
        "",
        "Single-run checks on the real problem (seed 0). The full unit tests are in `tests/` (`python -m pytest`).",
        "",
        tab.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Methods: `graph_social` (SOCIAL over the MOF topology, h-hop agent neighbourhoods, community-aware mutation), "
        "`random`, `greedy_walk` (best-first walk on the MOF graph), `gp_ei`, `ensemble_ts` (10 bootstrap random "
        "forests), `de`/`pso`/`ga` with the snap operator, `social_ws` (original SOCIAL with a Watts–Strogatz "
        "agent graph and continuous mutation, plus snap) and `static_diverse` (k-center greedy).",
    ]
    (reports / "PHASE5_METHODS.md").write_text("\n".join(lines) + "\n")
    return GateResult(5, ok, gate, summary)
