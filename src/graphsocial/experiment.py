"""Execute run specifications: one (method, variant, objective, budget, seed) per result file, resumable.

Each run writes ``<runs>/<group>/<key>.parquet`` (the per-evaluation log) and ``<key>.json`` (metrics).
Runs whose JSON already exists are skipped, so an interrupted batch resumes where it stopped.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import config as C
from . import metrics, methods, store


@dataclass
class RunSpec:
    group: str            # e.g. "phase6", "phase7_main", "phase7_ablation"
    method: str           # registry or preset name
    variant: str          # label for the configuration (topology / ablation)
    objective: str
    budget_frac: float
    seed: int
    topology: str         # graph name used as search topology
    comm_phi: float       # phi of the reference Leiden communities
    P: int = 10
    params: dict = field(default_factory=dict)
    embedding: str = "default"  # "default" | "geometric" (decoupled search space)

    @property
    def key(self) -> str:
        emb = "" if self.embedding == "default" else f"__emb={self.embedding}"
        raw = (f"{self.method}__{self.variant}__{self.objective}__b{self.budget_frac:g}__P{self.P}"
               f"__c{self.comm_phi:.2f}{emb}__s{self.seed}")
        return re.sub(r"[^A-Za-z0-9_.=-]+", "-", raw)

    def paths(self, runs_dir: Path) -> tuple[Path, Path]:
        # The key (not the group) names the file, so identical configurations requested by several
        # stages (e.g. the default Graph-SOCIAL in the benchmark and in the ablations) run once.
        return runs_dir / f"{self.key}.parquet", runs_dir / f"{self.key}.json"


def budget_for(frac: float, n: int, min_budget: int) -> int:
    return max(int(math.ceil(frac * n)), min_budget)


def execute(spec: RunSpec, cfg: dict) -> dict:
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    trace_path, json_path = spec.paths(runs_dir)
    if json_path.exists():
        return json.loads(json_path.read_text())
    json_path.parent.mkdir(parents=True, exist_ok=True)
    prob = store.problem(cfg, spec.objective, spec.topology, spec.comm_phi, spec.embedding)
    budget = budget_for(spec.budget_frac, prob.n, cfg["experiments"]["min_budget"])
    init = methods.initial_design(prob.n, spec.P, spec.seed)
    method = methods.make(spec.method, **spec.params)
    with threadpool_limits(1):
        t0 = time.perf_counter()
        oracle = method.run(prob, budget, spec.seed, init)
        wall = time.perf_counter() - t0 - oracle.lookup_time
    assert oracle.used == budget, (spec.key, oracle.used, budget)
    order = np.asarray(oracle.order)
    summary = metrics.summarize(order, prob.objective, prob.communities)

    trace = pd.DataFrame(oracle.meta)
    trace.insert(0, "eval", np.arange(1, budget + 1))
    trace["mof_idx"] = order
    trace["mof_id"] = prob.mof_ids[order]
    trace["value"] = oracle.values
    trace["is_hit"] = prob.objective.hits[order]
    trace["community"] = prob.communities[order]
    trace["new_family_hit"] = metrics.new_family_hits(order, prob.objective.hits, prob.communities)
    for col in ("top_nb_mof", "top_nb_cent"):
        if col not in trace:
            trace[col] = np.nan
    bc = prob.cent["betweenness"]
    decile = np.quantile(bc, 0.9)
    nb = trace["top_nb_mof"].fillna(-1).astype(np.int64).to_numpy()
    trace["nb_top_decile_bc"] = np.where(nb >= 0, bc[np.clip(nb, 0, None)] >= decile, False)
    trace["has_nb"] = nb >= 0
    for k, v in (("seed", spec.seed), ("method", spec.method), ("variant", spec.variant),
                 ("objective", spec.objective)):
        trace[k] = v
    trace.to_parquet(trace_path, index=False)

    res = {**asdict(spec), "key": spec.key, "budget": budget, "wall_time_s": wall,
           "h3_new_family_hits": int(trace["new_family_hit"].sum()),
           "h3_new_family_hits_with_nb": int((trace["new_family_hit"] & trace["has_nb"]).sum()),
           "h3_new_family_hits_nb_top_decile": int((trace["new_family_hit"] & trace["nb_top_decile_bc"]).sum()),
           "h3_evals_with_nb": int(trace["has_nb"].sum()),
           "h3_evals_nb_top_decile": int(trace["nb_top_decile_bc"].sum()),
           **summary}
    tmp = json_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=_json_default))
    tmp.replace(json_path)
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def run_all(specs: list[RunSpec], cfg: dict, n_jobs: int = 1, desc: str = "runs") -> list[dict]:
    """Run specs (slowest methods first) in parallel; returns all summaries (including cached ones)."""
    import logging

    from joblib import Parallel, delayed

    log = logging.getLogger("graphsocial")
    slow = cfg["experiments"].get("slow_methods", ["gp_ei", "ensemble_ts"])
    specs = sorted(specs, key=lambda s: (s.method not in slow, slow.index(s.method) if s.method in slow else 0))
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    todo = [s for s in specs if not s.paths(runs_dir)[1].exists()]
    log.info("%s: %d specs, %d cached, %d to run (n_jobs=%d)", desc, len(specs), len(specs) - len(todo), len(todo), n_jobs)
    if todo:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        Parallel(n_jobs=n_jobs, verbose=5 if n_jobs > 1 else 0)(delayed(execute)(s, cfg) for s in todo)
    return [json.loads(s.paths(runs_dir)[1].read_text()) for s in specs]


def load_results(specs: list[RunSpec], cfg: dict) -> pd.DataFrame:
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    rows = []
    for s in specs:
        p = s.paths(runs_dir)[1]
        if p.exists():
            rows.append(json.loads(p.read_text()))
    return pd.DataFrame(rows)
