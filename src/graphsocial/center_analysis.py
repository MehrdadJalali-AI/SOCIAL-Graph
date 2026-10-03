"""Post hoc analyses of centre bias (DEVIATIONS D13): where hits lie relative to the embedding centroid, a centroid-only
policy, and how far from the centroid each method's evaluations fall.

Called by stage 8; writes results/tables/center_hits.csv, center_baseline.csv and center_contraction.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import objectives as O
from . import plan
from .store import Store


def _dist_to_centroid(X: np.ndarray) -> np.ndarray:
    return np.linalg.norm(X - X.mean(axis=0), axis=1)


def _percentile_rank(d_pool: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Fraction of the pool that is at least as close to the centroid (0 = most central)."""
    srt = np.sort(d_pool)
    return np.searchsorted(srt, d, side="right") / len(srt)


def run(cfg: dict) -> dict[str, pd.DataFrame]:
    st = Store(cfg)
    tables = C.path(cfg, "tables")
    df = st.load_clean()
    embeds = {"chemistry-aware": np.load(st.embedding), "geometry-only": np.load(st.ensure_geometric_embedding())}
    objs = plan.objectives_to_run(cfg)

    # 1. Position of the hit sets relative to the centroid.
    rows = []
    for ename, X in embeds.items():
        for o in objs:
            ob = O.make(o, df, cfg["objectives"]["hit_fraction"])
            d = _dist_to_centroid(X[ob.universe])
            q = _percentile_rank(d, d[ob.hits])
            rows.append({"embedding": ename, "objective": o, "hits": int(ob.hits.sum()),
                         "share_hits_innermost_quarter": float(np.mean(q <= 0.25)),
                         "median_hit_percentile": float(np.median(q))})
    hits = pd.DataFrame(rows)
    hits.to_csv(tables / "center_hits.csv", index=False)

    # 2. Centroid-only policy. Main variant (used in text and tables): the shared 10-MOF random initial design
    #    (seeds as in the benchmark) followed by the B - 10 unevaluated MOFs closest to the centroid; the initial
    #    MOFs count toward budget and recall. Also reported: the policy without the initial design.
    from . import methods
    from .experiment import budget_for

    rows = []
    X = embeds["chemistry-aware"]
    P = cfg["experiments"]["default_P"]
    for o in objs:
        ob = O.make(o, df, cfg["objectives"]["hit_fraction"])
        d = _dist_to_centroid(X[ob.universe])
        order = np.argsort(d, kind="stable")
        for b in cfg["phase7"]["budgets"]:
            B = budget_for(b, len(order), cfg["experiments"]["min_budget"])
            rec = []
            for seed in plan.seeds(cfg, "seeds_full"):
                init = methods.initial_design(len(order), P, seed)
                taken = np.zeros(len(order), dtype=bool)
                taken[init] = True
                ev = np.concatenate([init, order[~taken[order]][: B - P]])
                rec.append(ob.hits[ev].sum() / ob.hits.sum())
            rows.append({"objective": o, "budget_frac": b, "budget": B,
                         "centroid_policy_recall": float(np.mean(rec)),
                         "centroid_policy_recall_std": float(np.std(rec, ddof=1)),
                         "centroid_policy_no_init_recall": float(ob.hits[order[:B]].sum() / ob.hits.sum())})
    base = pd.DataFrame(rows)
    base.to_csv(tables / "center_baseline.csv", index=False)

    # 3. Contraction: distance-to-centroid percentile of each method's evaluations (after the initial design),
    #    in the first and last thirds of the run, 2% budget, all objectives and seeds.
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    rows = []
    cache = {}
    for s in plan.main_specs(cfg):
        if s.budget_frac != 0.02:
            continue
        f = s.paths(runs_dir)[0]
        if not f.exists():
            continue
        if s.objective not in cache:
            ob = O.make(s.objective, df, cfg["objectives"]["hit_fraction"])
            d = _dist_to_centroid(X[ob.universe])
            cache[s.objective] = _percentile_rank(d, d)
        pr = cache[s.objective]
        tr = pd.read_parquet(f, columns=["eval", "mof_idx"])
        tr = tr[tr["eval"] > cfg["experiments"]["default_P"]]
        n = len(tr)
        if n < 3:
            continue
        first, last = tr.iloc[: n // 3], tr.iloc[-(n // 3):]
        rows.append({"method": s.method, "objective": s.objective, "seed": s.seed,
                     "early_median_percentile": float(np.median(pr[first.mof_idx.to_numpy()])),
                     "late_median_percentile": float(np.median(pr[last.mof_idx.to_numpy()]))})
    con = pd.DataFrame(rows)
    if len(con):
        con = con.groupby("method")[["early_median_percentile", "late_median_percentile"]].mean().reset_index()
    con.to_csv(tables / "center_contraction.csv", index=False)

    # 4. Mechanism check: contraction for the default Graph-SOCIAL and four ablations (O2, O3; 2% budget).
    labels = ["default", "sync off", "no neighbor term (α=β=0)", "elite off", "mutation off"]
    rows = []
    for label, s in plan.ablation_specs(cfg):
        if label not in labels:
            continue
        f = s.paths(runs_dir)[0]
        if not f.exists():
            continue
        if s.objective not in cache:
            ob = O.make(s.objective, df, cfg["objectives"]["hit_fraction"])
            d = _dist_to_centroid(X[ob.universe])
            cache[s.objective] = _percentile_rank(d, d)
        pr = cache[s.objective]
        tr = pd.read_parquet(f, columns=["eval", "mof_idx"])
        tr = tr[tr["eval"] > cfg["experiments"]["default_P"]]
        n = len(tr)
        rows.append({"configuration": label, "objective": s.objective, "seed": s.seed,
                     "early": float(np.median(pr[tr.iloc[: n // 3].mof_idx.to_numpy()])),
                     "late": float(np.median(pr[tr.iloc[-(n // 3):].mof_idx.to_numpy()]))})
    abl = pd.DataFrame(rows)
    abl = (abl.groupby(["configuration", "objective"])[["early", "late"]].mean().reset_index()
           .rename(columns={"early": "early_median_percentile", "late": "late_median_percentile"}))
    abl["configuration"] = pd.Categorical(abl.configuration, labels, ordered=True)
    abl = abl.sort_values(["objective", "configuration"])
    abl.to_csv(tables / "center_contraction_ablations.csv", index=False)
    return {"hits": hits, "baseline": base, "contraction": con, "ablations": abl}
