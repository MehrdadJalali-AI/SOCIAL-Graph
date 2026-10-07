"""Revision 2 (post hoc) analyses; writes results/revision2/. Existing results are read, never modified.

    python scripts/revision2_analyses.py stats        # items 1-5, 7-9 (minutes)
    python scripts/revision2_analyses.py homophily    # item 6a (collapsed building blocks)
    python scripts/revision2_analyses.py dedup [--n-jobs 8]   # item 6b (deduplicated pool, new runs)

Items (revision brief):
 1  confirmatory topology comparisons on seeds 10-29 (pilot seeds 0-9 excluded)
 2  matched-pairs rank-biserial correlation r_rb and 95% bootstrap CI of the paired mean difference
 3  Holm families
 4  Friedman / Nemenyi sensitivity (per budget; 2% only) and Nemenyi critical differences
 5  RQ2 bridge analysis at the run level (cluster bootstrap over runs)
 6a homophily after collapsing building blocks; 6b acquisition on a deduplicated pool
 8  enrichment factors; 9 contraction by objective
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsocial import config as C, plan  # noqa: E402
from graphsocial import objectives as O  # noqa: E402
from graphsocial import stats as S  # noqa: E402
from graphsocial.experiment import budget_for, load_results  # noqa: E402
from graphsocial.store import Store, graph_name  # noqa: E402

OUT = ROOT / "results" / "revision2"
OUT.mkdir(parents=True, exist_ok=True)
T = ROOT / "results" / "tables"
N_BOOT = 10_000
CONFIRM_SEEDS = list(range(10, 30))
TOPOLOGY_PAIRS = [
    ("topology (a) MOFGalaxyNet", "topology (c) degree-preserving random"),
    ("topology (b) +ρ=0.10", "topology (c) degree-preserving random"),
    ("topology (a) MOFGalaxyNet", "topology (d) Watts–Strogatz"),
    ("geometry-only embedding, topology (a) MOFGalaxyNet", "geometry-only embedding, topology (c) degree-preserving random"),
    ("default", "no neighbor term (α=β=0)"),
]
PILOT_PAIRS = [("MOFGalaxyNet", "degree-preserving random"), ("MOFGalaxyNet+ρ=0.10", "degree-preserving random"),
               ("MOFGalaxyNet", "Watts–Strogatz"), ("MOFGalaxyNet", "random search")]


# ----------------------------------------------------------------------------------------------- helpers
def rank_biserial(a, b) -> float:
    """Matched-pairs rank-biserial correlation (W+ - W-)/(W+ + W-) with zero differences dropped, as in the
    Wilcoxon signed-rank test (zero_method="wilcox"); differences rounded to 1e-10 like stats.wilcoxon_paired."""
    d = np.round(np.asarray(a, float) - np.asarray(b, float), 10)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = sps.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def boot_ci(a, b, seed: int = 0, n: int = N_BOOT) -> tuple[float, float]:
    """95% percentile bootstrap CI of the paired mean difference (resampling seeds)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a: pd.Series, b: pd.Series, alternative: str = "two-sided") -> dict:
    common = a.index.intersection(b.index)
    a, b = a.loc[common].sort_index(), b.loc[common].sort_index()
    lo, hi = boot_ci(a.to_numpy(), b.to_numpy())
    return {"n_pairs": len(common), "mean_A": a.mean(), "mean_B": b.mean(), "mean_diff": float((a - b).mean()),
            "ci95_lo": lo, "ci95_hi": hi, "p_wilcoxon": S.wilcoxon_paired(a, b, alternative),
            "r_rb": rank_biserial(a, b), "cliffs_delta": S.cliffs_delta(a, b)}


def cfg_() -> dict:
    return C.load_config(None)


def main_results(cfg) -> pd.DataFrame:
    return load_results(plan.main_specs(cfg), cfg)


def ablation_frame(cfg) -> pd.DataFrame:
    pairs = plan.ablation_specs(cfg)
    return plan.labelled_frame(pairs, load_results([s for _, s in pairs], cfg))


# ----------------------------------------------------------------------------------------------- items 1-5, 8, 9
def item1_2_topology(cfg) -> pd.DataFrame:
    abl = ablation_frame(cfg)
    rows = []
    for obj, blk in abl.groupby("objective"):
        w = blk.pivot_table(index="seed", columns="label", values="final_recall")
        for a_lab, b_lab in TOPOLOGY_PAIRS:
            for scope, seeds in (("30 seeds (0–29)", list(range(30))), ("confirmatory seeds 10–29", CONFIRM_SEEDS)):
                a, b = w.loc[seeds, a_lab], w.loc[seeds, b_lab]
                pw = S.power_paired((a - b).to_numpy())
                rows.append({"scope": scope, "objective": obj, "A": a_lab, "B": b_lab,
                             **paired(a, b, "greater"), "d_z": pw["d_z"], "power_t": pw["power_t"],
                             "power_wilcoxon_boot": pw["power_wilcoxon_boot"], "n_for_80pct": pw["n_for_80pct"]})
    pil = pd.read_csv(T / "phase6_pilot.csv").pivot_table(index="seed", columns="label", values="final_recall")
    for a_lab, b_lab in PILOT_PAIRS:
        pw = S.power_paired((pil[a_lab] - pil[b_lab]).to_numpy())
        rows.append({"scope": "pilot (seeds 0–9)", "objective": "O2", "A": a_lab, "B": b_lab,
                     **paired(pil[a_lab], pil[b_lab], "greater"), "d_z": pw["d_z"], "power_t": pw["power_t"],
                     "power_wilcoxon_boot": pw["power_wilcoxon_boot"], "n_for_80pct": pw["n_for_80pct"]})
    tp = pd.DataFrame(rows)
    # Holm family (b): the 13 topology tests of Table S11 (3 pilot + 10 thirty-seed), one-sided.
    fam = tp[(tp.scope == "30 seeds (0–29)") | ((tp.scope == "pilot (seeds 0–9)") & (tp.B != "random search"))]
    tp["p_holm_topology_family"] = np.nan
    tp.loc[fam.index, "p_holm_topology_family"] = S.holm(fam.p_wilcoxon.tolist())
    conf = tp[tp.scope == "confirmatory seeds 10–29"]
    tp.loc[conf.index, "p_holm_confirmatory_family"] = S.holm(conf.p_wilcoxon.tolist())
    tp.to_csv(OUT / "topology_tests.csv", index=False)
    return tp


def item2_3_pairwise(cfg, res) -> pd.DataFrame:
    rows = []
    for (obj, b), blk in res.groupby(["objective", "budget_frac"]):
        w = blk.pivot_table(index="seed", columns="method", values="final_recall")
        for m in [c for c in w.columns if c != "graph_social"]:
            rows.append({"objective": obj, "budget_frac": b, "baseline": m, **paired(w["graph_social"], w[m])})
    pw = pd.DataFrame(rows)
    # Holm family (a): one family per baseline over its 16 objective-budget cells (two-sided).
    pw["p_holm_per_baseline"] = pw.groupby("baseline").p_wilcoxon.transform(lambda p: S.holm(p.tolist()))
    # Previous convention (one family per cell over the 10 baselines), kept for comparison.
    pw["p_holm_per_cell"] = pw.groupby(["objective", "budget_frac"]).p_wilcoxon.transform(lambda p: S.holm(p.tolist()))
    for col in ("per_baseline", "per_cell"):
        sig = pw[f"p_holm_{col}"] < 0.05
        pw[f"gs_better_{col}"] = sig & (pw.mean_diff > 0)
        pw[f"gs_worse_{col}"] = sig & (pw.mean_diff < 0)
    pw.to_csv(OUT / "pairwise_rrb.csv", index=False)
    summ = pw.groupby("baseline")[["gs_better_per_baseline", "gs_worse_per_baseline", "gs_better_per_cell",
                                   "gs_worse_per_cell"]].sum().astype(int)
    summ["r_rb_min"] = pw.groupby("baseline").r_rb.min()
    summ["r_rb_max"] = pw.groupby("baseline").r_rb.max()
    summ.to_csv(OUT / "pairwise_summary.csv")
    return pw


def item2_3_ablations(cfg) -> pd.DataFrame:
    abl = ablation_frame(cfg)
    rows = []
    for obj, blk in abl.groupby("objective"):
        w = blk.pivot_table(index="seed", columns="label", values="final_recall")
        for lab in [c for c in w.columns if c != "default"]:
            rows.append({"objective": obj, "ablation": lab, **paired(w[lab], w["default"])})
    ab = pd.DataFrame(rows)
    rest = ab[~ab.ablation.str.startswith("geometry-only") & ~ab.ablation.isin(["topology (b) +ρ=0.10", "φ = 0.7"])]
    ab["in_holm_family_36"] = ab.index.isin(rest.index)
    ab.loc[rest.index, "p_holm_36"] = S.holm(rest.p_wilcoxon.tolist())
    ab.to_csv(OUT / "ablations_rrb.csv", index=False)
    return ab


def item3_families(tp, pw, ab) -> pd.DataFrame:
    rc = pd.read_csv(T / "random_search_check.csv")
    rc["p_holm"] = S.holm(rc.exact_p_two_sided.tolist())
    rc.to_csv(OUT / "random_check_holm.csv", index=False)
    fam_b = tp.dropna(subset=["p_holm_topology_family"])
    rows = [
        {"family": "(a) Graph-SOCIAL vs each baseline", "definition": "one family per baseline: its 16 objective–budget "
         "cells, two-sided paired Wilcoxon on final recall", "families": pw.baseline.nunique(),
         "tests_per_family": int(pw.groupby("baseline").size().iloc[0]), "tests_total": len(pw),
         "significant": int((pw.p_holm_per_baseline < 0.05).sum()), "min_adjusted_p": pw.p_holm_per_baseline.min()},
        {"family": "(b) topology comparisons", "definition": "3 pilot (seeds 0–9) + 10 thirty-seed comparisons, one-sided",
         "families": 1, "tests_per_family": len(fam_b), "tests_total": len(fam_b),
         "significant": int((fam_b.p_holm_topology_family < 0.05).sum()),
         "min_adjusted_p": fam_b.p_holm_topology_family.min()},
        {"family": "(c) non-representation ablations", "definition": "each ablation vs default on O2 and O3, two-sided",
         "families": 1, "tests_per_family": int(ab.in_holm_family_36.sum()), "tests_total": int(ab.in_holm_family_36.sum()),
         "significant": int((ab.p_holm_36 < 0.05).sum()), "min_adjusted_p": ab.p_holm_36.min()},
        {"family": "(d) random search vs expectation", "definition": "exact two-sided hypergeometric test per cell",
         "families": 1, "tests_per_family": len(rc), "tests_total": len(rc),
         "significant": int((rc.p_holm < 0.05).sum()), "min_adjusted_p": rc.p_holm.min()},
    ]
    fams = pd.DataFrame(rows)
    fams.to_csv(OUT / "holm_families.csv", index=False)
    return fams


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    q = sps.studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
    return float(q * math.sqrt(k * (k + 1) / (6 * n_blocks)))


def item4_friedman(cfg, res) -> pd.DataFrame:
    methods = cfg["phase7"]["methods"]
    k = len(methods)
    rows, ranks_out = [], []
    blocks = res.groupby(["objective", "budget_frac", "method"])["final_recall"].mean().unstack("method")[methods]
    scopes = [("all 16 blocks", blocks)] + [(f"budget {b:g}", blocks.xs(b, level="budget_frac"))
                                             for b in sorted(res.budget_frac.unique())]
    for name, x in scopes:
        chi2, p, ranks, _ = S.friedman_nemenyi(x)
        rows.append({"scope": name, "blocks": len(x), "methods": k, "chi2": chi2, "p": p,
                     "nemenyi_cd_0.05": nemenyi_cd(k, len(x))})
        for m, r in ranks.items():
            ranks_out.append({"scope": name, "method": m, "mean_rank": r})
    fr = pd.DataFrame(rows)
    fr.to_csv(OUT / "friedman_sensitivity.csv", index=False)
    rk = pd.DataFrame(ranks_out).pivot(index="method", columns="scope", values="mean_rank")
    rk = rk[[s for s, _ in scopes]].sort_values("all 16 blocks")
    rk.to_csv(OUT / "friedman_sensitivity_ranks.csv")
    return fr


def item5_bridge(cfg, res) -> dict:
    gs = res[res.method == "graph_social"].copy()
    gs = gs[gs.h3_new_family_hits_with_nb > 0]
    share = gs.h3_new_family_hits_nb_top_decile / gs.h3_new_family_hits_with_nb
    base = gs.h3_evals_nb_top_decile / gs.h3_evals_with_nb
    d = (share - base).to_numpy()
    rng = np.random.default_rng(0)
    boot = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    obs = float(d.mean())
    # one-sided p for "share > base rate": centered (null-shifted) cluster bootstrap
    p_greater = float((1 + np.sum(boot - obs >= obs)) / (1 + N_BOOT))
    p_two = float(min(1.0, (1 + np.sum(np.abs(boot - obs) >= abs(obs))) / (1 + N_BOOT)))
    out = {"runs_with_new_family_hits": int(len(d)), "runs_total": int((res.method == "graph_social").sum()),
           "mean_share_top_decile": float(share.mean()), "mean_base_rate": float(base.mean()),
           "mean_difference": obs, "ci95_lo": float(np.quantile(boot, 0.025)),
           "ci95_hi": float(np.quantile(boot, 0.975)), "p_one_sided_greater": p_greater, "p_two_sided": p_two,
           "n_boot": N_BOOT}
    pd.DataFrame([out]).to_csv(OUT / "rq2_cluster.csv", index=False)
    pd.DataFrame({"key": gs.key, "objective": gs.objective, "budget_frac": gs.budget_frac, "seed": gs.seed,
                  "share_top_decile": share, "base_rate": base, "difference": d}).to_csv(
        OUT / "rq2_cluster_runs.csv", index=False)
    return out


def item8_ef(cfg, res) -> pd.DataFrame:
    g = res.groupby(["objective", "budget_frac", "method"])
    ef = pd.DataFrame({"budget": g.budget.first(), "N": g.n_universe.first(), "hits_K": g.n_hits_total.first(),
                       "final_recall_mean": g.final_recall.mean(), "final_recall_sd": g.final_recall.std(ddof=1),
                       "ef_mean": g.enrichment_factor.mean(), "ef_sd": g.enrichment_factor.std(ddof=1),
                       "recall_auc_mean": g.recall_auc.mean(), "recall_auc_sd": g.recall_auc.std(ddof=1),
                       "coverage_mean": g.family_coverage.mean(), "coverage_sd": g.family_coverage.std(ddof=1),
                       "regret_mean_eV": g.simple_regret.mean(), "regret_sd_eV": g.simple_regret.std(ddof=1),
                       "runs": g.size()}).reset_index()
    # check: EF = recall * N / B exactly
    assert np.allclose(ef.ef_mean, ef.final_recall_mean * ef.N / ef.budget)
    ef.to_csv(OUT / "full_benchmark.csv", index=False)
    return ef


def item9_contraction(cfg) -> pd.DataFrame:
    from graphsocial.center_analysis import _dist_to_centroid, _percentile_rank

    st = Store(cfg)
    df = st.load_clean()
    X = np.load(st.embedding)
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    P = cfg["experiments"]["default_P"]
    cache, rows = {}, []
    for s in plan.main_specs(cfg):
        if s.budget_frac != 0.02:
            continue
        if s.objective not in cache:
            ob = O.make(s.objective, df, cfg["objectives"]["hit_fraction"])
            d = _dist_to_centroid(X[ob.universe])
            cache[s.objective] = _percentile_rank(d, d)
        pr = cache[s.objective]
        tr = pd.read_parquet(s.paths(runs_dir)[0], columns=["eval", "mof_idx"])
        tr = tr[tr["eval"] > P]
        n = len(tr)
        rows.append({"method": s.method, "objective": s.objective, "seed": s.seed,
                     "early": float(np.median(pr[tr.iloc[: n // 3].mof_idx.to_numpy()])),
                     "late": float(np.median(pr[tr.iloc[-(n // 3):].mof_idx.to_numpy()]))})
    con = pd.DataFrame(rows)
    out = con.groupby(["method", "objective"]).agg(early_median_percentile=("early", "mean"),
                                                   late_median_percentile=("late", "mean")).reset_index()
    # paired change late - early over seeds, Wilcoxon two-sided
    tests = []
    for (m, o), blk in con.groupby(["method", "objective"]):
        tests.append({"method": m, "objective": o, "change_late_minus_early": float((blk.late - blk.early).mean()),
                      "p_wilcoxon_change": S.wilcoxon_paired(blk.late.to_numpy(), blk.early.to_numpy())})
    out = out.merge(pd.DataFrame(tests), on=["method", "objective"])
    out.to_csv(OUT / "contraction_by_objective.csv", index=False)
    return out


def run_stats() -> None:
    cfg = cfg_()
    res = main_results(cfg)
    tp = item1_2_topology(cfg)
    pw = item2_3_pairwise(cfg, res)
    ab = item2_3_ablations(cfg)
    fams = item3_families(tp, pw, ab)
    fr = item4_friedman(cfg, res)
    br = item5_bridge(cfg, res)
    ef = item8_ef(cfg, res)
    con = item9_contraction(cfg)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(tp[["scope", "objective", "A", "B", "mean_diff", "p_wilcoxon", "r_rb", "cliffs_delta", "power_t",
              "p_holm_topology_family"]].to_string())
    print(pd.read_csv(OUT / "pairwise_summary.csv").to_string())
    print(fams.to_string())
    print(fr.to_string())
    print(pd.read_csv(OUT / "friedman_sensitivity_ranks.csv").to_string())
    print(br)
    print(con.to_string())


# ----------------------------------------------------------------------------------------------- item 6a
def run_homophily() -> None:
    from graphsocial.graph import topologies
    from graphsocial.graph.topologies import Topology
    from graphsocial.stages.phase4 import homophily_stats

    cfg = cfg_()
    st = Store(cfg)
    df = st.load_clean()
    phi = plan.phi_star(cfg)
    base = Topology.load(st.graph(graph_name(phi)))
    rows = []
    for tname, col in (("PBE", "pbe_gap"), ("HSE06", "hse_gap")):
        nodes = np.flatnonzero(df[col].notna().to_numpy())
        g = base if len(nodes) == base.n else base.subgraph(nodes)
        bb = df["bb_group"].to_numpy()[nodes]
        codes, uniq = pd.factorize(bb)
        y = pd.Series(df[col].to_numpy(float)[nodes]).groupby(codes).mean().to_numpy()
        e = codes[g.edges]
        e = e[e[:, 0] != e[:, 1]]
        e = np.unique(np.sort(e, axis=1), axis=0)
        cg = Topology.from_pairs(f"collapsed_{tname}", len(uniq), e)
        obs = homophily_stats(cg, y)
        null = [homophily_stats(topologies.degree_preserving(cg, 1000 + r, cfg["phase3"]["swaps_per_edge"]), y)
                for r in range(cfg["phase4"]["n_random"])]
        v = np.array([d["pearson"] for d in null], float)
        rows.append({"phi": phi, "target": tname, "mofs": len(nodes), "building_blocks": len(uniq),
                     "edges_collapsed": len(e), "nodes_with_edges": obs["n_nodes"], "pearson": obs["pearson"],
                     "null_mean_pearson": float(np.nanmean(v)), "null_std_pearson": float(np.nanstd(v)),
                     "z_pearson": float((obs["pearson"] - np.nanmean(v)) / np.nanstd(v)),
                     "p_pearson": float((1 + np.sum(v >= obs["pearson"])) / (1 + len(v))),
                     "n_null": len(v)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "homophily_collapsed.csv", index=False)
    print(out.to_string())


# ----------------------------------------------------------------------------------------------- item 6b
DEDUP_METHODS = ["graph_social", "random", "ensemble_ts", "cmaes"]
DEDUP_BUDGETS = [0.02, 0.05]


def _dedup_seed(seed: int) -> list[dict]:
    import time

    from threadpoolctl import threadpool_limits

    from graphsocial import methods
    from graphsocial.center_analysis import _dist_to_centroid
    from graphsocial.graph import centrality
    from graphsocial.graph.topologies import Topology
    from graphsocial.store import Problem

    cfg = cfg_()
    cache_dir = OUT / "dedup_runs"
    cache_dir.mkdir(exist_ok=True)
    out_path = cache_dir / f"seed{seed}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    st = Store(cfg)
    df = st.load_clean()
    X_all = np.load(st.embedding)
    comm_all = np.load(st.communities(plan.phi_star(cfg)))
    full_top = Topology.load(st.graph(plan.best_topology(cfg)))
    rng = np.random.default_rng(10_000 + seed)
    bb = df["bb_group"].to_numpy()
    order = rng.permutation(len(df))
    _, first = np.unique(bb[order], return_index=True)
    reps = np.sort(order[first])                      # one random representative per building block
    dfd = df.iloc[reps].reset_index(drop=True)
    P = cfg["experiments"]["default_P"]
    c3 = cfg["phase3"]
    rows = []
    cent_cache = {}
    for obj_name in plan.objectives_to_run(cfg):
        ob = O.make(obj_name, dfd, cfg["objectives"]["hit_fraction"])
        nodes = reps[ob.universe]
        key = len(ob.universe) == len(dfd)
        if key not in cent_cache:
            top = full_top.subgraph(nodes)
            cent = centrality.all_centralities(top, c3["betweenness_exact_max_n"], c3["betweenness_k"], c3["seed"])
            cent.pop("_betweenness_method")
            cent_cache[key] = (top, cent)
        top, cent = cent_cache[key]
        prob = Problem(ob, X_all[nodes], top, cent, comm_all[nodes], dfd["qmof_id"].to_numpy()[ob.universe],
                       plan.best_topology(cfg))
        d = _dist_to_centroid(prob.X)
        inner = d <= np.quantile(d, 0.25)
        hit_share_inner = float(inner[ob.hits].mean())
        cen_order = np.argsort(d, kind="stable")
        for bfrac in DEDUP_BUDGETS:
            B = budget_for(bfrac, prob.n, cfg["experiments"]["min_budget"])
            init = methods.initial_design(prob.n, P, seed)
            base = {"seed": seed, "objective": obj_name, "budget_frac": bfrac, "budget": B, "N": prob.n,
                    "hits_K": int(ob.hits.sum()), "hit_share_innermost_quarter": hit_share_inner}
            for m in DEDUP_METHODS:
                meth = methods.make(m)
                with threadpool_limits(1):
                    t0 = time.perf_counter()
                    oracle = meth.run(prob, B, seed, init)
                    wall = time.perf_counter() - t0
                ordr = np.asarray(oracle.order)
                assert len(ordr) == B and len(np.unique(ordr)) == B
                rows.append({**base, "method": m, "final_recall": float(ob.hits[ordr].sum() / ob.hits.sum()),
                             "wall_time_s": wall})
            taken = np.zeros(prob.n, dtype=bool)
            taken[init] = True
            ev = np.concatenate([init, cen_order[~taken[cen_order]][: B - P]])
            rows.append({**base, "method": "centroid_policy_with_init",
                         "final_recall": float(ob.hits[ev].sum() / ob.hits.sum()), "wall_time_s": 0.0})
    out_path.write_text(json.dumps(rows))
    return rows


def run_dedup(n_jobs: int) -> None:
    from joblib import Parallel, delayed

    cfg = cfg_()
    seeds = plan.seeds(cfg, "seeds_full")
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_dedup_seed)(s) for s in seeds)
    runs = pd.DataFrame([r for rr in res for r in rr])
    runs.to_csv(OUT / "dedup_runs.csv", index=False)
    g = runs.groupby(["objective", "budget_frac", "method"])
    summ = pd.DataFrame({"N_mean": g.N.mean(), "budget_mean": g.budget.mean(), "hits_K_mean": g.hits_K.mean(),
                         "final_recall_mean": g.final_recall.mean(), "final_recall_sd": g.final_recall.std(ddof=1),
                         "hit_share_innermost_quarter": g.hit_share_innermost_quarter.mean(),
                         "runs": g.size()}).reset_index()
    summ.to_csv(OUT / "dedup_summary.csv", index=False)
    # paired comparisons: centroid policy and baselines vs Graph-SOCIAL
    rows = []
    for (o, b), blk in runs.groupby(["objective", "budget_frac"]):
        w = blk.pivot_table(index="seed", columns="method", values="final_recall")
        for m in [c for c in w.columns if c != "graph_social"]:
            rows.append({"objective": o, "budget_frac": b, "method": m, **paired(w[m], w["graph_social"])})
    pd.DataFrame(rows).to_csv(OUT / "dedup_vs_graph_social.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.to_string())


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if what == "stats":
        run_stats()
    elif what == "homophily":
        run_homophily()
    elif what == "dedup":
        nj = int(sys.argv[sys.argv.index("--n-jobs") + 1]) if "--n-jobs" in sys.argv else 1
        run_dedup(nj)
