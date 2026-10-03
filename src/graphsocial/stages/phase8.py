"""Phase 8: statistics, tables T1-T6, figures F1-F9, RESULTS.md and results_bundle.zip."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
from scipy import stats as sps

from .. import config as C
from .. import plan, plots, report
from .. import stats as S
from ..experiment import load_results
from ..graph.topologies import Topology
from ..store import Store, graph_name
from . import GateResult

BASELINE_ORDER = ["random", "greedy_walk", "gp_ei", "ensemble_ts", "de", "pso", "ga", "cmaes", "social_ws",
                  "static_diverse"]


def _write(tab: pd.DataFrame, tables, name: str, floatfmt: str = ".4g") -> str:
    tab.to_csv(tables / f"{name}.csv", index=False)
    md = tab.to_markdown(index=False, floatfmt=floatfmt)
    (tables / f"{name}.md").write_text(md + "\n")
    return md


def _ci(x: pd.Series) -> float:
    return float(sps.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def run(cfg: dict, force: bool = False) -> GateResult:
    st = Store(cfg)
    tables, figs = C.path(cfg, "tables"), C.path(cfg, "figures")
    phi = plan.phi_star(cfg)
    out: dict[str, str] = {}

    # T1 dataset + graph statistics.
    g3 = pd.read_csv(tables / "phase3_graph_stats.csv")
    out["T1"] = _write(g3, tables, "T1_graph_stats", ".3f")
    # T2 homophily.
    h = pd.read_csv(tables / "phase4_homophily.csv")
    out["T2"] = _write(h[["phi", "target", "n_nodes", "pearson", "null_mean_pearson", "z_pearson", "p_pearson",
                          "spearman", "z_spearman", "assortativity", "z_assortativity", "p_assortativity"]],
                       tables, "T2_homophily")

    # T3 main results.
    main_specs = plan.main_specs(cfg)
    complete = _completeness(cfg, main_specs, tables)
    provisional = bool(cfg.get("provisional"))
    if not complete and not provisional:
        raise RuntimeError("Phase 7 runs are incomplete (see results/tables/completeness.csv); finish stage 7, "
                           "or pass --provisional for a clearly marked draft analysis")
    res = load_results(main_specs, cfg)
    if res.empty:
        raise RuntimeError("no Phase 7 results found; run stage 7 first")
    keys = ["objective", "budget_frac", "method"]
    g = res.groupby(keys)
    t3 = pd.DataFrame({
        "runs": g.size(),
        "recall_auc": g["recall_auc"].apply(S.mean_std),
        "final_recall": g["final_recall"].apply(S.mean_std),
        "family_coverage": g["family_coverage"].apply(S.mean_std),
        "simple_regret": g["simple_regret"].apply(lambda x: S.mean_std(x, "{:.4f}")),
        "enrichment": g["enrichment_factor"].apply(lambda x: S.mean_std(x, "{:.2f}")),
        "first_hit_eval_median": g["first_hit_eval"].median(),
    }).reset_index()
    out["T3"] = _write(t3, tables, "T3_main_results")

    # T4 pairwise statistics: Graph-SOCIAL vs each baseline per (objective, budget), Holm within block.
    rows = []
    for (obj, b), blk in res.groupby(["objective", "budget_frac"]):
        for metric in ("final_recall", "recall_auc"):
            w = blk.pivot_table(index="seed", columns="method", values=metric)
            base = [m for m in BASELINE_ORDER if m in w.columns]
            ps = [S.wilcoxon_paired(w["graph_social"], w[m]) for m in base]
            for m, p, padj in zip(base, ps, S.holm(ps)):
                rows.append({"objective": obj, "budget_frac": b, "metric": metric, "baseline": m,
                             "mean_graph_social": w["graph_social"].mean(), "mean_baseline": w[m].mean(),
                             "p_wilcoxon": p, "p_holm": padj, "cliffs_delta": S.cliffs_delta(w["graph_social"], w[m]),
                             "significant_holm_0.05": padj < 0.05})
    t4 = pd.DataFrame(rows)
    out["T4"] = _write(t4, tables, "T4_pairwise_stats")

    # Friedman + Nemenyi over (objective x budget) blocks, F5.
    # Rank within each (objective, budget) block on mean final recall over seeds, then average over blocks.
    blocks = res.groupby(["objective", "budget_frac", "method"])["final_recall"].mean().unstack("method")
    blocks = blocks.reindex(columns=cfg["phase7"]["methods"])
    n_blocks = len(plan.objectives_to_run(cfg)) * len(cfg["phase7"]["budgets"])
    try:
        chi2, pf, ranks, nem = S.friedman_nemenyi(blocks, n_blocks, len(cfg["phase7"]["methods"]))
    except S.IncompleteBlocks as exc:
        if not provisional:
            raise
        chi2, pf = float("nan"), float("nan")
        ranks = pd.Series(np.nan, index=blocks.columns)
        nem = pd.DataFrame(np.nan, index=blocks.columns, columns=blocks.columns)
        print(f"[provisional] Friedman not computed: {exc}")
    blocks.to_csv(tables / "T4d_friedman_blocks.csv")
    fried = pd.DataFrame({"method": ranks.index, "mean_rank": ranks.values}).sort_values("mean_rank")
    out["friedman"] = (f"Friedman χ² = {chi2:.2f}, p = {pf:.3g} over {len(blocks)} blocks.\n\n"
                       + _write(fried, tables, "T4b_friedman_ranks", ".2f"))
    nem.to_csv(tables / "T4c_nemenyi_pvalues.csv")
    pd.DataFrame([{"chi2": chi2, "p": pf, "blocks": len(blocks), "methods": blocks.shape[1]}]).to_csv(
        tables / "T4e_friedman_test.csv", index=False)
    if ranks.notna().all():
        plots.cd_diagram(ranks, nem, figs / "F5_critical_difference")

    # F3 recall curves at 2% budget, F4 coverage vs recall.
    curves_b = 0.02 if 0.02 in res.budget_frac.unique() else sorted(res.budget_frac.unique())[-1]
    objs = sorted(res.objective.unique())
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(objs), figsize=(4.6 * len(objs), 3.8), squeeze=False)
    for ax, obj in zip(axes[0], objs):
        sub = res[(res.objective == obj) & (res.budget_frac == curves_b)]
        curves = {m: np.vstack(sub[sub.method == m].sort_values("seed")["recall_curve"].tolist())
                  for m in ["graph_social", *BASELINE_ORDER] if (sub.method == m).any()}
        plots.recall_curves(curves, None, f"{obj}, budget {curves_b:.1%}", ax=ax)
    axes[0][0].legend(frameon=False, fontsize=6)
    plots.save(fig, figs / "F3_recall_curves")
    means = res.groupby(["objective", "method"])[["final_recall", "family_coverage"]].mean().reset_index()
    plots.coverage_vs_recall(means, figs / "F4_coverage_vs_recall")

    # T5 ablations, F6.
    abl_pairs = plan.ablation_specs(cfg)
    abl = plan.labelled_frame(abl_pairs, load_results([s for _, s in abl_pairs], cfg))
    rows, bars = [], []
    for (obj, lab), blk in abl.groupby(["objective", "label"], sort=False):
        d = abl[(abl.objective == obj) & (abl.label == "default")].set_index("seed")["final_recall"]
        x = blk.set_index("seed")["final_recall"]
        common = d.index.intersection(x.index)
        rows.append({"objective": obj, "ablation": lab, "runs": len(blk),
                     "final_recall": S.mean_std(blk.final_recall), "recall_auc": S.mean_std(blk.recall_auc),
                     "family_coverage": S.mean_std(blk.family_coverage),
                     "p_vs_default": np.nan if lab == "default" else S.wilcoxon_paired(x[common], d[common]),
                     "cliffs_delta_vs_default": np.nan if lab == "default" else S.cliffs_delta(x, d)})
        bars.append({"objective": obj, "label": lab, "mean": blk.final_recall.mean(), "ci": _ci(blk.final_recall)})
    out["T5"] = _write(pd.DataFrame(rows), tables, "T5_ablations")
    plots.ablation_bars(pd.DataFrame(bars), figs / "F6_ablations")

    # F7 phi x rho heatmap.
    sens = load_results(plan.phase7_sensitivity_specs(cfg), cfg)
    grids = {}
    if not sens.empty:
        sens["phi"] = sens.topology.str.extract(r"phi([0-9.]+)")[0].astype(float)
        sens["rho"] = sens.topology.str.extract(r"rho([0-9.]+)")[0].astype(float).fillna(0.0)
        for obj, blk in sens.groupby("objective"):
            grids[obj] = blk.pivot_table(index="phi", columns="rho", values="final_recall", aggfunc="mean")
            grids[obj].to_csv(tables / f"phi_rho_grid_{obj}.csv")
        plots.heatmap(grids, figs / "F7_phi_rho_heatmap")
    out["F7"] = "\n\n".join(f"**{o}**\n\n" + gr.to_markdown(floatfmt=".4f") for o, gr in grids.items())

    # H3 / F8.
    gs = res[res.method == "graph_social"]
    k, n = int(gs.h3_new_family_hits_nb_top_decile.sum()), int(gs.h3_new_family_hits_with_nb.sum())
    kb, nb = int(gs.h3_evals_nb_top_decile.sum()), int(gs.h3_evals_with_nb.sum())
    p0 = kb / nb if nb else np.nan
    p_h3 = float(sps.binomtest(k, n, p0, alternative="greater").pvalue) if n and nb else np.nan
    plots.h3_bars(k / n if n else 0.0, n, p0 if nb else 0.0, nb, p_h3, figs / "F8_h3_bridge_nodes")
    pd.DataFrame([{"new_family_hit_evals": n, "with_top_decile_nb": k, "rate": k / n if n else np.nan,
                   "base_evals": nb, "base_top_decile": kb, "base_rate": p0, "p_binomial_greater": p_h3}]).to_csv(
        tables / "h3.csv", index=False)
    out["H3"] = (f"Evaluations that found the first hit of a Leiden community and had a top-weighted neighbor: "
                 f"{n}; of these, {k} ({k / max(n, 1):.1%}) had that neighbor in the top betweenness decile, vs "
                 f"a base rate of {p0:.1%} over all {nb} Graph-SOCIAL update evaluations (one-sided binomial "
                 f"p = {p_h3:.3g}).")

    # T6 runtime, F9.
    t6 = res.groupby("method")["wall_time_s"].agg(["mean", "std", "median", "max"]).reset_index()
    out["T6"] = _write(t6.sort_values("median"), tables, "T6_runtime", ".3f")
    plots.runtime_box(res, figs / "F9_runtime")

    # F1 graph overview, F2 homophily (copied from Phase 4).
    best = plan.best_topology(cfg)
    top = Topology.load(st.graph(graph_name(phi)))
    plots.graph_overview(top, np.load(st.communities(phi)), st.centralities(graph_name(phi))["betweenness"],
                         figs / "F1_graph_overview")
    for ext in ("png", "pdf"):
        src = figs / f"phase4_homophily.{ext}"
        if src.exists():
            shutil.copy(src, figs / f"F2_homophily.{ext}")

    out.update(_checks_and_power(cfg, res, abl, figs))
    out.update(_bb_recall(cfg, main_specs))
    from .. import center_analysis
    ca = center_analysis.run(cfg)
    out["center"] = (ca["hits"].to_markdown(index=False, floatfmt=".3f") + "\n\n"
                     + ca["baseline"].to_markdown(index=False, floatfmt=".4f") + "\n\n"
                     + ca["contraction"].to_markdown(index=False, floatfmt=".3f"))
    report.build(cfg, out, {"phi_star": phi, "best_topology": best})
    wins = t4[(t4.metric == "final_recall") & (t4.mean_graph_social > t4.mean_baseline) & t4["significant_holm_0.05"]]
    summary = (f"Graph-SOCIAL significantly better (Holm p<0.05, final recall) in {len(wins)} of "
               f"{int((t4.metric == 'final_recall').sum())} comparisons; mean rank "
               f"{ranks.get('graph_social', np.nan):.2f} of {len(ranks)}")
    return GateResult(8, True, "n/a", summary)


TOPOLOGY_PAIRS = [
    ("topology (a) MOFGalaxyNet", "topology (c) degree-preserving random"),
    ("topology (b) +ρ=0.10", "topology (c) degree-preserving random"),
    ("topology (a) MOFGalaxyNet", "topology (d) Watts–Strogatz"),
    ("geometry-only embedding, topology (a) MOFGalaxyNet", "geometry-only embedding, topology (c) degree-preserving random"),
    ("default", "no neighbor term (α=β=0)"),
]
PILOT_PAIRS = [("MOFGalaxyNet", "degree-preserving random"), ("MOFGalaxyNet+ρ=0.10", "degree-preserving random"),
               ("MOFGalaxyNet", "Watts–Strogatz")]


def _pair_row(scope: str, obj: str, a_lab: str, b_lab: str, a: pd.Series, b: pd.Series) -> dict:
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    pw = S.power_paired((a - b).to_numpy())
    return {"scope": scope, "objective": obj, "A": a_lab, "B": b_lab, "mean_A": a.mean(), "mean_B": b.mean(),
            "mean_diff": (a - b).mean(), "p_wilcoxon_one_sided": S.wilcoxon_paired(a, b, "greater"),
            "cliffs_delta": S.cliffs_delta(a, b), **pw}


def _checks_and_power(cfg: dict, res: pd.DataFrame, abl: pd.DataFrame, figs) -> dict[str, str]:
    from .. import objectives as O

    tables = C.path(cfg, "tables")
    st = Store(cfg)
    df = st.load_clean()
    out = {}

    # Hit-set sizes.
    rows = []
    for obj in sorted(res.objective.unique()):
        o = O.make(obj, df, cfg["objectives"]["hit_fraction"])
        rule = "1% lowest f"
        if obj == "O3":
            inside, k = int((o.f == 0).sum()), int(np.ceil(cfg["objectives"]["hit_fraction"] * o.n))
            rule = (f"all {inside} in-window MOFs" if inside < k
                    else f"1% closest to 2.0 eV ({inside} MOFs lie inside the window)")
        rows.append({"objective": obj, "name": O.NAMES[obj], "universe_N": o.n, "hit_set_size": int(o.hits.sum()),
                     "rule": rule, "hit_gap_min_eV": float(o.gap[o.hits].min()),
                     "hit_gap_max_eV": float(o.gap[o.hits].max())})
    out["hits"] = _write(pd.DataFrame(rows), tables, "hit_sets", ".4g")

    # Random search vs its analytical expectation (recall = B/N; hits ~ hypergeometric).
    rows = []
    for (obj, b), blk in res[res.method == "random"].groupby(["objective", "budget_frac"]):
        N, K, B = int(blk.n_universe.iloc[0]), int(blk.n_hits_total.iloc[0]), int(blk.budget.iloc[0])
        exp_total, p = S.hypergeom_sum_test(int(blk.hits_found.sum()), len(blk), N, K, B)
        rows.append({"objective": obj, "budget_frac": b, "budget": B, "N": N, "hits_K": K, "seeds": len(blk),
                     "expected_recall_B_over_N": B / N, "observed_recall_mean": blk.final_recall.mean(),
                     "observed_recall_std": blk.final_recall.std(ddof=1), "expected_total_hits": exp_total,
                     "observed_total_hits": int(blk.hits_found.sum()), "exact_p_two_sided": p})
    out["random_check"] = _write(pd.DataFrame(rows), tables, "random_search_check", ".4g")

    # Topology comparison: effect sizes and achieved power (pilot, n=10; ablation, n=30).
    rows = []
    pilot = pd.read_csv(tables / "phase6_pilot.csv")
    for a_lab, b_lab in PILOT_PAIRS:
        w = pilot.pivot_table(index="seed", columns="label", values="final_recall")
        scope = f"pilot (seeds {cfg['seeds_pilot'][0]}–{cfg['seeds_pilot'][1]})"
        rows.append(_pair_row(scope, cfg["phase6"]["objective"], a_lab, b_lab, w[a_lab], w[b_lab]))
    for obj, blk in abl.groupby("objective"):
        w = blk.pivot_table(index="seed", columns="label", values="final_recall")
        for a_lab, b_lab in TOPOLOGY_PAIRS:
            if a_lab in w and b_lab in w:
                rows.append(_pair_row(f"ablation (seeds {cfg['seeds_full'][0]}–{cfg['seeds_full'][1]})", obj,
                                      a_lab, b_lab, w[a_lab], w[b_lab]))
    out["topology_power"] = _write(pd.DataFrame(rows), tables, "topology_effect_power", ".4g")
    return out


def _bb_recall(cfg: dict, specs) -> dict[str, str]:
    """Unique-building-block recall (post hoc, from the evaluation logs).

    Hits are grouped by building block (identical linker + metal). A run's unique-BB recall is the number of
    distinct hit groups it found divided by the number of distinct hit groups; finding several duplicates of
    one group counts once.
    """
    from .. import objectives as O
    from ..experiment import RunSpec  # noqa: F401  (type of specs)

    tables = C.path(cfg, "tables")
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    df = Store(cfg).load_clean()
    budgets = [b for b in (0.02, 0.05) if b in cfg["phase7"]["budgets"]] or cfg["phase7"]["budgets"][-1:]
    comp, rows = [], []
    for obj in plan.objectives_to_run(cfg):
        o = O.make(obj, df, cfg["objectives"]["hit_fraction"])
        bb = df["bb_group"].to_numpy()[o.universe]
        size_in_pool = pd.Series(bb).map(pd.Series(bb).value_counts()).to_numpy()
        hit_groups = np.unique(bb[o.hits])
        comp.append({"objective": obj, "hits": int(o.hits.sum()), "distinct_hit_groups": len(hit_groups),
                     "hits_in_duplicate_groups": int((size_in_pool[o.hits] > 1).sum()),
                     "fraction_hits_in_duplicate_groups": float((size_in_pool[o.hits] > 1).mean()),
                     "max_hits_in_one_group": int(pd.Series(bb[o.hits]).value_counts().max())})
        for s in specs:
            if s.objective != obj or s.budget_frac not in budgets:
                continue
            path = s.paths(runs_dir)[0]
            if not path.exists():  # run not finished yet (e.g. provisional analysis)
                continue
            tr = pd.read_parquet(path, columns=["mof_idx", "is_hit"])
            found = np.unique(bb[tr.mof_idx.to_numpy()[tr.is_hit.to_numpy()]])
            rows.append({"objective": obj, "budget_frac": s.budget_frac, "method": s.method, "seed": s.seed,
                         "bb_recall": len(found) / len(hit_groups),
                         "final_recall": float(tr.is_hit.sum() / o.hits.sum())})
    runs = pd.DataFrame(rows)
    runs.to_csv(tables / "bb_recall_runs.csv", index=False)
    g = runs.groupby(["objective", "budget_frac", "method"])
    summ = pd.DataFrame({"bb_recall": g["bb_recall"].apply(S.mean_std), "final_recall": g["final_recall"].apply(S.mean_std),
                         "bb_recall_mean": g["bb_recall"].mean(), "final_recall_mean": g["final_recall"].mean()}).reset_index()
    out = {"bb_composition": _write(pd.DataFrame(comp), tables, "bb_hit_composition", ".3f"),
           "bb_recall": _write(summ, tables, "bb_recall", ".4f")}
    # Rank agreement between ordinary and unique-BB recall (per objective x budget, method means).
    agree = []
    for (obj, b), blk in summ.groupby(["objective", "budget_frac"]):
        agree.append({"objective": obj, "budget_frac": b,
                      "spearman_rank_corr": float(blk.bb_recall_mean.corr(blk.final_recall_mean, method="spearman")),
                      "top_method_recall": blk.loc[blk.final_recall_mean.idxmax(), "method"],
                      "top_method_bb_recall": blk.loc[blk.bb_recall_mean.idxmax(), "method"]})
    out["bb_agreement"] = _write(pd.DataFrame(agree), tables, "bb_recall_rank_agreement", ".3f")
    return out


def _completeness(cfg: dict, specs, tables) -> bool:
    """Write completeness.csv (expected vs finished runs per method and budget); True if everything exists."""
    runs_dir = C.PROJECT_ROOT / cfg["paths"]["runs"]
    rows = [{"key": s.key, "method": s.method, "objective": s.objective, "budget_frac": s.budget_frac,
             "done": s.paths(runs_dir)[1].exists()} for s in specs]
    abl = [s for _, s in plan.ablation_specs(cfg)] + plan.phase7_sensitivity_specs(cfg)
    rows += [{"key": s.key, "method": f"{s.method} (ablation/sensitivity)", "objective": s.objective,
              "budget_frac": s.budget_frac, "done": s.paths(runs_dir)[1].exists()} for s in abl]
    d = pd.DataFrame(rows).drop_duplicates("key")  # one row per unique run
    summ = d.groupby(["method", "objective", "budget_frac"]).done.agg(expected="size", finished="sum").reset_index()
    summ.to_csv(tables / "completeness.csv", index=False)
    missing = summ[summ.finished < summ.expected]
    if len(missing):
        print(f"[completeness] {int((summ.expected - summ.finished).sum())} runs missing in {len(missing)} cells:")
        print(missing.to_string(index=False))
    return missing.empty
