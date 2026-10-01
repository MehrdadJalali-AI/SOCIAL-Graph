"""Phase 6: H1 pilot (GATE 6) — does a chemically meaningful topology beat a degree-preserving random one?

Graph-SOCIAL on MOFGalaxyNet(phi*), MOFGalaxyNet + rho long-range edges, the degree-preserving random
graph and a Watts-Strogatz graph, plus random search; O2 (min PBE gap), budget 2% of N, pilot seeds.
Gate: some MOFGalaxyNet variant beats the random-topology variant on final top-1% recall, one-sided
paired Wilcoxon p < 0.05. On FAIL the phi x rho sensitivity grid is run at pilot scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from .. import plan, plots
from .. import stats as S
from ..experiment import load_results, run_all
from ..store import Store
from . import GateResult


def summary_table(df: pd.DataFrame, by: str = "label") -> pd.DataFrame:
    g = df.groupby(by)
    return pd.DataFrame({
        "runs": g.size(),
        "final_recall": g["final_recall"].apply(S.mean_std),
        "recall_auc": g["recall_auc"].apply(S.mean_std),
        "family_coverage": g["family_coverage"].apply(S.mean_std),
        "simple_regret": g["simple_regret"].apply(lambda x: S.mean_std(x, "{:.4f}")),
        "_mean_recall": g["final_recall"].mean(),
    }).sort_values("_mean_recall", ascending=False).drop(columns="_mean_recall").reset_index()


def run(cfg: dict, force: bool = False) -> GateResult:
    st = Store(cfg)
    tables, figs, reports = C.path(cfg, "tables"), C.path(cfg, "figures"), C.path(cfg, "reports")
    p6 = cfg["phase6"]
    phi = plan.phi_star(cfg)
    pairs = plan.phase6_specs(cfg)
    run_all([s for _, s in pairs], cfg, cfg.get("n_jobs", 1), "phase6 pilot")
    res = plan.labelled_frame(pairs, load_results([s for _, s in pairs], cfg))
    res.drop(columns=["recall_curve"]).to_csv(tables / "phase6_pilot.csv", index=False)
    tab = summary_table(res)

    wide = res.pivot_table(index="seed", columns="label", values="final_recall")
    null_label = "degree-preserving random"
    pilot_vs = [k for k in plan.pilot_variants(cfg, phi) if k.startswith("MOFGalaxyNet")]
    tests = []
    for v in pilot_vs:
        p = S.wilcoxon_paired(wide[v], wide[null_label], alternative="greater")
        tests.append({"variant": v, "vs": null_label, "mean_diff": float((wide[v] - wide[null_label]).mean()),
                      "wilcoxon_p_one_sided": p, "cliffs_delta": S.cliffs_delta(wide[v], wide[null_label])})
    for v in ["Watts–Strogatz", "random search"]:
        tests.append({"variant": "MOFGalaxyNet", "vs": v,
                      "mean_diff": float((wide["MOFGalaxyNet"] - wide[v]).mean()),
                      "wilcoxon_p_one_sided": S.wilcoxon_paired(wide["MOFGalaxyNet"], wide[v], "greater"),
                      "cliffs_delta": S.cliffs_delta(wide["MOFGalaxyNet"], wide[v])})
    tests = pd.DataFrame(tests)
    tests.to_csv(tables / "phase6_wilcoxon.csv", index=False)

    curves = {lab: np.vstack(res[res.label == lab].sort_values("seed")["recall_curve"].tolist())
              for lab in [*plan.pilot_variants(cfg, phi), "random search"]}
    plots.recall_curves(curves, figs / "phase6_recall_curves", f"Pilot: O2, budget {p6['budget_frac']:.0%}, φ={phi}")

    win = tests[tests.variant.isin(pilot_vs) & (tests.vs == null_label)]
    winners = win[(win.wilcoxon_p_one_sided < p6["alpha"]) & (win.mean_diff > 0)]
    ok = len(winners) > 0
    means = res[res.label.isin(pilot_vs)].groupby("label")["final_recall"].mean()
    best_label = means.idxmax()
    best_topo = plan.pilot_variants(cfg, phi)[best_label]
    st.write_choice("best_topology", best_topo)

    gate = "PASS" if ok else ("SKIPPED (smoke)" if cfg["smoke"] else "FAIL")
    summary = (f"{', '.join(winners.variant)} beat(s) the degree-preserving random topology "
               f"(p<{p6['alpha']}); best variant: {best_label}" if ok else
               f"no MOFGalaxyNet variant beats the random topology at p<{p6['alpha']} "
               f"(p = {', '.join(f'{p:.3g}' for p in win.wilcoxon_p_one_sided)})")

    lines = [
        "# Phase 6 — H1 pilot",
        "",
        f"**Gate 6: {gate}** — {summary}",
        "",
        f"Objective O2 (min PBE gap), budget {p6['budget_frac']:.0%} of N ({int(res.budget.iloc[0])} evaluations), "
        f"seeds {cfg['seeds_pilot'][0]}–{cfg['seeds_pilot'][1]}, φ* = {phi}, P = {cfg['experiments']['default_P']}. "
        "Mean ± std over seeds.",
        "",
        tab.to_markdown(index=False),
        "",
        "## Paired one-sided Wilcoxon on final top-1% recall (by seed)",
        "",
        tests.to_markdown(index=False, floatfmt=".4g"),
        "",
        f"Topology carried forward to Phase 7 (highest mean final recall among MOFGalaxyNet variants): "
        f"**{best_label}** (`{best_topo}`).",
        "",
        "Figure: `phase6_recall_curves.png`.",
    ]
    if not ok and not cfg["smoke"]:
        sens = plan.sensitivity_specs(cfg, p6["objective"], p6["budget_frac"], plan.seeds(cfg, "seeds_pilot"),
                                      "phase6_sensitivity")
        run_all(sens, cfg, cfg.get("n_jobs", 1), "phase6 φ×ρ sensitivity")
        sres = load_results(sens, cfg)
        sres["phi"] = sres.topology.str.extract(r"phi([0-9.]+)")[0].astype(float)
        sres["rho"] = sres.topology.str.extract(r"rho([0-9.]+)")[0].astype(float).fillna(0.0)
        grid = sres.pivot_table(index="phi", columns="rho", values="final_recall", aggfunc="mean")
        grid.to_csv(tables / "phase6_sensitivity.csv")
        lines += ["", "## φ × ρ sensitivity at pilot scale (mean final recall)", "",
                  grid.to_markdown(floatfmt=".4f"), "",
                  "Gate 6 failed: the pipeline stops here unless re-run with `--force`."]
    (reports / "PHASE6_PILOT.md").write_text("\n".join(lines) + "\n")
    return GateResult(6, ok or cfg["smoke"], gate, summary)
