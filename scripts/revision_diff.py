"""Write reports/REVISION_DIFF.md: how results changed between v0.2 (results/v0.2/tables) and the revision.

    python scripts/revision_diff.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD, NEW = ROOT / "results" / "v0.2" / "tables", ROOT / "results" / "tables"


def mean_of(s: pd.Series) -> pd.Series:
    return s.str.split(" ±").str[0].astype(float)


def counts(t4: pd.DataFrame) -> pd.DataFrame:
    t4 = t4[t4.metric == "final_recall"].copy()
    sig = t4["significant_holm_0.05"].astype(bool)
    t4["better"] = sig & (t4.mean_graph_social > t4.mean_baseline)
    t4["worse"] = sig & (t4.mean_graph_social < t4.mean_baseline)
    g = t4.groupby("baseline")
    return pd.DataFrame({"GS better": g.better.sum(), "n.s.": g.size() - g.better.sum() - g.worse.sum(),
                         "GS worse": g.worse.sum()}).astype(int)


def main() -> int:
    if not OLD.exists():
        print("no v0.2 tables archived")
        return 1
    out = ["# Revision diff (v0.2 → v0.3-revision)", "",
           "Changes: O3 redefined as |gap − 2.0 eV| (same 87-MOF hit set); GP-EI with ARD Matérn kernel, 5 optimiser "
           "restarts and ξ = 0.01. Everything else unchanged. Repository-only document (not part of the manuscript).", ""]

    ro = pd.read_csv(OLD / "T4b_friedman_ranks.csv").set_index("method").mean_rank
    rn = pd.read_csv(NEW / "T4b_friedman_ranks.csv").set_index("method").mean_rank
    rk = pd.DataFrame({"rank_v0.2": ro, "rank_new": rn})
    rk["position_v0.2"] = rk["rank_v0.2"].rank().astype(int)
    rk["position_new"] = rk["rank_new"].rank().astype(int)
    rk["Δ rank"] = rk["rank_new"] - rk["rank_v0.2"]
    out += ["## Friedman mean ranks (final recall; 1 = best)", "", rk.sort_values("rank_new").to_markdown(floatfmt=".2f"), ""]

    co, cn = counts(pd.read_csv(OLD / "T4_pairwise_stats.csv")), counts(pd.read_csv(NEW / "T4_pairwise_stats.csv"))
    both = co.join(cn, lsuffix=" (v0.2)", rsuffix=" (new)")
    out += ["## Graph-SOCIAL vs each baseline (16 cells, Holm-corrected Wilcoxon, final recall)", "",
            both.to_markdown(), ""]

    t3o, t3n = pd.read_csv(OLD / "T3_main_results.csv"), pd.read_csv(NEW / "T3_main_results.csv")
    for t in (t3o, t3n):
        t["recall"] = mean_of(t.final_recall)
    m = t3o.merge(t3n, on=["objective", "budget_frac", "method"], suffixes=("_v0.2", "_new"))
    gp = m[m.method == "gp_ei"][["objective", "budget_frac", "recall_v0.2", "recall_new"]]
    rnd = t3n[t3n.method == "random"].set_index(["objective", "budget_frac"]).recall.rename("random_new")
    gp = gp.join(rnd, on=["objective", "budget_frac"])
    gp["GP-EI ≥ random (new)"] = gp.recall_new >= gp.random_new
    out += ["## GP-EI final recall, isotropic (v0.2) vs ARD (new)", "", gp.to_markdown(index=False, floatfmt=".4f"), ""]
    o3 = m[m.objective == "O3"].pivot_table(index="method", columns="budget_frac", values=["recall_v0.2", "recall_new"])
    out += ["## O3 final recall, window objective (v0.2) vs target-value objective (new)", "",
            o3.round(4).to_markdown(), ""]

    for name in ("topology_effect_power.csv",):
        po, pn = pd.read_csv(OLD / name), pd.read_csv(NEW / name)
        cols = ["scope", "objective", "A", "B"]
        j = po.merge(pn, on=cols, suffixes=("_v0.2", "_new"))
        j = j[j.objective == "O3"][cols + ["mean_diff_v0.2", "mean_diff_new", "p_wilcoxon_one_sided_v0.2",
                                            "p_wilcoxon_one_sided_new", "cliffs_delta_v0.2", "cliffs_delta_new"]]
        out += ["## O3 topology comparisons", "", j.to_markdown(index=False, floatfmt=".4f"), ""]

    import re

    old_txt = (ROOT / "results" / "v0.2" / "RESULTS.md").read_text()
    mo = re.search(r"had a top-weighted neighbour: (\d+); of these, (\d+) \(([\d.]+)%\).*?base rate of ([\d.]+)% over all "
                   r"(\d+).*?p = ([\d.e-]+)\)", old_txt, re.S)
    h3n = pd.read_csv(NEW / "h3.csv").iloc[0]
    out += ["## H3 (bridge nodes)", "", "| version | new-family hit evals | share top-decile nb | base rate | p |",
            "|---|---|---|---|---|"]
    if mo:
        out.append(f"| v0.2 | {mo.group(1)} | {mo.group(3)}% | {mo.group(4)}% | {mo.group(6)} |")
    out.append(f"| new | {int(h3n.new_family_hit_evals)} | {100 * h3n.rate:.1f}% | {100 * h3n.base_rate:.1f}% | "
               f"{h3n.p_binomial_greater:.3g} |")
    out.append("")
    (ROOT / "reports" / "REVISION_DIFF.md").write_text("\n".join(out) + "\n")
    print("wrote reports/REVISION_DIFF.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
