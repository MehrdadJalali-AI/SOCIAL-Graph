"""Phase 4: landscape pre-checks (GATE 4) — band-gap homophily on the graphs vs degree-preserving nulls,
and building-block degeneracy."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from .. import config as C
from .. import plots
from ..graph import topologies
from ..graph.topologies import Topology
from ..store import Store, graph_name
from . import GateResult


def homophily_stats(top: Topology, y: np.ndarray) -> dict[str, float]:
    """Pearson/Spearman between a node's value and its neighbours' mean, and value assortativity."""
    deg = top.degree()
    mask = deg > 0
    if mask.sum() < 3:
        return {"pearson": np.nan, "spearman": np.nan, "assortativity": np.nan, "n_nodes": int(mask.sum())}
    nb_mean = (top.csr @ y)[mask] / deg[mask]
    yy = y[mask]
    return {
        "pearson": float(stats.pearsonr(yy, nb_mean)[0]),
        "spearman": float(stats.spearmanr(yy, nb_mean)[0]),
        "assortativity": float(top.to_igraph().assortativity(types1=y.tolist(), directed=False)),
        "n_nodes": int(mask.sum()),
    }


def run(cfg: dict, force: bool = False) -> GateResult:
    t0 = time.time()
    p4, p3 = cfg["phase4"], cfg["phase3"]
    st = Store(cfg)
    tables, figs, reports = C.path(cfg, "tables"), C.path(cfg, "figures"), C.path(cfg, "reports")
    df = st.load_clean()
    hse_idx = np.flatnonzero(df["hse_gap"].notna().to_numpy())
    targets = {"PBE": (np.arange(len(df)), df["pbe_gap"].to_numpy(float)),
               "HSE06": (hse_idx, df["hse_gap"].to_numpy(float)[hse_idx])}

    rows, nulls = [], {}
    for phi in p3["phis"]:
        base = Topology.load(st.graph(graph_name(phi)))
        for tname, (nodes, y) in targets.items():
            if len(nodes) < 10:
                continue
            g = base if len(nodes) == base.n else base.subgraph(nodes)
            obs = homophily_stats(g, y)
            null = [homophily_stats(topologies.degree_preserving(g, 1000 + r, p3["swaps_per_edge"]), y)
                    for r in range(p4["n_random"])]
            row = {"phi": phi, "target": tname, **obs}
            for k in ("pearson", "spearman", "assortativity"):
                v = np.array([d[k] for d in null], dtype=float)
                row[f"null_mean_{k}"] = float(np.nanmean(v))
                row[f"null_std_{k}"] = float(np.nanstd(v))
                row[f"z_{k}"] = float((obs[k] - np.nanmean(v)) / np.nanstd(v)) if np.nanstd(v) > 0 else np.nan
                row[f"p_{k}"] = float((1 + np.sum(v >= obs[k])) / (1 + len(v)))
            nulls[(tname, phi)] = np.array([d["pearson"] for d in null])
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(tables / "phase4_homophily.csv", index=False)
    plots.homophily(res, nulls, figs / "phase4_homophily")

    # Degeneracy.
    grp = df.groupby("bb_group")
    sizes = grp.size()
    multi = sizes[sizes > 1].index
    within = grp["pbe_gap"].std().loc[multi]
    deg_tab = pd.DataFrame([{
        "groups": len(sizes), "groups_size_gt1": len(multi), "mofs_in_dup_groups": int(sizes[multi].sum()),
        "largest_group": int(sizes.max()), "mean_within_group_std_pbe": float(within.mean()),
        "median_within_group_std_pbe": float(within.median()), "global_std_pbe": float(df.pbe_gap.std()),
    }])
    deg_tab.to_csv(tables / "phase4_degeneracy.csv", index=False)

    pbe = res[res.target == "PBE"]
    passing = pbe[(pbe.pearson > pbe.null_mean_pearson) & (pbe.p_pearson < p4["alpha"])]
    ok = len(passing) > 0
    phi_star = float(pbe.sort_values(["pearson", "z_pearson"], ascending=False).iloc[0].phi)
    st.write_choice("phi_star", phi_star)
    gate = "PASS" if ok else ("SKIPPED (smoke)" if cfg["smoke"] else "FAIL")
    summary = (f"PBE homophily significant (p<{p4['alpha']}) at φ ∈ {sorted(passing.phi.tolist())}; "
               f"strongest at φ*={phi_star}") if ok else "no φ with significant PBE homophily"

    cols = ["phi", "target", "n_nodes", "pearson", "null_mean_pearson", "z_pearson", "p_pearson",
            "spearman", "z_spearman", "p_spearman", "assortativity", "null_mean_assortativity", "z_assortativity",
            "p_assortativity"]
    lines = [
        "# Phase 4 — Landscape pre-checks",
        "",
        f"**Gate 4: {gate}** — {summary}",
        "",
        "## Homophily",
        "",
        "For each node with at least one neighbour: the correlation between its band gap and its neighbours' "
        "mean band gap, plus the value-assortativity coefficient. Null model: "
        f"{p4['n_random']} degree-preserving randomisations of the same graph (10·|E| swaps each). Empirical "
        "one-sided p = (1 + #null ≥ observed)/(1 + n). For HSE06, the graph is induced on the MOFs that have "
        "an HSE06 gap.",
        "",
        res[cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"φ* (strongest observed PBE homophily, used for the pilot): **{phi_star}**.",
        "",
        "## Degeneracy (identical linker + metal)",
        "",
        deg_tab.to_markdown(index=False, floatfmt=".3f"),
        "",
        "Within-group spread well below the global spread means that much of the graph homophily comes from "
        "building-block duplicates (near-cliques of identical linker and metal).",
        "",
        f"Figure: `phase4_homophily.png`. Runtime {time.time() - t0:.0f} s.",
    ]
    if not ok:
        lines += ["", "## Gate 4 failed — descriptor additions that might help",
                  "- node SMILES / secondary building unit identity, not only the metal element;",
                  "- topology code (`info.mofid.topology`) and pore geometry (PLD, LCD, density);",
                  "- metal oxidation state / coordination environment."]
    (reports / "PHASE4_PRECHECKS.md").write_text("\n".join(lines) + "\n")
    return GateResult(4, ok or cfg["smoke"], gate, summary)
