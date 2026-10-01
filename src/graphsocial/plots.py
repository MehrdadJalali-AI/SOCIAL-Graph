"""Figures (PNG at 300 dpi plus PDF). Every function takes plain data and an output stem."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d",
           "#475569", "#b45309", "#1e3a8a"]


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def degree_distributions(degs: dict[str, np.ndarray], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (label, d), c in zip(degs.items(), PALETTE):
        vals, counts = np.unique(d[d > 0], return_counts=True)
        ax.loglog(vals, counts / len(d), "o", ms=3, color=c, label=label, alpha=0.8)
    ax.set_xlabel("degree k")
    ax.set_ylabel("P(k)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Degree distribution (isolated nodes omitted)")
    save(fig, stem)


def community_sizes(sizes: dict[str, np.ndarray], stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for (label, s), c in zip(sizes.items(), PALETTE):
        s = np.sort(s)[::-1]
        ax.loglog(np.arange(1, len(s) + 1), s, "-", color=c, label=label)
    ax.set_xlabel("community rank")
    ax.set_ylabel("size")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Leiden community sizes")
    save(fig, stem)


def homophily(obs: pd.DataFrame, null: dict[tuple, np.ndarray], stem: Path) -> None:
    """One panel per (target, phi): null distribution of the Pearson homophily vs the observed value."""
    keys = list(null)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        target, phi = key
        o = obs[(obs.target == target) & (obs.phi == phi)].iloc[0]
        ax.hist(null[key], bins=20, color="#94a3b8", label="degree-preserving null")
        ax.axvline(o.pearson, color="#dc2626", lw=2, label="MOFGalaxyNet")
        ax.set_title(f"{target}, φ={phi}  (z={o.z_pearson:.1f})", fontsize=9)
        ax.set_xlabel("Pearson r(node, neighbour mean)")
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, stem)


def recall_curves(curves: dict[str, np.ndarray], stem: Path, title: str = "", ax=None) -> None:
    """``curves``: label -> (seeds, B) array. Mean with a 95% CI band (t-based)."""
    from scipy import stats

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    for (label, arr), c in zip(curves.items(), PALETTE):
        arr = np.asarray(arr, dtype=float)
        x = np.arange(1, arr.shape[1] + 1)
        m = arr.mean(axis=0)
        if arr.shape[0] > 1:
            h = stats.t.ppf(0.975, arr.shape[0] - 1) * arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            ax.fill_between(x, np.clip(m - h, 0, 1), np.clip(m + h, 0, 1), color=c, alpha=0.15, lw=0)
        ax.plot(x, m, color=c, label=label, lw=1.5)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("top-1% recall")
    ax.set_title(title, fontsize=9)
    if own:
        ax.legend(frameon=False, fontsize=7)
        save(fig, stem)


def graph_overview(top, comm: np.ndarray, bc: np.ndarray, stem: Path, max_nodes: int = 3000, seed: int = 0) -> None:
    """F1: giant component (subsampled to ``max_nodes`` by BFS from the highest-betweenness node)."""
    import igraph as ig

    g = top.to_igraph()
    giant = max(g.connected_components(), key=len)
    nodes = np.asarray(giant)
    if len(nodes) > max_nodes:
        start = int(nodes[np.argmax(bc[nodes])])
        order = g.bfs(start)[0]
        nodes = np.asarray(order[:max_nodes])
    nodes = np.sort(nodes)  # induced_subgraph orders vertices by id; keep attributes aligned with it
    sub = g.induced_subgraph(nodes.tolist())
    ig.set_random_number_generator(__import__("random").Random(seed))
    lay = np.asarray(sub.layout_fruchterman_reingold(niter=500).coords)
    cm = comm[nodes]
    top_comms = pd.Series(cm).value_counts().index[:10]
    colors = [PALETTE[list(top_comms).index(c)] if c in set(top_comms) else "#cbd5e1" for c in cm]
    sizes = 4 + 60 * bc[nodes] / max(bc[nodes].max(), 1e-12)
    fig, ax = plt.subplots(figsize=(7, 7))
    for e in sub.get_edgelist():
        ax.plot(lay[list(e), 0], lay[list(e), 1], color="#e2e8f0", lw=0.3, zorder=1)
    ax.scatter(lay[:, 0], lay[:, 1], s=sizes, c=colors, linewidths=0, zorder=2)
    ax.set_axis_off()
    ax.set_title(f"Giant component ({len(nodes)} of {len(giant)} nodes shown); colour = Leiden community "
                 "(10 largest), size = betweenness", fontsize=8)
    save(fig, stem)


def coverage_vs_recall(means: pd.DataFrame, stem: Path) -> None:
    """F4: method means of family coverage vs final recall, one marker shape per objective."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = sorted(means.method.unique())
    markers = ["o", "s", "^", "D"]
    for oi, obj in enumerate(sorted(means.objective.unique())):
        sub = means[means.objective == obj]
        for r in sub.itertuples():
            ax.scatter(r.final_recall, r.family_coverage, color=PALETTE[methods.index(r.method) % len(PALETTE)],
                       marker=markers[oi % 4], s=40)
    for mi, m in enumerate(methods):
        ax.scatter([], [], color=PALETTE[mi % len(PALETTE)], label=m)
    for oi, obj in enumerate(sorted(means.objective.unique())):
        ax.scatter([], [], color="k", marker=markers[oi % 4], label=obj)
    ax.set_xlabel("final top-1% recall (mean)")
    ax.set_ylabel("family coverage (mean)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, stem)


def cd_diagram(ranks: pd.Series, nemenyi: pd.DataFrame, stem: Path) -> None:
    """F5: critical-difference diagram (scikit-posthocs), lower rank = better."""
    import scikit_posthocs as sp

    fig, ax = plt.subplots(figsize=(8, 3))
    sp.critical_difference_diagram(ranks, nemenyi, ax=ax)
    ax.set_title("Mean rank across objective × budget blocks (final recall; 1 = best)", fontsize=9)
    save(fig, stem)


def ablation_bars(tab: pd.DataFrame, stem: Path) -> None:
    """F6: ``tab`` has columns label, objective, mean, ci (95% half-width)."""
    objs = sorted(tab.objective.unique())
    labels = list(dict.fromkeys(tab.label))
    fig, axes = plt.subplots(1, len(objs), figsize=(5 * len(objs), 0.32 * len(labels) + 1.5), sharey=True,
                             squeeze=False)
    for ax, obj in zip(axes[0], objs):
        sub = tab[tab.objective == obj].set_index("label").reindex(labels)
        y = np.arange(len(labels))
        ax.barh(y, sub["mean"], xerr=sub["ci"], color=["#2563eb" if lb == "default" else "#94a3b8" for lb in labels])
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_title(obj, fontsize=9)
        ax.set_xlabel("final top-1% recall")
    save(fig, stem)


def heatmap(grids: dict[str, pd.DataFrame], stem: Path) -> None:
    """F7: phi x rho heatmaps of mean final recall, one per objective."""
    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.4), squeeze=False)
    for ax, (obj, g) in zip(axes[0], grids.items()):
        im = ax.imshow(g.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(g.shape[1]), [f"{c:.2f}" for c in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{r:.2f}" for r in g.index])
        ax.set_xlabel("ρ (long-range edge fraction)")
        ax.set_ylabel("φ")
        for (i, j), v in np.ndenumerate(g.to_numpy()):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(obj, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, stem)


def h3_bars(rate_new: float, n_new: int, rate_base: float, n_base: int, p: float, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["new-family hits", "all evaluations"], [rate_new, rate_base], color=["#dc2626", "#94a3b8"])
    for x, (r, n) in enumerate([(rate_new, n_new), (rate_base, n_base)]):
        ax.text(x, r, f"{r:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("share with top-weighted neighbour\nin top betweenness decile")
    ax.set_title(f"H3 bridge nodes (binomial p = {p:.3g})", fontsize=9)
    ax.set_ylim(0, max(rate_new, rate_base, 0.1) * 1.3)
    save(fig, stem)


def runtime_box(df: pd.DataFrame, stem: Path) -> None:
    order = df.groupby("method")["wall_time_s"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df[df.method == m]["wall_time_s"] for m in order], tick_labels=list(order), showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock per run (s, excl. oracle)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    save(fig, stem)
