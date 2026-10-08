"""Figures (PNG at 300 dpi plus PDF). Every function takes plain data and an output stem."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DISPLAY = {"graph_social": "Graph-SOCIAL", "random": "Random", "greedy_walk": "Greedy walk", "gp_ei": "GP-EI",
           "ensemble_ts": "Ensemble TS", "de": "DE", "pso": "PSO", "ga": "GA", "social_ws": "SOCIAL-WS",
           "static_diverse": "Static diverse", "cmaes": "CMA-ES"}


def display(name: str) -> str:
    return DISPLAY.get(name, name)


PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d",
           "#475569", "#b45309", "#1e3a8a", "#be185d"]


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
        ax.set_xlabel("Pearson r(node, neighbor mean)")
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
        ax.plot(x, m, color=c, label=display(label), lw=1.5)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("top-1% recall")
    ax.set_title(title, fontsize=9)
    if own:
        ax.legend(frameon=False, fontsize=7)
        save(fig, stem)


def _giant_sample(top, bc: np.ndarray, max_nodes: int, seed: int):
    """Giant component, subsampled to ``max_nodes`` by BFS from its highest-betweenness node, with an FR layout."""
    import igraph as ig

    g = top.to_igraph()
    nodes = np.asarray(max(g.connected_components(), key=len))
    if len(nodes) > max_nodes:
        start = int(nodes[np.argmax(bc[nodes])])
        nodes = np.asarray(g.bfs(start)[0][:max_nodes])
    nodes = np.sort(nodes)  # induced_subgraph orders vertices by id; keep attributes aligned with it
    sub = g.induced_subgraph(nodes.tolist())
    ig.set_random_number_generator(__import__("random").Random(seed))
    return nodes, sub, np.asarray(sub.layout_fruchterman_reingold(niter=500).coords)


def _community_colors(cm: np.ndarray) -> list[str]:
    top_comms = list(pd.Series(cm).value_counts().index[:10])
    return [PALETTE[top_comms.index(c)] if c in top_comms else "#cbd5e1" for c in cm]


def graph_overview(top, comm: np.ndarray, bc: np.ndarray, stem: Path, max_nodes: int = 3000, seed: int = 0,
                   gap: np.ndarray | None = None) -> None:
    """F1: giant component (subsampled to ``max_nodes`` by BFS from the highest-betweenness node), colored by Leiden
    community; with ``gap``, a second panel shows the same layout colored by the PBE band gap (homophily)."""
    from matplotlib.collections import LineCollection

    nodes, sub, lay = _giant_sample(top, bc, max_nodes, seed)
    sizes = 4 + 60 * bc[nodes] / max(bc[nodes].max(), 1e-12)
    segs = [lay[list(e)] for e in sub.get_edgelist()]
    panels = 1 if gap is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(7 * panels / 1.4, 7 / 1.4), squeeze=False)
    for k, ax in enumerate(axes[0]):
        ax.add_collection(LineCollection(segs, colors="#e2e8f0", linewidths=0.3, zorder=1))
        if k == 0:
            ax.scatter(lay[:, 0], lay[:, 1], s=sizes, c=_community_colors(comm[nodes]), linewidths=0, zorder=2)
        else:
            g = gap[nodes]
            sc = ax.scatter(lay[:, 0], lay[:, 1], s=sizes, c=g, cmap="viridis", vmin=0,
                            vmax=float(np.quantile(g, 0.99)), linewidths=0, zorder=2)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01)
            cb.set_label("PBE band gap (eV)", fontsize=9)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_axis_off()
        if panels == 2:
            ax.text(0.0, 1.0, "ab"[k], transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")
    fig.subplots_adjust(wspace=0.02)
    save(fig, stem)


def network_full(top, comm: np.ndarray, bc: np.ndarray, stem: Path, seed: int = 0, min_size: int = 5) -> dict:
    """F10: all connected components with at least ``min_size`` MOFs (isolated MOFs and smaller components are
    omitted for clarity). Each component gets its own Fruchterman-Reingold layout scaled to a box whose side grows
    with sqrt(size); boxes are shelf-packed by size. Returns the counts of shown and omitted MOFs/components."""
    import random

    import igraph as ig
    from matplotlib.collections import LineCollection

    g = top.to_igraph()
    ig.set_random_number_generator(random.Random(seed))
    comps = sorted(g.connected_components(), key=len, reverse=True)
    xy = np.zeros((g.vcount(), 2))
    big = [c for c in comps if len(c) >= min_size]
    small = [c for c in comps if len(c) < min_size]
    row_w = 2.2 * np.sqrt(len(big[0]))
    x0 = y0 = row_h = 0.0
    for c in big:
        side = max(np.sqrt(len(c)), 1.5)
        lay = (np.asarray(g.induced_subgraph(c).layout_fruchterman_reingold(niter=500).coords) if len(c) > 2
               else np.array([[0.0, 0.0], [1.0, 0.0]]))
        lay = lay - lay.min(0)
        lay = lay / max(lay.max(), 1e-9) * side * 0.9
        if x0 + side > row_w and x0 > 0:
            x0, y0, row_h = 0.0, y0 - row_h - 0.6, 0.0
        xy[c] = lay + [x0, y0 - side]
        x0 += side + 0.6
        row_h = max(row_h, side)
    shown = np.zeros(g.vcount(), dtype=bool)
    for c in big:
        shown[c] = True  # isolated MOFs and components smaller than min_size are not drawn
    fig, ax = plt.subplots(figsize=(7, 7))
    edges = np.asarray(g.get_edgelist())
    edges = edges[shown[edges[:, 0]] & shown[edges[:, 1]]]
    ax.add_collection(LineCollection(xy[edges], colors="#94a3b8", linewidths=0.15, alpha=0.15, zorder=1))
    cols = np.array(_community_colors(comm), dtype=object)
    ax.scatter(xy[shown, 0], xy[shown, 1], s=(1 + 15 * bc / max(bc.max(), 1e-12))[shown], c=list(cols[shown]),
               linewidths=0, zorder=2)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.set_axis_off()
    save(fig, stem)
    return {"min_component_size": min_size, "mofs_total": int(g.vcount()), "mofs_shown": int(shown.sum()),
            "components_shown": len(big), "mofs_omitted": int((~shown).sum()), "components_omitted": len(small),
            "isolated_mofs": sum(len(c) == 1 for c in comps), "giant_component_mofs": len(comps[0])}


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
        ax.scatter([], [], color=PALETTE[mi % len(PALETTE)], label=display(m))
    for oi, obj in enumerate(sorted(means.objective.unique())):
        ax.scatter([], [], color="k", marker=markers[oi % 4], label=obj)
    ax.set_xlabel("final top-1% recall (mean)")
    ax.set_ylabel("family coverage (mean)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, stem)


def cd_diagram(ranks: pd.Series, nemenyi: pd.DataFrame, stem: Path, cd: float | None = None) -> None:
    """F5: critical-difference diagram (scikit-posthocs), lower rank = better; ``cd`` adds a Nemenyi CD scale bar."""
    import scikit_posthocs as sp

    fig, ax = plt.subplots(figsize=(8, 3))
    ranks = ranks.rename(index=display)
    nemenyi = nemenyi.rename(index=display, columns=display)
    # two decimals, as in the text (the library default .2g prints 10.50 as '10')
    sp.critical_difference_diagram(ranks, nemenyi, ax=ax, label_fmt_left="{label} ({rank:.2f})",
                                   label_fmt_right="({rank:.2f}) {label}")
    if cd is not None:
        lo = float(np.floor(ranks.min()))
        tr = ax.get_xaxis_transform()
        ax.plot([lo, lo + cd], [1.32, 1.32], color="k", lw=1.6, transform=tr, clip_on=False)
        for x in (lo, lo + cd):
            ax.plot([x, x], [1.28, 1.36], color="k", lw=1.2, transform=tr, clip_on=False)
        ax.text(lo + cd / 2, 1.39, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=9, transform=tr)
    save(fig, stem)


def ablation_bars(tab: pd.DataFrame, stem: Path) -> None:
    """F6: ``tab`` has columns label, objective, mean, ci (95% half-width)."""
    objs = sorted(tab.objective.unique())
    # Runs shared by two labels are drawn once: "+ρ=0.10" is the default run, "φ = 0.7" is topology (a).
    labels = [lb for lb in dict.fromkeys(tab.label) if lb not in ("topology (b) +ρ=0.10", "φ = 0.7")]
    fig, axes = plt.subplots(1, len(objs), figsize=(5 * len(objs), 0.32 * len(labels) + 1.5), sharey=True,
                             squeeze=False)
    for ax, obj in zip(axes[0], objs):
        sub = tab[tab.objective == obj].set_index("label").reindex(labels)
        y = np.arange(len(labels))
        ax.barh(y, sub["mean"], xerr=sub["ci"], color=["#2563eb" if lb == "default" else "#94a3b8" for lb in labels])
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlabel(f"final top-1% recall ({obj})")
    axes[0][0].invert_yaxis()  # shared y axis: invert once so the default is at the top
    save(fig, stem)


def heatmap(grids: dict[str, pd.DataFrame], stem: Path) -> None:
    """F7: phi x rho heatmaps of mean final recall, one per objective."""
    fig, axes = plt.subplots(1, len(grids), figsize=(4.8 * len(grids), 3.4), squeeze=False)
    fig.subplots_adjust(wspace=0.55)
    for ax, (obj, g) in zip(axes[0], grids.items()):
        im = ax.imshow(g.to_numpy(), cmap="viridis", aspect="auto")
        ax.set_xticks(range(g.shape[1]), [f"{c:.2f}" for c in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{r:.2f}" for r in g.index])
        ax.set_xlabel("ρ (long-range edge fraction)")
        ax.set_ylabel("φ")
        for (i, j), v in np.ndenumerate(g.to_numpy()):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(obj, fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
        cb.set_label("mean final recall", fontsize=8)
    save(fig, stem)


def h3_bars(rate_new: float, n_new: int, rate_base: float, n_base: int, p: float, stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.bar(["new-family hits", "all evaluations"], [rate_new, rate_base], color=["#dc2626", "#94a3b8"])
    for x, (r, n) in enumerate([(rate_new, n_new), (rate_base, n_base)]):
        ax.text(x, r, f"{r:.2f}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("share with top-weighted neighbor\nin top betweenness decile")
    ax.set_ylim(0, max(rate_new, rate_base, 0.1) * 1.3)
    save(fig, stem)


def runtime_box(df: pd.DataFrame, stem: Path) -> None:
    order = df.groupby("method")["wall_time_s"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([df[df.method == m]["wall_time_s"] for m in order], tick_labels=[display(m) for m in order],
               showfliers=False)
    ax.set_yscale("log")
    ax.set_ylabel("wall-clock per run (s, excl. oracle)")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    save(fig, stem)
