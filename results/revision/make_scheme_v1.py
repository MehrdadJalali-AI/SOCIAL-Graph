"""Method-overview schematic: how the SOCIAL optimizer is used in this study.

(a) SOCIAL as published: agents in a continuous search space on a synthetic small-world graph.
(b) Graph-SOCIAL: agents sit on MOFs of the similarity network (neighborhoods, centrality) and move in the
    chemistry-aware embedding, where the SOCIAL update is applied and agents snap to unevaluated MOFs.
(c) One iteration of Graph-SOCIAL.

    python manuscript/make_scheme.py      -> results/figures/F0_graph_social_scheme.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans"})
COMM = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
AGENT, FOCUS, GREY = "#dc2626", "#991b1b", "#94a3b8"


def arrow(ax, a, b, color="k", lw=1.0, style="-|>", ms=7, ls="-", rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=2, shrinkB=2))


def panel_label(ax, letter, title):
    ax.text(0.0, 1.02, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.07, 1.02, title, transform=ax.transAxes, fontsize=8, va="bottom")


def panel_social(ax, rng):
    """(a) Agents in a continuous space on a Watts-Strogatz ring with shortcuts."""
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    zz = np.sin(1.3 * xx) * np.cos(1.1 * yy) + 0.15 * (xx ** 2 + yy ** 2)
    ax.contourf(xx, yy, zz, levels=12, cmap="Greys", alpha=0.35)
    n = 10
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
    pos = np.c_[2.0 * np.cos(ang), 2.0 * np.sin(ang)] + rng.normal(scale=0.25, size=(n, 2))
    edges = [(i, (i + 1) % n) for i in range(n)] + [(i, (i + 2) % n) for i in range(n)] + [(0, 5), (3, 8)]
    for i, j in edges:
        ax.plot(*pos[[i, j]].T, color="#64748b", lw=0.7, zorder=2)
    ax.scatter(*pos.T, s=34, color=AGENT, edgecolors="white", linewidths=0.6, zorder=3)
    i = 0
    ax.scatter(*pos[i], s=70, color=FOCUS, edgecolors="white", linewidths=0.8, zorder=4)
    ax.text(*(pos[i] + [0.25, 0.2]), "$i$", fontsize=8, color=FOCUS, zorder=6)
    target = 0.55 * pos[i] + 0.45 * pos[[1, 2, 5]].mean(axis=0)
    arrow(ax, pos[i], target, color=FOCUS, lw=1.2)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("continuous search space,\nsynthetic small-world agent graph", fontsize=7, pad=3)
    ax.text(-0.02, 1.17, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.09, 1.17, "SOCIAL (our previous work)", transform=ax.transAxes, fontsize=8, va="bottom")


def mof_layout(rng):
    centers = np.array([[0.2, 0.75], [0.75, 0.78], [0.28, 0.22], [0.78, 0.25]])
    pts, lab = [], []
    for c, ctr in enumerate(centers):
        k = 9
        pts.append(ctr + rng.normal(scale=0.075, size=(k, 2)))
        lab += [c] * k
    return np.vstack(pts), np.array(lab)


def panel_network(ax, pts, lab, agents, focus, hop2):
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    edges = [(i, j) for i in range(len(pts)) for j in range(i + 1, len(pts))
             if (lab[i] == lab[j] and d[i, j] < 0.15)]
    bridges = [(4, 30), (13, 22), (8, 15)]
    for i, j in edges + bridges:
        ax.plot(*pts[[i, j]].T, color="#cbd5e1", lw=0.6, zorder=1)
    deg = np.zeros(len(pts))
    for i, j in edges + bridges:
        deg[i] += 1
        deg[j] += 1
    bc = np.zeros(len(pts))
    for i, j in bridges:
        bc[i] += 1
        bc[j] += 1
    halo = Circle(pts[focus], 0.0)
    for k in hop2:
        ax.add_patch(Circle(pts[k], 0.045, color="#fecaca", zorder=1.5, lw=0))
    ax.scatter(*pts.T, s=10 + 26 * bc + 2 * deg, c=[COMM[c] for c in lab], edgecolors="white", linewidths=0.4, zorder=2)
    for a in agents:
        ax.scatter(*pts[a], s=95, facecolors="none", edgecolors=FOCUS if a == focus else AGENT,
                   linewidths=1.6 if a == focus else 1.1, zorder=4)
    for a in agents:
        if a != focus and a in hop2:
            ax.plot(*pts[[focus, a]].T, color=AGENT, lw=0.9, ls=(0, (2, 1.5)), zorder=3)
    ax.text(*(pts[focus] + [-0.035, 0.05]), "$i$", fontsize=8, color=FOCUS, zorder=6, ha="right")
    br = bridges[1][0]
    ax.annotate("bridge MOF\n(high $c_j$)", xy=pts[br], xytext=(0.52, 0.47), fontsize=6.5, ha="center",
                color="#334155", arrowprops=dict(arrowstyle="-", color="#64748b", lw=0.6), zorder=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.set_title("MOF similarity network\n(who listens to whom)", fontsize=7, pad=3)
    ax.text(0.5, 0.03, "neighbors of $i$: agents within $h=2$ hops (shaded)", ha="center", fontsize=6.5, color=FOCUS)
    _ = halo


def panel_embedding(ax, rng):
    """Embedding view with explicit geometry: four overlapping chemical families, one agent update and snap."""
    centers = {0: (0.24, 0.66), 1: (0.56, 0.86), 2: (0.30, 0.22), 3: (0.80, 0.28)}
    pts, lab = [], []
    for c, (cx, cy) in centers.items():
        cov = np.array([[0.010, 0.004], [0.004, 0.006]])
        pts.append(rng.multivariate_normal([cx, cy], cov, size=11))
        lab += [c] * 11
    pts, lab = np.clip(np.vstack(pts), 0.06, 0.94), np.array(lab)
    evaluated = {1, 4, 9, 13, 17, 24, 27, 35, 38, 41}
    free = np.array([k not in evaluated for k in range(len(pts))])
    ax.scatter(*pts[free].T, s=12, c=[COMM[c] for c in lab[free]], edgecolors="white", linewidths=0.3, zorder=2)
    ax.scatter(*pts[~free].T, s=13, color=GREY, marker="x", linewidths=0.8, zorder=2)
    xi = np.array([0.16, 0.50])                      # focus agent i (on an evaluated MOF)
    nbrs = [np.array([0.36, 0.72]), np.array([0.30, 0.86])]  # neighbors found through the MOF network
    gbest, elite = np.array([0.86, 0.66]), np.array([0.90, 0.54])
    for nb in nbrs:
        ax.scatter(*nb, s=80, facecolors="none", edgecolors=AGENT, linewidths=1.1, zorder=4)
        ax.plot(*np.c_[xi, nb], color=AGENT, lw=0.6, ls=":", zorder=2)
    ax.scatter(*xi, s=95, facecolors="none", edgecolors=FOCUS, linewidths=1.6, zorder=4)
    ax.text(xi[0] - 0.035, xi[1] + 0.05, "$\\mathbf{x}_i$", fontsize=8, color=FOCUS, ha="right", zorder=7)
    ax.scatter(*gbest, marker="*", s=95, color="#facc15", edgecolors="k", linewidths=0.4, zorder=5)
    ax.scatter(*elite, marker="D", s=30, color="#f97316", edgecolors="k", linewidths=0.4, zorder=5)
    ax.text(gbest[0], gbest[1] + 0.05, "global best", fontsize=6.5, ha="center", zorder=7)
    ax.text(elite[0], elite[1] - 0.05, "elite\nmemory", fontsize=6.5, ha="center", va="top", zorder=7)
    new = np.array([0.42, 0.58])
    arrow(ax, xi, new, color=FOCUS, lw=1.4, ms=8)
    ax.scatter(*new, s=55, facecolors="white", edgecolors=FOCUS, linewidths=1.2, zorder=6)
    ax.text(0.30, 0.44, "SOCIAL\nupdate", fontsize=6.5, color=FOCUS, ha="center", va="top", zorder=7)
    cand = np.where(free)[0]
    snap = cand[np.argmin(np.linalg.norm(pts[cand] - new, axis=1))]
    ax.scatter(*pts[snap], s=110, facecolors="none", edgecolors="#16a34a", linewidths=1.6, zorder=6)
    arrow(ax, new, pts[snap], color="#16a34a", lw=1.2, ls=(0, (2, 1.5)), ms=7)
    ax.text(pts[snap][0] + 0.06, pts[snap][1] + 0.03, "snap", fontsize=7, color="#15803d", ha="left",
            va="bottom", fontweight="bold", zorder=7)
    ax.scatter([], [], s=12, color=GREY, marker="x", label="evaluated MOF")
    ax.legend(loc="lower right", frameon=False, fontsize=6.5, handletextpad=0.2, borderaxespad=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cbd5e1")
    ax.set_title("chemistry-aware embedding\n(where agents move)", fontsize=7, pad=3)


def panel_cycle(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1  Neighborhoods", "agents within $h$ hops\nin the MOF network"),
        ("2  Weights", r"$w_{ij}\propto\alpha_t c_j+\beta_t I_j$"),
        ("3  SOCIAL update", r"$\mathbf{x}_i\leftarrow$ self + neighbors" "\n" r"+ $\gamma_t$ global best + $\delta_t$ elite"),
        ("4  Snap", "nearest MOF not\nyet evaluated"),
        ("5  Evaluate", "DFT oracle, each MOF\nonce, budget $B$"),
    ]
    w, h, gap, y = 0.168, 0.50, 0.032, 0.40
    xs = [0.01 + k * (w + gap) for k in range(len(steps))]
    fills = ["#eff6ff", "#eff6ff", "#fef2f2", "#f0fdf4", "#fffbeb"]
    for (head, body), x, fc in zip(steps, xs, fills):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.03", fc=fc,
                                    ec="#94a3b8", lw=0.7))
        ax.text(x + w / 2, y + h - 0.12, head, ha="center", va="center", fontsize=7.5, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.08, body, ha="center", va="center", fontsize=6.8)
    for k in range(len(steps) - 1):
        arrow(ax, (xs[k] + w, y + h / 2), (xs[k + 1], y + h / 2), lw=1.0)
    yb = y - 0.12
    ax.plot([xs[-1] + w / 2, xs[-1] + w / 2, xs[0] + w / 2], [y, yb, yb], color="k", lw=1.0)
    arrow(ax, (xs[0] + w / 2, yb), (xs[0] + w / 2, y), lw=1.0, ms=7)
    ax.text(0.5, yb + 0.015, r"new value updates influence $I_j$, global best and elite memory", ha="center",
            va="bottom", fontsize=6.5, color="#334155")
    ax.text(0.5, 0.06, r"schedules: $\alpha_t,\beta_t$ decrease and $\gamma_t,\delta_t$ increase over iterations;  "
            "worse-than-median agents may jump to the least-explored chemical community",
            ha="center", va="bottom", fontsize=6.5, color="#334155")
    ax.text(0.0, 0.96, "c", fontsize=10, fontweight="bold", va="bottom")
    ax.text(0.025, 0.96, "One Graph-SOCIAL iteration (SOCIAL learning rules unchanged)", fontsize=8, va="bottom")


def main() -> None:
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(7.1, 4.9))
    ax_a = fig.add_axes([0.015, 0.42, 0.245, 0.44])
    ax_b1 = fig.add_axes([0.375, 0.42, 0.29, 0.44])
    ax_b2 = fig.add_axes([0.695, 0.42, 0.29, 0.44])
    ax_c = fig.add_axes([0.015, 0.01, 0.97, 0.33])

    panel_social(ax_a, rng)
    pts, lab = mof_layout(np.random.default_rng(11))
    focus = int(np.argmin(np.where(lab == 0, pts[:, 0], np.inf)))  # left-most MOF of the blue family
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    adj = (d < 0.15) & (lab[:, None] == lab[None, :])
    np.fill_diagonal(adj, False)
    hop1 = set(np.flatnonzero(adj[focus]))
    hop2 = set(hop1)
    for k in hop1:
        hop2 |= set(np.flatnonzero(adj[k]))
    hop2.discard(focus)
    two_hop = sorted(hop2 - hop1, key=lambda k: -np.linalg.norm(pts[k] - pts[focus]))
    nb_agent = two_hop[0] if two_hop else sorted(hop1)[0]
    agents = [focus, nb_agent, 12, 20, 29]
    panel_network(ax_b1, pts, lab, agents, focus, hop2)
    panel_embedding(ax_b2, np.random.default_rng(5))
    ax_b1.text(-0.02, 1.17, "b", transform=ax_b1.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax_b1.text(0.07, 1.17, "Graph-SOCIAL (this study): the same agents in two views", transform=ax_b1.transAxes,
               fontsize=8, va="bottom")
    fig.add_artist(FancyArrowPatch((0.268, 0.64), (0.365, 0.64), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color="k", lw=1.3))
    fig.text(0.316, 0.665, "replace the\nagent graph\nby the MOF\nnetwork", ha="center", va="bottom", fontsize=6.5)
    panel_cycle(ax_c)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F0_graph_social_scheme.{ext}", dpi=400)
    print("wrote", OUT / "F0_graph_social_scheme.pdf")


if __name__ == "__main__":
    main()
