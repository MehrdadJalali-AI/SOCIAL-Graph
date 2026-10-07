"""Figure 1: how the SOCIAL optimizer is used in this study (method overview).

Layout follows an author-supplied design, rebuilt here so that it is reproducible and editable:

(a) SOCIAL as published: agents in a continuous search space on a synthetic small-world graph.
(b) Graph-SOCIAL: the same agents in two views. Left, the MOF similarity network defines neighborhoods (agents within
    h = 2 hops) and betweenness; right, the chemistry-aware embedding is where the SOCIAL update moves an agent, which
    then snaps to the nearest unevaluated MOF. The update arrow is the actual convex combination of eq 5.
(c) One Graph-SOCIAL iteration.

    python scripts/make_scheme.py      -> results/figures/F0_graph_social_scheme.{pdf,png,svg}
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "figures"
plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "font.size": 6.5,
                     "svg.fonttype": "none", "pdf.fonttype": 42})

INK, SUB = "#263238", "#55606a"
FAM = ["#e06c75", "#26a69a", "#f0a04b", "#8e6bd1"]          # red, teal, orange, purple chemical families
CONN, EVAL = "#9ca3af", "#8b95a1"
AGENT, FOCUS = "#d32f2f", "#8e1b1b"
HALO = "#f8c9cf"
GBEST, ELITE, SNAP = "#f5b800", "#ef6c00", "#2e9e4f"
NBR = "#90bfe8"


def arrow(ax, a, b, color=INK, lw=1.0, ms=7, ls="-", rad=0.0, z=5, style="-|>", sa=2, sb=2, transform=None):
    kw = {} if transform is None else {"transform": transform}
    p = FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=sa, shrinkB=sb, **kw)
    (ax.add_patch(p) if transform is None else ax.add_artist(p))


def clean(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def agent_ring(ax, xy, s=70, color=AGENT, lw=1.3, z=6):
    ax.scatter(*xy, s=s, facecolors="none", edgecolors=color, linewidths=lw, zorder=z)


# --------------------------------------------------------------------------------------------------- (a)
def panel_social(ax, rng):
    clean(ax)
    th = np.linspace(0, 2 * np.pi, 400)
    xx, yy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 400))
    c = np.array([0.5, 0.47])
    ang = np.arctan2(yy - c[1], xx - c[0])
    rr = np.hypot(xx - c[0], yy - c[1])
    rad = 0.42 * (1 + 0.07 * np.sin(5 * ang + 0.6) + 0.04 * np.sin(3 * ang + 2.0))
    f = (rr / rad) ** 2 + 0.03 * np.sin(9 * xx) * np.cos(8 * yy)
    ax.contourf(xx, yy, np.where(f <= 1, f, np.nan), levels=np.linspace(0, 1, 13), cmap="Blues", vmin=-0.25,
                vmax=1.9, zorder=0)
    ax.contour(xx, yy, np.where(f <= 1, f, np.nan), levels=np.linspace(0.08, 0.92, 12), colors="white",
               linewidths=0.35, alpha=0.8, zorder=0.5)
    _ = th
    n = 10
    a = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.25
    pos = c + np.c_[0.30 * np.cos(a), 0.29 * np.sin(a)] + rng.normal(scale=0.018, size=(n, 2))
    edges = {tuple(sorted((i, (i + k) % n))) for i in range(n) for k in (1, 2)}   # Watts-Strogatz ring, K = 4
    edges -= {(2, 4), (6, 8)}
    edges |= {(2, 5), (6, 9)}                                                  # two rewired shortcuts
    for i, j in edges:
        ax.plot(*pos[[i, j]].T, color="#455a64", lw=0.6, zorder=2)
    ax.scatter(*pos.T, s=26, color=AGENT, edgecolors="white", linewidths=0.6, zorder=3)
    i = 0
    ax.scatter(*pos[i], s=46, color=FOCUS, edgecolors="white", linewidths=0.7, zorder=4)
    ax.text(*(pos[i] + [0.012, 0.035]), "$i$", fontsize=7, color=FOCUS, style="italic", zorder=6)
    arrow(ax, pos[i], pos[i] + 0.42 * (c - pos[i]), color=FOCUS, lw=1.1, ms=7)
    ax.plot(*c, marker="+", ms=8, mew=1.0, color=INK, zorder=4)
    ax.text(c[0], c[1] - 0.06, "optimum at center", ha="center", va="top", fontsize=5.4, color=INK, zorder=4)
    ax.text(0.5, 1.0, "continuous search space,\nsynthetic small-world agent graph", ha="center", va="top",
            fontsize=5.8, color=INK, linespacing=1.15)


# --------------------------------------------------------------------------------------------------- (b) network
def build_network(rng):
    centers = [(0.16, 0.79), (0.82, 0.80), (0.15, 0.20), (0.80, 0.18)]
    pts, fam = [], []
    for k, ctr in enumerate(centers):
        p = ctr + rng.normal(scale=0.058, size=(12, 2))
        pts.append(p)
        fam += [k] * len(p)
    pts, fam = np.vstack(pts), np.array(fam)
    conn = np.array([[0.40, 0.80], [0.56, 0.82], [0.44, 0.52], [0.66, 0.62], [0.32, 0.36], [0.63, 0.35],
                     [0.48, 0.17]])
    pts = np.vstack([pts, conn])
    fam = np.r_[fam, [-1] * len(conn)]
    N = len(pts)
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    E = set()
    for i in range(N):
        for j in range(i + 1, N):
            if fam[i] == fam[j] >= 0 and d[i, j] < 0.10:
                E.add((i, j))
    for k in range(4):                                     # make each family connected
        idx = np.flatnonzero(fam == k)
        for i in idx:
            others = idx[idx != i]
            j = others[np.argmin(d[i, others])]
            E.add(tuple(sorted((i, j))))
    hub = {k: np.flatnonzero(fam == k)[np.argmin(np.linalg.norm(pts[fam == k] - np.array(
        [[0.5, 0.5]]), axis=1))] for k in range(4)}
    c0 = 48
    links = [(hub[0], c0), (c0, c0 + 1), (c0 + 1, hub[1]), (c0 + 1, c0 + 2), (c0 + 2, c0 + 3), (c0 + 3, hub[1]),
             (c0 + 2, c0 + 4), (c0 + 4, hub[2]), (c0 + 2, c0 + 5), (c0 + 5, hub[3]), (hub[2], c0 + 6),
             (c0 + 6, hub[3])]
    E |= {tuple(sorted(e)) for e in links}
    return pts, fam, sorted(E), hub, c0 + 2


def hops(N, E, src, h=2):
    adj = [[] for _ in range(N)]
    for i, j in E:
        adj[i].append(j)
        adj[j].append(i)
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return {v: k for v, k in dist.items() if 0 < k <= h}


def panel_network(ax, rng):
    clean(ax)
    pts, fam, E, hub, bridge = build_network(rng)
    N = len(pts)
    nb_hub = [b if a == hub[0] else a for a, b in E if hub[0] in (a, b) and fam[a] == fam[b] == 0]
    ctr0 = pts[fam == 0].mean(axis=0)
    focus = min(nb_hub, key=lambda k: np.linalg.norm(pts[k] - ctr0))
    near = hops(N, E, focus)
    deg = np.bincount(np.ravel(E), minlength=N)
    for i, j in E:
        inter = fam[i] != fam[j] or fam[i] < 0
        col = "#b0b7c0" if inter else FAM[fam[i]]
        ax.plot(*pts[[i, j]].T, color=col, lw=0.9 if inter else 0.45, alpha=1 if inter else 0.55, zorder=1)
    for k in near:
        ax.add_patch(Circle(pts[k], 0.032, color=HALO, lw=0, zorder=0.5))
    col = [FAM[f] if f >= 0 else CONN for f in fam]
    size = np.where(fam < 0, 34, 7 + 2.2 * deg)
    ax.scatter(*pts.T, s=size, c=col, edgecolors="white", linewidths=0.4, zorder=2)
    ax.scatter(*pts[bridge], s=70, color=CONN, edgecolors="#2f6db5", linewidths=1.6, zorder=3)
    ax.annotate("bridge MOF\n(high $c_j$)", xy=pts[bridge], xytext=(pts[bridge][0] + 0.10, pts[bridge][1] - 0.02),
                fontsize=5.6, color=INK, va="center", ha="left", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=4))
    # agents: focus i, one neighbor j within 2 hops (same family), one agent in each other family
    two = [k for k, v in near.items() if v == 2 and fam[k] == 0]
    j = max(two or [k for k in near if fam[k] == 0], key=lambda k: np.linalg.norm(pts[k] - pts[focus]))
    others = [np.flatnonzero(fam == f)[np.argmax(np.linalg.norm(pts[fam == f] - pts[hub[f]], axis=1))]
              for f in (1, 2, 3)]
    ax.scatter(*pts[focus], s=46, color=FOCUS, edgecolors="white", linewidths=0.6, zorder=6)
    agent_ring(ax, pts[focus], s=95, color=FOCUS, lw=1.6, z=7)
    for a in [j, *others]:
        agent_ring(ax, pts[a], s=70)
    ax.plot(*pts[[focus, j]].T, color=AGENT, lw=0.9, ls=(0, (2, 1.4)), zorder=5)
    ax.text(pts[focus][0] + 0.03, pts[focus][1] + 0.04, "$i$", fontsize=7.5, color=FOCUS, ha="left", zorder=8,
            bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.8))
    ax.text(pts[j][0] + 0.03, pts[j][1] - 0.035, "$j$", fontsize=6.5, color=AGENT, zorder=8)
    lab = (0.03, 0.52)
    lo = min(near, key=lambda k: pts[k][1] if fam[k] == 0 else 9)
    ax.annotate("neighbors of $i$:\nagents within $h$ = 2 hops", xy=pts[lo] - [0, 0.03], xytext=lab, fontsize=5.6,
                color=INK, va="center", ha="left", zorder=7,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=1, shrinkB=0))
    ax.set_title("MOF similarity network: who listens to whom", fontsize=5.8, color=INK, pad=2)
    return fam


# --------------------------------------------------------------------------------------------------- (b) embedding
def panel_embedding(ax, rng):
    clean(ax)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("#b0b7c0")
        ax.spines[side].set_linewidth(0.6)
    regions = [(0.12, 0.84), (0.88, 0.70), (0.14, 0.20), (0.82, 0.18)]
    pts, fam = [], []
    for k, ctr in enumerate(regions):
        sc = (0.045, 0.035) if k == 0 else (0.06, 0.05)           # rose family sits around agents i and j
        p = np.clip(ctr + rng.normal(scale=sc, size=(7, 2)), 0.04, 0.96)
        pts.append(p)
        fam += [k] * len(p)
    conn = np.array([[0.36, 0.42], [0.58, 0.40], [0.62, 0.80], [0.46, 0.95], [0.52, 0.22], [0.70, 0.50],
                     [0.24, 0.40]])
    pts = np.vstack(pts + [conn])
    fam = np.r_[fam, [-1] * len(conn)]
    centroid = np.array([0.49, 0.53])
    xi = np.array([0.20, 0.72])                 # focus agent i
    xj = np.array([0.30, 0.87])                 # neighbor j, found through the network
    gb = np.array([0.72, 0.88])                 # global best (best current agent)
    el = np.array([0.70, 0.22])                 # elite memory (best found so far)
    tau, a, b, g, d = 0.6, 0.4, 0.4, 0.2, 0.2   # eq 2 schedules at tau = t/T
    at, bt, gt, dt = a * (1 - tau), b * (1 - tau), g * tau, d * tau
    new = (1 - at - bt - gt - dt) * xi + (at + bt) * xj + gt * gb + dt * el          # eq 5
    snap_to = new + np.array([0.05, -0.065])
    pts = np.vstack([pts, snap_to])
    fam = np.r_[fam, -2]
    evaluated = np.array([0.0] * len(pts), dtype=bool)
    evaluated[[1, 5, 8, 10, 16, 19, 24, 30]] = True
    ex = np.array([[0.04, 0.70], [0.06, 0.52], [0.58, 0.86], [0.88, 0.70], [0.08, 0.10], [0.92, 0.10],
                   [0.62, 0.12], [0.40, 0.08], [0.92, 0.40]])
    free = ~evaluated
    col = [FAM[f] if f >= 0 else (SNAP if f == -2 else CONN) for f in fam]
    sz = [9 if f >= 0 else (26 if f == -2 else 8) for f in fam]
    for k in np.flatnonzero(free):
        ax.scatter(*pts[k], s=sz[k], color=col[k], edgecolors="white", linewidths=0.3, zorder=2)
    ax.scatter(*np.vstack([pts[evaluated], ex]).T, s=12, color=EVAL, marker="x", linewidths=0.7, zorder=2)
    ax.plot(*centroid, marker="+", ms=8, mew=1.0, color=INK, zorder=3)
    ax.text(centroid[0] + 0.04, centroid[1], "centroid", ha="left", va="center", fontsize=5.2, color=INK)
    # pulls on agent i
    arrow(ax, xi, xj, color=AGENT, lw=0.7, ls=(0, (2, 1.4)), ms=5, sa=5, sb=5, z=4)
    arrow(ax, xi, gb, color=GBEST, lw=0.7, ls=(0, (2, 1.4)), ms=5, sa=5, sb=6, z=4)
    arrow(ax, xi, el, color=ELITE, lw=0.7, ls=(0, (2, 1.4)), ms=5, sa=5, sb=6, z=4)
    ax.scatter(*xj, s=12, color=FAM[0], zorder=5)
    agent_ring(ax, xj, s=60)
    ax.text(xj[0] + 0.035, xj[1] + 0.02, "$j$", fontsize=6.5, color=AGENT, zorder=8)
    ax.scatter(*gb, marker="*", s=70, color=GBEST, edgecolors="k", linewidths=0.4, zorder=6)
    agent_ring(ax, gb, s=95)
    ax.text(gb[0] + 0.05, gb[1], "global best", fontsize=5.6, va="center", color=INK)
    ax.scatter(*el, marker="D", s=26, color=ELITE, edgecolors="k", linewidths=0.4, zorder=6)
    ax.text(el[0] + 0.045, el[1] + 0.005, "elite memory", fontsize=5.6, va="center", color=INK)
    ax.scatter(*xi, s=40, color=FOCUS, edgecolors="white", linewidths=0.6, zorder=6)
    agent_ring(ax, xi, s=90, color=FOCUS, lw=1.6, z=7)
    ax.text(xi[0] - 0.045, xi[1], r"$\mathbf{x}_i$", fontsize=7.5, color=FOCUS, ha="right", va="center", zorder=8)
    # SOCIAL update and snap
    arrow(ax, xi, new, color=FOCUS, lw=1.4, ms=7, sa=5, sb=4, z=6)
    ax.scatter(*new, s=48, facecolors="white", edgecolors=FOCUS, linewidths=0.9, linestyle=(0, (2, 1.2)), zorder=7)
    ax.text(0.12, xi[1] - 0.06, "SOCIAL update\n(eq 5)", fontsize=5.3, color=FOCUS, ha="center", va="top",
            linespacing=1.1)
    arrow(ax, new, snap_to, color=SNAP, lw=1.2, ms=6, sa=4, sb=4, z=6)
    ax.text(new[0] + 0.045, new[1] + 0.01, "snap", fontsize=6.2, color=SNAP, fontweight="bold", va="center")
    ax.text(snap_to[0] + 0.045, snap_to[1] - 0.005, "nearest\nunevaluated MOF", fontsize=5.3, color=INK,
            va="center", linespacing=1.1)
    ax.set_title("chemistry-aware embedding: where agents move", fontsize=5.8, color=INK, pad=2)


def legend_row(fig, y):
    h = [Line2D([], [], ls="", marker="o", ms=3.6, mfc=c, mec="none") for c in FAM]
    h += [Line2D([], [], ls="", marker="o", ms=3.6, mfc=CONN, mec="none"),
          Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec=AGENT, mew=1.2),
          Line2D([], [], ls="", marker="x", ms=4, mec=EVAL, mew=0.8),
          Line2D([], [], ls="", marker="o", ms=6.5, mfc=HALO, mec="none")]
    labels = ["", "", "", "MOF families", "connector MOF", "agent", "evaluated MOF", "2-hop neighborhood of $i$"]
    fig.legend(h, labels, loc="center", bbox_to_anchor=(0.635, y), ncol=8, frameon=False, fontsize=5.4,
               handletextpad=0.3, columnspacing=0.9, handlelength=1.0)


# --------------------------------------------------------------------------------------------------- (c)
def panel_cycle(ax):
    clean(ax)
    ax.set_ylim(0, 1)
    steps = [("Neighborhoods", "#9ec5e8", "agents within $h$ hops\nin the MOF network"),
             ("Weights", "#80cbc4", r"$w_{ij}\propto\alpha_t c_j+\beta_t I_j$"),
             ("SOCIAL update", "#ef8a80", r"$\mathbf{x}_i\leftarrow$ self + $(\alpha_t+\beta_t)$ neighbors" "\n"
              r"+ $\gamma_t$ global best + $\delta_t$ elite"),
             ("Snap", "#6cc070", "nearest MOF\nnot yet evaluated"),
             ("Evaluate", "#f4c542", "DFT value revealed (look-up);\neach MOF once; budget $B$")]
    xs = [0.075, 0.30, 0.50, 0.725, 0.905]
    yh, yi, yt = 0.86, 0.56, 0.24
    for k, ((head, c, body), x) in enumerate(zip(steps, xs)):
        ax.scatter(x - 0.055, yh, s=150, color=c, zorder=3)
        ax.text(x - 0.055, yh, str(k + 1), ha="center", va="center", fontsize=7, fontweight="bold", color=INK,
                zorder=4)
        ax.text(x - 0.037, yh, head, ha="left", va="center", fontsize=6.6, fontweight="bold", color=INK)
        ax.text(x, yt, body, ha="center", va="center", fontsize=5.8, color=INK, linespacing=1.25)
    heads = [t for t in ax.texts if t.get_fontweight() == "bold" and t.get_text() in [st[0] for st in steps]]
    ax.figure.canvas.draw()
    inv = ax.transData.inverted()
    for t, x in zip(heads, xs[1:]):
        e = inv.transform(t.get_window_extent())[1, 0]
        arrow(ax, (e + 0.015, yh), (x - 0.085, yh), lw=0.8, ms=6, sa=0, sb=0)
    # icons
    s = 0.024
    c1 = np.array([xs[0], yi])
    ring = [c1 + 2.4 * s * np.array([np.cos(t), 1.9 * np.sin(t)]) for t in np.linspace(0.3, 2 * np.pi + 0.3, 6)[:-1]]
    for p in ring:
        ax.plot(*np.c_[c1, p], color="#78909c", lw=0.6, zorder=1)
        ax.scatter(*p, s=26, color=NBR, edgecolors="#2f6db5", linewidths=0.5, zorder=2)
    ax.scatter(*c1, s=46, color=FOCUS, zorder=3)
    agent_ring(ax, c1, s=95, color=AGENT, lw=1.0)
    c2 = np.array([xs[1], yi - 0.05])
    for p, lw in ((c2 + [-0.055, 0.12], 0.7), (c2 + [0.0, 0.25], 1.6), (c2 + [0.055, 0.12], 1.1)):
        ax.scatter(*p, s=26, color=NBR, edgecolors="#2f6db5", linewidths=0.5, zorder=2)
        arrow(ax, p, c2, color="#2f6db5", lw=lw, ms=5, sa=3, sb=4)
    ax.scatter(*c2, s=46, color=FOCUS, zorder=3)
    c3 = np.array([xs[2], yi])
    src = [(c3 + [-0.05, 0.13], dict(s=26, color="white", edgecolors=FOCUS, linewidths=1.0)),
           (c3 + [-0.05, -0.13], dict(s=26, color=NBR, edgecolors="#2f6db5", linewidths=0.5)),
           (c3 + [0.05, 0.13], dict(s=40, color=GBEST, marker="*", edgecolors="k", linewidths=0.3)),
           (c3 + [0.05, -0.13], dict(s=16, color=ELITE, marker="D", edgecolors="k", linewidths=0.3))]
    for p, kw in src:
        ax.scatter(*p, zorder=3, **kw)
        arrow(ax, p, c3, lw=0.7, ms=5, sa=4, sb=4)
    ax.scatter(*c3, s=46, color=AGENT, zorder=3)
    c4 = np.array([xs[3], yi])
    ax.scatter(*(c4 - [0.03, 0]), s=60, facecolors="white", edgecolors="#607d8b", linewidths=0.8,
               linestyle=(0, (2, 1.2)), zorder=3)
    arrow(ax, c4 - [0.03, 0], c4 + [0.03, 0], lw=0.9, ms=5, sa=5, sb=5)
    ax.scatter(*(c4 + [0.03, 0]), s=60, color=SNAP, zorder=3)
    c5 = np.array([xs[4], yi])
    w, h, f = 0.024, 0.15, 0.010
    doc = np.array([[-w, -h], [w, -h], [w, h - 3 * f], [w - f, h], [-w, h]]) + c5
    ax.add_patch(Polygon(doc, closed=True, fc="white", ec=INK, lw=0.7, zorder=3))
    ax.plot([c5[0] + w - f, c5[0] + w - f, c5[0] + w], [c5[1] + h, c5[1] + h - 3 * f, c5[1] + h - 3 * f], color=INK,
            lw=0.6, zorder=4)
    for dy in (0.06, 0.03):
        ax.plot([c5[0] - w + 0.006, c5[0] + w - 0.006], [c5[1] + dy] * 2, color="#90a4ae", lw=0.6, zorder=4)
    ax.text(c5[0], c5[1] - 0.06, "DFT", ha="center", va="center", fontsize=5.4, fontweight="bold", color=INK,
            zorder=4)
    # feedback loop
    yl = 0.03
    ax.plot([xs[4], xs[4], xs[0]], [yt - 0.10, yl, yl], color=INK, lw=0.8)
    arrow(ax, (xs[0], yl), (xs[0], yt - 0.10), lw=0.8, ms=6, sa=0, sb=0)
    ax.text(0.49, yl + 0.025, r"new value updates influence $I_j$, global best and elite memory", ha="center",
            va="bottom", fontsize=5.7, color=INK)


def main() -> None:
    fig = plt.figure(figsize=(7.26, 4.45))
    ax_a = fig.add_axes([0.01, 0.43, 0.27, 0.49])
    ax_n = fig.add_axes([0.315, 0.43, 0.34, 0.49])
    ax_e = fig.add_axes([0.675, 0.43, 0.31, 0.49])
    ax_c = fig.add_axes([0.02, 0.055, 0.96, 0.285])
    panel_social(ax_a, np.random.default_rng(3))
    panel_network(ax_n, np.random.default_rng(11))
    panel_embedding(ax_e, np.random.default_rng(5))
    panel_cycle(ax_c)
    legend_row(fig, 0.405)
    fig.add_artist(Line2D([0.298, 0.298], [0.43, 0.975], color="#b0b7c0", lw=0.6, ls=(0, (2, 2))))
    for x, y, letter, title in ((0.012, 0.965, "a", "SOCIAL (previous work)"),
                                (0.312, 0.965, "b", r"$\bf{Graph}$-$\bf{SOCIAL}$ (this study): the same agents in two views"),
                                (0.012, 0.355, "c", "One Graph-SOCIAL iteration (SOCIAL learning rules unchanged)")):
        fig.text(x, y, letter, fontsize=8.5, fontweight="bold", va="center")
        fig.text(x + 0.02, y, title, fontsize=7.2, va="center")
    fig.text(0.5, 0.018, r"$\alpha_t,\beta_t$ decrease and $\gamma_t,\delta_t$ increase over iterations; "
             "worse-than-median agents may jump to the least-explored chemical community",
             ha="center", va="center", fontsize=5.6, color=SUB, style="italic")
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"F0_graph_social_scheme.{ext}", dpi=600 if ext == "png" else None)
    print("wrote", OUT / "F0_graph_social_scheme.pdf")


if __name__ == "__main__":
    main()
