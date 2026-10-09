"""Figure 1: how the SOCIAL optimizer is used in this study (method overview, schematic; no result data).

(a) SOCIAL as published: agents in a continuous search space on a synthetic small-world graph.
(b) SOCIAL-MGN: the same agents in two views. Left, the MOF similarity network defines neighborhoods (agents within
    h = 2 hops) and betweenness; right, the chemistry-aware embedding is where the SOCIAL update moves an agent, which
    then snaps to the nearest unevaluated MOF. The update arrow is eq 5 with the default schedules (tau = 0.6).
(c) One SOCIAL-MGN iteration.

    python scripts/make_scheme.py      -> results/figures/F0_graph_social_scheme.{pdf,svg,png}
"""
from pathlib import Path
import matplotlib.font_manager as fm
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

FONT = "Liberation Sans" if "Liberation Sans" in {f.name for f in fm.fontManager.ttflist} else "Arial"
mpl.rcParams.update({
    "font.family": FONT, "font.size": 6.5,
    "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.5,
})

K = dict(rose="#E06C78", teal="#2A9D8F", orange="#E8963A", purple="#8E6CC9",
         grey="#A3AAB2", red="#C62828", dred="#8E1B1B", dark="#263238",
         green="#2E9E4F", gold="#F2B705", edge="#C9D1D9", blue="#2F6DB5")
COMM = [K["rose"], K["teal"], K["orange"], K["purple"], K["grey"]]
rng = np.random.default_rng(4)

W, H = 7.2, 4.6
fig = plt.figure(figsize=(W, H))


def blank(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def arrow(ax, p, q, color, lw=0.8, ls="-", ms=7, style="-|>", sa=2, sb=2,
          z=6, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms,
                 lw=lw, ls=ls, color=color, shrinkA=sa, shrinkB=sb, zorder=z,
                 connectionstyle=conn))


# ======================================================== panel a
top, ht = 0.505, 0.43
axa = fig.add_axes([0.012, top, 0.275, ht]); blank(axa)
th = np.linspace(0, 2 * np.pi, 400)
g = np.linspace(-1.25, 1.25, 500)
X, Y = np.meshgrid(g, g)
R = np.hypot(X, Y); T = np.arctan2(Y, X)
bound = 1.08 + 0.10 * np.sin(3 * T + 0.4) + 0.06 * np.cos(5 * T)
rn = R / bound + 0.06 * np.sin(4 * X) * np.cos(3 * Y)
Z = np.ma.masked_where(rn > 1.0, rn)
axa.contourf(X, Y, Z, levels=np.linspace(0, 1, 16), cmap="Blues",
             vmin=-0.3, vmax=2.6, zorder=0)
axa.contour(X, Y, Z, levels=np.linspace(0, 1, 16), colors="white",
            linewidths=0.3, alpha=0.6, zorder=1)
axa.plot(0, 0, "+", color=K["dark"], ms=6, mew=0.8, zorder=2)
axa.text(0, -0.13, "optimum at center", ha="center", va="top", fontsize=5.9,
         color=K["dark"], zorder=2)
n = 11
ws = nx.watts_strogatz_graph(n, 4, 0.3, seed=2)
ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.25
pa = {k: np.array([0.72 * np.cos(a), 0.72 * np.sin(a)]) +
      rng.normal(0, 0.04, 2) for k, a in enumerate(ang)}
for u, v in ws.edges():
    axa.plot(*zip(pa[u], pa[v]), color=K["dark"], lw=0.5, alpha=0.75, zorder=3)
for k, p in pa.items():
    axa.scatter(*p, s=20, color="#E53935", ec="white", lw=0.6, zorder=4)
ia = 0
axa.scatter(*pa[ia], s=34, color=K["dred"], ec="white", lw=0.7, zorder=5)
arrow(axa, pa[ia], pa[ia] * 0.45, K["dred"], lw=0.9, sa=4)
axa.text(pa[ia][0] + 0.06, pa[ia][1] + 0.07, r"$i$", fontsize=7.5,
         color=K["dred"])
axa.set_xlim(-1.25, 1.25); axa.set_ylim(-1.25, 1.25); axa.set_aspect("equal")
fig.text(0.012, 0.985, "a", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.032, 0.985, "SOCIAL (previous work)", fontsize=7.2, va="top")
fig.text(0.15, 0.955, "continuous search space,\nsynthetic small-world agent graph",
         fontsize=6.4, ha="center", va="top", color=K["dark"], linespacing=1.15)

fig.lines.append(Line2D([0.298, 0.298], [0.49, 0.985], transform=fig.transFigure,
                        ls=(0, (2, 2)), lw=0.5, color="#9AA4AE"))

# ======================================================== MOF pool
# community centres in network view and embedding view (deliberately different)
net_c = [(-0.55, 0.45), (0.6, 0.5), (-0.58, -0.5), (0.6, -0.52)]
emb_c = [(-0.5, 0.45), (0.5, 0.55), (-0.55, -0.5), (0.58, -0.5)]
sizes = [12, 11, 11, 10]
P_net, P_emb, comm = [], [], []
for c in range(4):
    m = sizes[c]; kk = np.arange(m) + 0.5
    rr = 0.19 * np.sqrt(kk / m); aa = kk * 2.39996 + c
    base = np.c_[rr * np.cos(aa), rr * np.sin(aa)] + rng.normal(0, 0.018, (m, 2))
    P_net.append(np.array(net_c[c]) + base)
    P_emb.append(np.array(emb_c[c]) + rng.normal(0, 0.14, (sizes[c], 2)))
    comm += [c] * sizes[c]
# grey connector MOFs (between families)
conn_net = np.array([[-0.1, 0.42], [0.18, 0.42], [0.0, 0.05], [-0.25, -0.2],
                     [0.25, -0.22], [-0.05, -0.55], [0.32, 0.12]])
conn_emb = np.array([[-0.12, 0.30], [0.12, 0.38], [0.02, 0.02], [-0.22, -0.18],
                     [0.24, -0.15], [0.0, -0.42], [0.35, 0.1]])
P_net.append(conn_net); P_emb.append(conn_emb); comm += [4] * len(conn_net)
Pn = np.vstack(P_net); Pe = np.vstack(P_emb); comm = np.array(comm)
N = len(comm)
starts = np.cumsum([0] + sizes)
cidx = list(range(starts[4], N))

G = nx.Graph(); G.add_nodes_from(range(N))
for a in range(N):
    for b in range(a + 1, N):
        if comm[a] == comm[b] < 4 and np.linalg.norm(Pn[a] - Pn[b]) < 0.24:
            G.add_edge(a, b)


def nearest(c, p):
    idx = [k for k in range(N) if comm[k] == c]
    return idx[int(np.argmin([np.linalg.norm(Pn[k] - p) for k in idx]))]


hub = cidx[2]
links = [(cidx[0], nearest(0, conn_net[0])), (cidx[0], cidx[1]),
         (cidx[1], nearest(1, conn_net[1])), (cidx[0], hub), (cidx[1], hub),
         (hub, cidx[3]), (hub, cidx[4]), (cidx[3], nearest(2, conn_net[3])),
         (cidx[4], nearest(3, conn_net[4])), (cidx[5], nearest(2, conn_net[5])),
         (cidx[5], nearest(3, conn_net[5])), (hub, cidx[6]),
         (cidx[6], nearest(1, conn_net[6]))]
G.add_edges_from(links)
btw = nx.betweenness_centrality(G)
bmax = max(btw.values())

# agents: two in rose (i and a 2-hop neighbour j), one per other family
rose = [k for k in range(N) if comm[k] == 0]
i_ag = rose[int(np.argmax([G.degree(k) for k in rose]))]
hop1 = set(G.neighbors(i_ag))
hop2 = set(nx.single_source_shortest_path_length(G, i_ag, cutoff=2))
j_ag = [k for k in hop2 if k != i_ag and comm[k] == 0 and k not in hop1]
j_ag = j_ag[0] if j_ag else sorted(hop1)[0]
others = [starts[1] + 2, starts[2] + 3, starts[3] + 4]
agents = [i_ag, j_ag] + others

# ======================================================== panel b, network
axn = fig.add_axes([0.312, top, 0.33, ht]); blank(axn)
for k in hop2:
    axn.scatter(*Pn[k], s=95, color="#F8C9CE", lw=0, alpha=0.75, zorder=0)
for u, v in G.edges():
    cross = comm[u] != comm[v] or comm[u] == 4
    axn.plot(*zip(Pn[u], Pn[v]), color="#AEB6BF" if cross else COMM[comm[u]],
             lw=0.45 if cross else 0.35, alpha=0.9 if cross else 0.45, zorder=1)
for k in range(N):
    axn.scatter(*Pn[k], s=8 + 45 * btw[k] / bmax, color=COMM[comm[k]],
                ec="white", lw=0.4, zorder=2)
axn.scatter(*Pn[hub], s=8 + 45, color=K["grey"], ec=K["blue"], lw=1.0, zorder=3)
for a in agents:
    axn.scatter(*Pn[a], s=58, facecolor="none", ec=K["red"], lw=0.9, zorder=4)
axn.scatter(*Pn[i_ag], s=62, facecolor="none", ec=K["dred"], lw=1.8, zorder=5)
axn.text(Pn[i_ag][0] + 0.07, Pn[i_ag][1] + 0.09, r"$i$", fontsize=7.5,
         color=K["dred"], zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
axn.text(Pn[j_ag][0] - 0.09, Pn[j_ag][1] - 0.02, r"$j$", fontsize=7.5,
         color=K["red"], ha="right", va="top", zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
arrow(axn, Pn[j_ag], Pn[i_ag], K["red"], lw=0.7, ls=(0, (2, 1.5)), ms=6,
      sa=4, sb=5)
axn.annotate("bridge MOF\n(high $c_j$)", xy=Pn[hub], xytext=(0.18, -0.02),
             fontsize=6.2, ha="left", va="center", color=K["dark"],
             arrowprops=dict(arrowstyle="-", lw=0.4, color=K["dark"],
                             shrinkB=4))
axn.annotate("neighbors of $i$:\nagents within $h$ = 2 hops", xy=Pn[i_ag] + [-0.12, -0.12],
             xytext=(-0.92, 0.12), fontsize=6.2, ha="left", va="top", color=K["dark"],
             arrowprops=dict(arrowstyle="-", lw=0.4, color=K["dark"]))
axn.set_xlim(-0.92, 0.92); axn.set_ylim(-0.83, 0.92); axn.set_aspect("equal")
fig.text(0.312, 0.985, "b", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.332, 0.985, r"$\bf{SOCIAL}$-$\bf{MGN}$ (this study): the same agents in two views",
         fontsize=7.2, va="top")
fig.text(0.477, 0.955, "MOF similarity network: who listens to whom",
         fontsize=6.4, ha="center", va="top", color=K["dark"])

# ======================================================== panel b, embedding
axe = fig.add_axes([0.655, top, 0.335, ht]); blank(axe)
evaluated = set(rng.choice([k for k in range(N) if k not in agents
                            and comm[k] < 4], 11, replace=False))
Pe[i_ag] = np.array(emb_c[0]) + np.array([0.2, -0.06])
Pe[j_ag] = np.array(emb_c[0]) + np.array([0.34, 0.24])
for k in range(N):  # keep other rose points clear of the two agents
    if comm[k] == 0 and k not in (i_ag, j_ag):
        for a in (i_ag, j_ag):
            d = Pe[k] - Pe[a]; dn = np.linalg.norm(d)
            if dn < 0.12:
                Pe[k] = Pe[a] + d / (dn + 1e-9) * 0.12
xi, xj = Pe[i_ag], Pe[j_ag]
gbest = Pe[others[0]]
pur_ev = [k for k in evaluated if comm[k] == 3]
elite_k = pur_ev[int(np.argmax(Pe[pur_ev][:, 1]))] if pur_ev else starts[3]
elite = Pe[elite_k]
# SOCIAL update, eq 5: defaults alpha = beta = 0.4, gamma = delta = 0.2 at tau = t/T = 0.6
tau = 0.6
at, bt, gt, dt = 0.4 * (1 - tau), 0.4 * (1 - tau), 0.2 * tau, 0.2 * tau
xnew = (1 - at - bt - gt - dt) * xi + (at + bt) * xj + gt * gbest + dt * elite
# keep pool points clear of the agent, global-best and elite markers
for k in range(N):
    if k in agents or k == elite_k:
        continue
    for anchor in (gbest, elite, xi, xj):
        d = Pe[k] - anchor; dn = np.linalg.norm(d)
        if 1e-9 < dn < 0.11:
            Pe[k] = anchor + d / dn * 0.11
unev = [k for k in range(N) if k not in evaluated and k not in agents]
for k in unev:
    d = Pe[k] - xnew; dn = np.linalg.norm(d)
    if k != cidx[0] and dn < 0.26:
        Pe[k] = xnew + d / dn * 0.26
others_d = min(np.linalg.norm(Pe[k] - xnew) for k in unev if k != cidx[0])
for off in [(0.13, -0.15), (0.15, -0.08), (0.12, 0.12), (0.0, -0.17), (-0.1, -0.14), (0.1, -0.1), (0.08, -0.06)]:
    if np.linalg.norm(off) < others_d - 0.01:
        break
Pe[cidx[0]] = xnew + np.array(off)  # a connector MOF near the updated position
snap = unev[int(np.argmin(np.linalg.norm(Pe[unev] - xnew, axis=1)))]
assert snap == cidx[0], "snap target is not the intended MOF"

for k in range(N):
    if k == elite_k:
        continue
    if k in evaluated:
        axe.scatter(*Pe[k], marker="x", s=11, color="#9AA4AE", lw=0.6, zorder=2)
    elif k != snap:
        axe.scatter(*Pe[k], s=10, color=COMM[comm[k]], ec="white", lw=0.3,
                    zorder=2)
ctr = Pe.mean(0)
axe.scatter(*ctr, marker="+", s=45, color=K["dark"], lw=0.9, zorder=3)
# label below-left of the centroid: clear of the x_i -> elite arrow and of the snap labels
axe.text(ctr[0] - 0.06, ctr[1] - 0.05, "centroid", fontsize=5.8, color=K["dark"], ha="right", va="top", zorder=8,
         bbox=dict(fc="white", ec="none", alpha=0.9, pad=0.4))
axe.scatter(*gbest, s=60, facecolor="white", ec=K["red"], lw=0.9, zorder=4)
axe.scatter(*gbest, marker="*", s=55, color=K["gold"], ec=K["dark"], lw=0.35,
            zorder=5)
axe.text(gbest[0] + 0.1, gbest[1] - 0.02, "global best", fontsize=6.2, ha="left", va="center", zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
axe.scatter(*elite, marker="D", s=26, color="#EF6C00", ec=K["dark"], lw=0.4,
            zorder=5)
axe.text(elite[0] + 0.08, elite[1] + 0.02, "elite memory", fontsize=6.2,
         ha="left", va="center", zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
for a in agents[3:]:
    axe.scatter(*Pe[a], s=40, facecolor="white", ec=K["red"], lw=0.9, zorder=4)
    axe.scatter(*Pe[a], s=9, color=COMM[comm[a]], zorder=5)
axe.scatter(*xj, s=40, facecolor="white", ec=K["red"], lw=0.9, zorder=4)
axe.scatter(*xj, s=9, color=COMM[0], zorder=5)
axe.scatter(*xi, s=52, facecolor="white", ec=K["dred"], lw=1.8, zorder=6)
axe.scatter(*xi, s=12, color=K["dred"], zorder=7)
axe.text(xi[0] - 0.09, xi[1] + 0.02, r"$\mathbf{x}_i$", fontsize=7.5,
         color=K["dred"], ha="right", va="center", zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
axe.text(xj[0] + 0.07, xj[1] + 0.05, r"$j$", fontsize=7.5, color=K["red"], zorder=8, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
# pulls on x_i
arrow(axe, xi, xj, K["rose"], lw=0.6, ls=(0, (2, 1.5)), ms=5, sa=5, sb=5)
arrow(axe, xi, gbest, "#D99A00", lw=0.6, ls=(0, (2, 1.5)), ms=5, sa=5, sb=6)
arrow(axe, xi, elite, "#EF6C00", lw=0.6, ls=(0, (2, 1.5)), ms=5, sa=5, sb=5)
# resulting move and snap
arrow(axe, xi, xnew, K["dred"], lw=1.1, ms=8, sa=5, sb=4, z=7)
axe.scatter(*xnew, s=46, facecolor="white", ec=K["dred"], lw=0.8,
            ls=(0, (2, 1.2)), zorder=7)
axe.text(xi[0] - 0.06, xi[1] - 0.09, "SOCIAL update\n(Eq. 5)", fontsize=5.9,
         color=K["dred"], ha="right", va="top", linespacing=1.05, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
arrow(axe, xnew, Pe[snap], K["green"], lw=1.2, ms=8, sa=5, sb=5, z=7)
axe.scatter(*Pe[snap], s=48, color=K["green"], ec="white", lw=0.6, zorder=8)
mid = (xnew + Pe[snap]) / 2
axe.text(mid[0] + 0.07, mid[1] + 0.04, "snap", fontsize=6.8, fontweight="bold",
         color=K["green"], ha="left", va="center")
axe.text(Pe[snap][0] + 0.08, Pe[snap][1] - 0.01, "nearest\nunevaluated MOF",
         fontsize=5.9, ha="left", va="top", color=K["dark"], linespacing=1.0, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.4))
axe.set_xlim(-0.95, 1.0); axe.set_ylim(-0.83, 0.92); axe.set_aspect("equal")
axe.plot([-0.95, -0.95, 1.0], [0.92, -0.83, -0.83], color="#B0B8C0", lw=0.5)
fig.text(0.822, 0.955, "chemistry-aware embedding: where agents move",
         fontsize=6.4, ha="center", va="top", color=K["dark"])

# shared legend for b
hd = [Line2D([], [], marker="o", ls="", ms=3.4, color=c) for c in COMM[:4]]
hd += [Line2D([], [], marker="o", ls="", ms=3.4, color=K["grey"]),
       Line2D([], [], marker="o", ls="", ms=5, mfc="white", mec=K["red"], mew=0.9),
       Line2D([], [], marker="x", ls="", ms=3.6, color="#9AA4AE", mew=0.6),
       Line2D([], [], marker="o", ls="", ms=6.5, color="#F8C9CE")]
fig.legend(hd, ["", "", "", "MOF families", "connector MOF", "agent",
                "evaluated MOF", "2-hop neighborhood of $i$"],
           loc="center", bbox_to_anchor=(0.648, 0.478), ncol=8, frameon=False,
           fontsize=6.0, handletextpad=0.15, columnspacing=0.55)

# ======================================================== panel c
axc = fig.add_axes([0.0, 0.0, 1.0, 0.44]); blank(axc)
wc, hc = W * 1.0, H * 0.44
axc.set_xlim(0, 100); axc.set_ylim(0, 100 * hc / wc); axc.set_aspect("equal")
ytop = 100 * hc / wc
fig.text(0.012, 0.445, "c", fontsize=8.5, fontweight="bold", va="top")
fig.text(0.032, 0.445, "One SOCIAL-MGN iteration (SOCIAL learning rules unchanged)",
         fontsize=7.2, va="top")

xs = [9, 29, 50, 71, 90]
ytitle, yicon, ydesc = ytop - 4.6, ytop - 10.8, ytop - 16.0
cols = ["#9EC5E8", "#7FCFC4", "#F28B82", "#6CC07A", "#F6C445"]
titles = ["Neighborhoods", "Weights", "SOCIAL update", "Snap", "Evaluate"]
descs = ["agents within $h$ hops\nin the MOF network",
         r"$w_{ij} \propto \alpha_t c_j + \beta_t I_j$",
         r"$\mathbf{x}_i \leftarrow$ self + $(\alpha_t + \beta_t)$ neighbors" "\n"
         r"+ $\gamma_t$ global best + $\delta_t$ elite",
         "nearest MOF\nnot yet evaluated",
         "DFT value revealed (look-up);\neach MOF once; budget $B$"]
for k, x in enumerate(xs):
    tw = 3.1 + 0.6 + 0.78 * len(titles[k]); lft = x - tw / 2
    axc.add_patch(Circle((lft + 1.55, ytitle), 1.55, color=cols[k], zorder=3))
    axc.text(lft + 1.55, ytitle, str(k + 1), ha="center", va="center",
             fontsize=7, fontweight="bold", color=K["dark"], zorder=4)
    axc.text(lft + 3.7, ytitle, titles[k], ha="left", va="center",
             fontsize=6.6, fontweight="bold", color=K["dark"])
    axc.text(x, ydesc, descs[k], ha="center", va="center", fontsize=6.5,
             linespacing=1.2, color=K["dark"])
    if k < 4:
        nt = 3.7 + 0.78 * len(titles[k + 1])
        arrow(axc, (x + tw / 2 + 1.2, ytitle),
              (xs[k + 1] - nt / 2 - 1.2, ytitle), K["dark"], lw=0.6, ms=6,
              sa=0, sb=0)

# icon 1: small graph, centre ringed
c1 = np.array([xs[0], yicon]); nb = [(-3.2, 0.9), (-1, 2.4), (2.6, 1.6), (3.3, -0.6), (0.6, -2.2)]
for d in nb:
    axc.plot(*zip(c1, c1 + d), color="#7A8794", lw=0.5, zorder=1)
for d in nb:
    axc.add_patch(Circle(c1 + d, 0.65, fc="#9EC5E8", ec=K["blue"], lw=0.5, zorder=2))
axc.add_patch(Circle(c1, 0.85, fc=K["dred"], ec="white", lw=0.4, zorder=3))
axc.add_patch(Circle(c1, 1.25, fc="none", ec=K["red"], lw=0.8, zorder=3))
# icon 2: neighbours -> i, arrow width = weight
c2 = np.array([xs[1], yicon - 0.4]); nb2 = [(-3.4, 1.3), (0, 2.8), (3.4, 1.3)]
wts = [0.5, 1.4, 0.9]
for d, w in zip(nb2, wts):
    axc.add_patch(Circle(c2 + d, 0.65, fc="#9EC5E8", ec=K["blue"], lw=0.5, zorder=2))
    arrow(axc, tuple(c2 + d), tuple(c2), K["blue"], lw=w, ms=4 + 2 * w,
          sa=4, sb=5)
axc.add_patch(Circle(c2, 0.9, fc=K["dred"], ec="white", lw=0.4, zorder=3))
# icon 3: self, neighbor, global best, elite -> i
c3 = np.array([xs[2], yicon - 0.2])
axc.add_patch(Circle(c3 + (-3.5, 1.8), 0.65, fc="white", ec=K["dred"], lw=0.9, zorder=2))
axc.add_patch(Circle(c3 + (-3.2, -1.6), 0.65, fc="#9EC5E8", ec=K["blue"], lw=0.5, zorder=2))
axc.scatter(*(c3 + (3.5, 1.8)), marker="*", s=60, color=K["gold"], ec=K["dark"], lw=0.35, zorder=3)
axc.scatter(*(c3 + (3.3, -1.6)), marker="D", s=20, color="#EF6C00", ec=K["dark"], lw=0.35, zorder=3)
for d in [(-3.5, 1.8), (-3.2, -1.6), (3.5, 1.8), (3.3, -1.6)]:
    arrow(axc, tuple(c3 + d), tuple(c3), K["dark"], lw=0.6, ms=5, sa=4, sb=5)
axc.add_patch(Circle(c3, 0.95, fc=K["red"], ec="white", lw=0.4, zorder=3))
# icon 4: dashed position -> MOF
c4 = np.array([xs[3], yicon])
axc.add_patch(Circle(c4 + (-2.8, 0), 1.0, fc="white", ec="#5F6B76", lw=0.7,
                     ls=(0, (2, 1.4)), zorder=2))
arrow(axc, tuple(c4 + (-1.6, 0)), tuple(c4 + (1.6, 0)), K["dark"], lw=0.8,
      ms=6, sa=0, sb=0)
axc.add_patch(Circle(c4 + (2.8, 0), 1.0, fc=K["green"], ec="white", lw=0.5, zorder=2))
# icon 5: document labelled DFT
c5 = np.array([xs[4], yicon]); w5, h5 = 3.4, 4.2
x0, y0 = c5[0] - w5 / 2, c5[1] - h5 / 2
axc.add_patch(Polygon([(x0, y0), (x0 + w5, y0), (x0 + w5, y0 + h5 - 1),
                       (x0 + w5 - 1, y0 + h5), (x0, y0 + h5)], closed=True,
                      fc="white", ec=K["dark"], lw=0.7, zorder=2))
axc.plot([x0 + w5 - 1, x0 + w5 - 1, x0 + w5], [y0 + h5, y0 + h5 - 1, y0 + h5 - 1],
         color=K["dark"], lw=0.6, zorder=3)
for yy in [y0 + h5 - 1.6, y0 + h5 - 2.3]:
    axc.plot([x0 + 0.6, x0 + w5 - 0.7], [yy, yy], color="#9AA4AE", lw=0.6, zorder=3)
axc.text(c5[0], y0 + 0.95, "DFT", ha="center", va="center", fontsize=6.2,
         fontweight="bold", color=K["dark"], zorder=3)

# feedback loop
yl = 6.0
axc.plot([xs[4], xs[4], xs[0]], [ydesc - 2.2, yl, yl], color=K["dark"], lw=0.6)
arrow(axc, (xs[0], yl), (xs[0], ydesc - 2.0), K["dark"], lw=0.6, ms=6, sa=0, sb=0)
axc.text(50, yl + 0.5, "new value updates influence $I_j$, global best and elite memory",
         ha="center", va="bottom", fontsize=6.3, color=K["dark"])
axc.text(50, 2.0, r"$\alpha_t, \beta_t$ decrease and $\gamma_t, \delta_t$ increase over "
         "iterations; worse-than-median agents may jump to the least-explored "
         "chemical community", ha="center", va="bottom", fontsize=6.0,
         color="#37424A", style="italic")

for ext in ["pdf", "svg", "png", "tiff"]:
    kw = dict(dpi=600) if ext in ("png", "tiff") else {}
    if ext == "tiff":
        continue
    fig.savefig(Path(__file__).resolve().parents[1] / "results" / "figures" / f"F0_graph_social_scheme.{ext}",
                bbox_inches="tight", pad_inches=0.03, **kw)
print("wrote results/figures/F0_graph_social_scheme.{pdf,svg,png}")
