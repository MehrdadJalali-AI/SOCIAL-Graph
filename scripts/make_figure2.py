"""Figure 2: the QMOF similarity network at phi* (3,000-MOF sample) with zoom panels and MOF labels.

    python scripts/make_figure2.py   -> results/figures/Figure2.{pdf,svg,png,tiff}, results/figures/fig2_layout.csv

Data are the pipeline's: the phi* network, Leiden partition, max-normalized betweenness, PBE gaps and the same
3,000-node BFS sample and Fruchterman-Reingold layout as stage 8 (plots._giant_sample, seed 0). The layout is saved
to fig2_layout.csv and reused on later runs.

Panels: (a) Leiden communities with boxes marking the zoom regions, (b) PBE band gap, and three zooms, each with its
own local layout (Kamada-Kawai, deterministic) of the selected MOFs and their neighbors:
  (c) the largest building-block group (identical linker and metal) in the sample, plus the 20 neighbors with the most
      edges into the group (open circles in their community color);
  (d) the highest-betweenness MOF in the center with all its neighbors, one Leiden community on each side;
  (e) the O2 hit with the most other O2 hits within 6% of the layout span, those hits and the 25 neighbors with the most
      edges into them.
If a zoom region would overlap an earlier one in panel a, the next-best region under the same rule is taken. Chosen
nodes are printed.

Labels come from QMOF metadata only: the CSD refcode for MOFs whose source is the CSD, otherwise the QMOF ID; band
gaps are the QMOF PBE values. The building-block tag of panel c is "metal | linker formula", with the formula computed
by RDKit from the QMOF linker SMILES. No common names are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsocial import config as C, objectives as O, plan, plots  # noqa: E402
from graphsocial.graph.topologies import Topology  # noqa: E402
from graphsocial.store import Store, graph_name  # noqa: E402

FONT = "Liberation Sans" if "Liberation Sans" in {f.name for f in fm.fontManager.ttflist} else "Arial"
plt.rcParams.update({"font.family": FONT, "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.linewidth": 0.5})
OUT = ROOT / "results" / "figures"
EDGE = "#c9d1d9"
OTHER = "#d5dbe1"
LABEL_PT = 5.6
GAPMAP = "viridis"


# --------------------------------------------------------------------------------------------- data
def load():
    cfg = C.load_config(None)
    st = Store(cfg)
    phi = plan.phi_star(cfg)
    top = Topology.load(st.graph(graph_name(phi)))
    bc = st.centralities(graph_name(phi))["betweenness"]
    comm = np.load(st.communities(phi)).astype(int)
    df = st.load_clean()
    nodes, sub, lay = plots._giant_sample(top, bc, 3000, 0)
    lay_path = OUT / "fig2_layout.csv"
    if lay_path.exists():
        saved = pd.read_csv(lay_path)
        assert np.array_equal(saved.node.to_numpy(), nodes), "saved layout does not match the 3,000-node sample"
        lay = saved[["x", "y"]].to_numpy()
    else:
        pd.DataFrame({"node": nodes, "qmof_id": df.qmof_id.to_numpy()[nodes], "x": lay[:, 0], "y": lay[:, 1]}).to_csv(
            lay_path, index=False)
    o2 = O.make("O2", df, cfg["objectives"]["hit_fraction"])
    hit = np.zeros(len(df), dtype=bool)
    hit[o2.universe[o2.hits]] = True
    return dict(df=df, nodes=nodes, sub=sub, lay=lay, bc=bc, comm=comm, hit=hit, phi=phi)


def label_of(df: pd.DataFrame, i: int) -> tuple[str, bool]:
    r = df.iloc[i]
    if r.source == "CSD" and isinstance(r.refcode, str) and r.refcode:
        return r.refcode, True
    return str(r.qmof_id), False


def linker_formula(smiles: str) -> str | None:
    """Molecular formula of the linker from its QMOF SMILES (RDKit), typeset with subscripts and charge."""
    import re

    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    f = CalcMolFormula(m)
    body, charge = re.match(r"^(.*?)([+-]\d*)?$", f).groups()
    body = re.sub(r"(\d+)", r"_{\1}", body)
    ch = "" if not charge else "^{" + (charge[1:] or "") + ("-" if charge[0] == "-" else "+") + "}"
    return r"$\mathrm{" + body + ch + "}$"


# --------------------------------------------------------------------------------------------- selection rules
N_NEIGH_C, N_NEIGH_E = 20, 25


def box_around(pts: np.ndarray, span: float, margin: float = 0.25, min_half: float = 0.025):
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    half = max(float(((hi - lo) / 2).max()) * (1 + margin), min_half * span)
    return c[0] - half, c[1] - half, c[0] + half, c[1] + half


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def candidates_c(D):
    df, nodes, bc = D["df"], D["nodes"], D["bc"]
    d = df.iloc[nodes].assign(node=nodes, bc=bc[nodes])
    grp = d.groupby("bb_group").agg(n=("node", "size"), bcs=("bc", "sum")).sort_values(["n", "bcs"], ascending=False)
    for g in grp.index:
        yield {"group": int(g), "nodes": d[d.bb_group == g].node.to_numpy()}


def candidates_d(D):
    nodes, bc, sub = D["nodes"], D["bc"], D["sub"]
    for k in np.argsort(-bc[nodes], kind="stable"):
        yield {"bridge": int(nodes[k]), "nodes": np.array([int(nodes[j]) for j in sub.neighbors(int(k))])}


def candidates_e(D):
    nodes, lay, hit = D["nodes"], D["lay"], D["hit"]
    span = float(np.ptp(lay, axis=0).max())
    H = np.array([k for k, v in enumerate(nodes) if hit[v]])
    dd = np.linalg.norm(lay[H][:, None] - lay[H][None], axis=-1)
    R = 0.06 * span
    cnt = (dd <= R).sum(1)
    for b in np.lexsort((np.where(dd <= R, dd, 0).sum(1) / cnt, -cnt)):
        yield {"nodes": np.array([int(nodes[k]) for k in H[dd[b] <= R]])}


def select(D: dict) -> dict:
    df, lay, nodes = D["df"], D["lay"], D["nodes"]
    idx = {int(v): k for k, v in enumerate(nodes)}
    span = float(np.ptp(lay, axis=0).max())
    core = {"c": lambda s: s["nodes"], "d": lambda s: [s["bridge"]], "e": lambda s: s["nodes"]}
    gens = {"c": candidates_c(D), "d": candidates_d(D), "e": candidates_e(D)}
    S, boxes = {}, []
    for key in "cde":
        for rank, cand in enumerate(gens[key]):
            box = box_around(lay[[idx[v] for v in core[key](cand)]], span)
            if not any(overlaps(box, b) for b in boxes):
                cand.update(box=box, rank=rank)
                S[key] = cand
                boxes.append(box)
                if rank:
                    print(f"{key}: best region overlapped an earlier box; using rank {rank} under the same rule")
                break
    c, d, e = S["c"], S["d"], S["e"]
    print("c: building-block group", c["group"], "|", len(c["nodes"]), "MOFs (rank", c["rank"], "):",
          ", ".join(df.qmof_id.to_numpy()[c["nodes"]][:10]), "...")
    print("d: bridge", df.qmof_id[d["bridge"]], label_of(df, d["bridge"])[0], "(rank", d["rank"], ") | 1-hop neighbors:",
          len(d["nodes"]), "| communities:", pd.Series(D["comm"][d["nodes"]]).value_counts().to_dict())
    print("e: O2 hits in region (rank", e["rank"], "):", ", ".join(df.qmof_id.to_numpy()[e["nodes"]]))
    return S


# --------------------------------------------------------------------------------------------- local layouts
def neighbors_most_connected(D, core, n):
    """The n non-core neighbors with the most edges into ``core`` (ties: higher betweenness, then node id)."""
    nodes, sub, bc = D["nodes"], D["sub"], D["bc"]
    idx = {int(v): k for k, v in enumerate(nodes)}
    cnt: dict[int, int] = {}
    cs = set(map(int, core))
    for v in core:
        for k in sub.neighbors(idx[int(v)]):
            u = int(nodes[k])
            if u not in cs:
                cnt[u] = cnt.get(u, 0) + 1
    return sorted(cnt, key=lambda u: (-cnt[u], -bc[u], u))[:n]


def local_graph(D, members):
    import networkx as nx

    nodes, sub = D["nodes"], D["sub"]
    idx = {int(v): k for k, v in enumerate(nodes)}
    ms = list(map(int, members))
    G = nx.Graph()
    G.add_nodes_from(ms)
    ks = {idx[v]: v for v in ms}
    for k, v in ks.items():
        for j in sub.neighbors(k):
            if j in ks and v < ks[j]:
                G.add_edge(v, ks[j])
    return G


def kk_layout(G, seed=0):
    import networkx as nx

    init = nx.spring_layout(G, seed=seed)
    return nx.kamada_kawai_layout(G, pos=init)


def bridge_layout(D, bridge, nbrs):
    """Bridge at the origin; neighbors grouped by Leiden community, each community laid out on its own side."""
    import networkx as nx

    comm = D["comm"]
    pos = {bridge: np.array([0.0, 0.0])}
    groups = pd.Series({v: comm[v] for v in nbrs}).groupby(lambda v: comm[v]).groups
    order = sorted(groups, key=lambda c: -len(groups[c]))
    angles = np.linspace(np.pi, -np.pi, len(order), endpoint=False) if len(order) > 2 else [np.pi, 0.0][: len(order)]
    for c, a in zip(order, angles):
        members = list(groups[c])
        G = local_graph(D, members)
        lp = kk_layout(G) if len(members) > 2 else {v: np.array([0.0, k * 0.5]) for k, v in enumerate(members)}
        P = np.array([lp[v] for v in members])
        P = P - P.mean(0)
        scale = 0.55 * np.sqrt(len(members) / max(len(nbrs), 1)) + 0.25
        P = P / max(np.abs(P).max(), 1e-9) * scale
        center = 1.15 * np.array([np.cos(a), np.sin(a)])
        for v, q in zip(members, P):
            pos[v] = q + center
    return pos


# --------------------------------------------------------------------------------------------- drawing
def top_communities(comm_s: np.ndarray) -> list[int]:
    return list(pd.Series(comm_s).value_counts().index[:10])


def comm_color(c: int, top10: list[int]) -> str:
    return plots.PALETTE[top10.index(c)] if c in top10 else "#b9c1c9"


def draw_local(ax, G, pos, colors, sizes, edge_extra=(), hollow=()):
    segs = [[pos[u], pos[v]] for u, v in G.edges()] + [list(e) for e in edge_extra]
    ax.add_collection(LineCollection(segs, colors="#8a96a3", linewidths=0.3, alpha=0.3, zorder=1))
    vs = list(G.nodes) if not hasattr(G, "draw_nodes") else G.draw_nodes
    P = np.array([pos[v] for v in vs])
    full = [v for v in vs if v not in hollow]
    if full:
        Q = np.array([pos[v] for v in full])
        ax.scatter(Q[:, 0], Q[:, 1], s=[sizes[v] for v in full], c=[colors[v] for v in full], linewidths=0.4,
                   edgecolors="white", zorder=2)
    ring = [v for v in vs if v in hollow]  # neighbors of the selected MOFs: open circles in their community color
    if ring:
        Q = np.array([pos[v] for v in ring])
        ax.scatter(Q[:, 0], Q[:, 1], s=[sizes[v] for v in ring], facecolors="white",
                   edgecolors=[colors[v] for v in ring], linewidths=0.8, zorder=2)
    return P


def finish_zoom(ax, P, letter, title):
    lo, hi = P.min(0), P.max(0)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * 1.18 + 0.08 * (hi - lo).max()
    ax.set_xlim(c[0] - half[0], c[0] + half[0])
    ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#9aa4ae")
        sp.set_linewidth(0.6)
    ax.text(0.0, 1.025, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    ax.text(0.075, 1.025, title, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom", color="#263238")


def add_labels(ax, pos, label_nodes, df, P_all, bold=(), unresolved=None, obstacles=()):
    from adjustText import adjust_text

    texts, fixed = [], []
    for v in label_nodes:
        lab, ok = label_of(df, v)
        if not ok and unresolved is not None:
            unresolved.append(lab)
        txt = f"{lab} · {df.pbe_gap[v]:.2f} eV"
        box = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
        if v in bold:  # pinned directly below its node; the other labels avoid it
            fixed.append(ax.text(pos[v][0], pos[v][1] - 0.16, txt, fontsize=LABEL_PT, zorder=6, fontweight="bold",
                                 ha="center", va="top", bbox=box))
        else:
            texts.append(ax.text(pos[v][0], pos[v][1], txt, fontsize=LABEL_PT, zorder=6, bbox=box))
    ax.figure.canvas.draw()
    anchors = [pos[v] for v in label_nodes if v not in bold]
    adjust_text(texts, x=P_all[:, 0], y=P_all[:, 1], objects=(fixed + list(obstacles)) or None, ax=ax, expand=(1.3, 1.6),
                force_text=(0.6, 0.8), force_static=(0.4, 0.6), ensure_inside_axes=True, max_move=None)
    # leader lines: from each node to the nearest point of its (moved) label box
    ax.figure.canvas.draw()
    r = ax.figure.canvas.get_renderer()
    inv = ax.transData.inverted()
    for t, (x, y) in zip(texts, anchors):
        (bx0, by0), (bx1, by1) = inv.transform(t.get_window_extent(r).get_points())
        qx, qy = min(max(x, bx0), bx1), min(max(y, by0), by1)
        if (qx, qy) != (x, y):
            ax.plot([x, qx], [y, qy], color="#37424a", lw=0.35, zorder=5)
    return texts + fixed + list(obstacles)


def covers_node(ax, bb, P) -> bool:
    """True if any node position (data coordinates) lies inside the display-space box ``bb``, with a small margin."""
    xy = ax.transData.transform(P)
    return bool(((xy[:, 0] >= bb.x0 - 3) & (xy[:, 0] <= bb.x1 + 3) & (xy[:, 1] >= bb.y0 - 3) & (xy[:, 1] <= bb.y1 + 3)).any())


def crosses_leader(ax, bb, r) -> bool:
    """True if any leader line of ``ax`` passes through the display-space box ``bb`` (sampled along the line)."""
    for ln in ax.lines:
        xy = ax.transData.transform(np.column_stack(ln.get_data()))
        t = np.linspace(0, 1, 50)[:, None]
        pts = xy[0] + t * (xy[-1] - xy[0])
        if ((pts[:, 0] >= bb.x0 - 2) & (pts[:, 0] <= bb.x1 + 2) & (pts[:, 1] >= bb.y0 - 2) & (pts[:, 1] <= bb.y1 + 2)).any():
            return True
    return False


def check_overlaps(fig, axes_texts) -> list[str]:
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ts in axes_texts:
        bbs = [(t.get_text(), t.get_window_extent(r)) for t in ts]
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                if bbs[i][1].overlaps(bbs[j][1]):
                    bad.append(f"{bbs[i][0]} / {bbs[j][0]}")
    return bad


def main() -> None:
    D = load()
    S = select(D)
    df, nodes, lay, bc, comm = D["df"], D["nodes"], D["lay"], D["bc"], D["comm"]
    cs = comm[nodes]
    top10 = top_communities(cs)
    ccols = [comm_color(c, top10) for c in cs]
    gap_all = df.pbe_gap.to_numpy()
    gap = gap_all[nodes]
    vmax = float(np.quantile(gap, 0.99))
    norm = matplotlib.colors.Normalize(0, vmax)
    cmap = matplotlib.colormaps[GAPMAP]
    size_main = 1.5 + 30 * bc[nodes] / bc[nodes].max()

    fig = plt.figure(figsize=(7.2, 5.15), facecolor="white")
    ax_a = fig.add_axes([0.0, 0.375, 0.49, 0.6])
    ax_b = fig.add_axes([0.47, 0.375, 0.45, 0.6])
    cax = fig.add_axes([0.93, 0.46, 0.011, 0.42])
    ax_z = [fig.add_axes([0.012 + k * 0.332, 0.012, 0.31, 0.33]) for k in range(3)]
    el = np.asarray(D["sub"].get_edgelist())
    for ax in (ax_a, ax_b):
        ax.add_collection(LineCollection(lay[el], colors=EDGE, linewidths=0.2, alpha=0.15, zorder=1, rasterized=True))
        ax.set_aspect("equal")
        ax.set_axis_off()
    ax_a.scatter(lay[:, 0], lay[:, 1], s=size_main, c=ccols, linewidths=0, zorder=2, rasterized=True)
    sc = ax_b.scatter(lay[:, 0], lay[:, 1], s=size_main, c=gap, cmap=GAPMAP, norm=norm, linewidths=0, zorder=2,
                      rasterized=True)
    for ax in (ax_a, ax_b):
        ax.autoscale()
    ax_b.set_xlim(ax_a.get_xlim())
    ax_b.set_ylim(ax_a.get_ylim())
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("PBE band gap (eV)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    for x, letter, title in ((0.012, "a", "Leiden communities"), (0.475, "b", "PBE band gap")):
        fig.text(x, 0.99, letter, fontsize=8.5, fontweight="bold", va="top")
        fig.text(x + 0.022, 0.988, title, fontsize=6.5, va="top", color="#263238")
    for key in "cde":
        x0, y0, x1, y1 = S[key]["box"]
        ax_a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.015",
                                      fc="none", ec="#37424a", lw=0.8, zorder=4))
        ax_a.text(x1, y1, key, fontsize=7, fontweight="bold", ha="left", va="bottom", color="#263238", zorder=5,
                  bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))

    unresolved, all_texts = [], []
    zsize = lambda v, base=14: base + 70 * bc[v] / bc[nodes].max()  # noqa: E731
    ccol = lambda v: comm_color(comm[v], top10)  # noqa: E731

    # c: building-block group + most-connected neighbors
    c = S["c"]
    nb_c = neighbors_most_connected(D, c["nodes"], N_NEIGH_C)
    Gc = local_graph(D, list(c["nodes"]) + nb_c)
    pc = kk_layout(Gc)
    core_c = set(map(int, c["nodes"]))
    colors = {v: ccol(v) for v in Gc.nodes}
    sizes = {v: zsize(v) for v in Gc.nodes}
    Pc = draw_local(ax_z[0], Gc, pc, colors, sizes, hollow=set(Gc.nodes) - core_c)
    r0 = df.iloc[c["nodes"][0]]
    finish_zoom(ax_z[0], Pc, "c", f"Building-block near-clique: {r0.metal} | {linker_formula(r0.linker_smiles)}")
    members = sorted(map(int, c["nodes"]), key=lambda v: -bc[v])
    lab_c, seen = [], set()
    for v in members:
        if label_of(df, v)[0] not in seen and len(lab_c) < 7:
            lab_c.append(v)
            seen.add(label_of(df, v)[0])
    all_texts.append(add_labels(ax_z[0], pc, lab_c, df, Pc, unresolved=unresolved))

    # d: bridge in the center, neighbor families on either side
    d = S["d"]
    nb_d = list(map(int, d["nodes"]))
    pdl = bridge_layout(D, d["bridge"], nb_d)
    Gd = local_graph(D, [d["bridge"]] + nb_d)
    colors = {v: ccol(v) for v in Gd.nodes}
    sizes = {v: zsize(v) for v in Gd.nodes}
    sizes[d["bridge"]] = zsize(d["bridge"], 30)
    Pd = draw_local(ax_z[1], Gd, pdl, colors, sizes)
    ax_z[1].scatter(*pdl[d["bridge"]], s=sizes[d["bridge"]] * 1.9, facecolors="none", edgecolors="k", linewidths=0.8,
                    zorder=3)
    finish_zoom(ax_z[1], Pd, "d", f"Bridge between families: {label_of(df, d['bridge'])[0]}")
    fams = pd.Series({v: comm[v] for v in nb_d})
    lab_d = []
    for cfam in fams.value_counts().index:  # the three highest-betweenness neighbors of each family
        lab_d.append(sorted(fams[fams == cfam].index, key=lambda v: -bc[v])[:3])
    lab_d = [v for grp in lab_d for v in grp][:6]
    labs = add_labels(ax_z[1], pdl, [d["bridge"]] + lab_d, df, Pd, bold=(d["bridge"],), unresolved=unresolved)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    taken = [t.get_window_extent(r) for t in labs]
    axbb = ax_z[1].get_window_extent(r)
    for cfam, n in fams.value_counts().items():  # community captions: first free spot around each group
        pts = np.array([pdl[v] for v in fams[fams == cfam].index])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        spots = [(cx, pts[:, 1].min() - 0.10, "center", "top"), (cx, pts[:, 1].max() + 0.10, "center", "bottom"),
                 (pts[:, 0].max() + 0.08, cy, "left", "center"), (pts[:, 0].min() - 0.08, cy, "right", "center"),
                 (pts[:, 0].max(), pts[:, 1].max() + 0.10, "right", "bottom"),
                 (pts[:, 0].max(), pts[:, 1].min() - 0.10, "right", "top")]
        for x, y, ha, va in spots:
            t = ax_z[1].text(x, y, f"community {cfam} ({n})", fontsize=5.6, ha=ha, va=va, zorder=4,
                             color=ccol(int(fams[fams == cfam].index[0])))
            bb = t.get_window_extent(r)
            ok = (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1
                  and not any(bb.overlaps(o) for o in taken) and not crosses_leader(ax_z[1], bb, r) and not covers_node(ax_z[1], bb, Pd))
            if ok:
                taken.append(bb)
                labs.append(t)
                break
            t.remove()
        else:  # fall back to free spots along the panel border (axes coordinates)
            grid = [(x, y, "center", "center") for y in np.arange(0.93, 0.05, -0.04)
                    for x in (0.85, 0.7, 0.55, 0.4, 0.25, 0.15)]
            for x, y, ha, va in grid:
                t = ax_z[1].text(x, y, f"community {cfam} ({n})", fontsize=5.6, ha=ha, va=va, zorder=4,
                                 transform=ax_z[1].transAxes, color=ccol(int(fams[fams == cfam].index[0])))
                bb = t.get_window_extent(r)
                if not any(bb.overlaps(o) for o in taken) and not crosses_leader(ax_z[1], bb, r) \
                        and not covers_node(ax_z[1], bb, Pd):
                    taken.append(bb)
                    labs.append(t)
                    break
                t.remove()
            else:
                print("could not place caption for community", cfam)
    all_texts.append(labs)

    # e: O2-hit region
    e = S["e"]
    nb_e = neighbors_most_connected(D, e["nodes"], N_NEIGH_E)
    Ge = local_graph(D, list(e["nodes"]) + nb_e)
    pe = kk_layout(Ge)
    colors = {v: cmap(norm(gap_all[v])) for v in Ge.nodes}
    sizes = {v: zsize(v) for v in Ge.nodes}
    Pe = draw_local(ax_z[2], Ge, pe, colors, sizes)
    H = np.array([pe[v] for v in e["nodes"]])
    ax_z[2].scatter(H[:, 0], H[:, 1], s=[sizes[v] * 2.4 for v in e["nodes"]], facecolors="none", edgecolors="k",
                    linewidths=0.8, zorder=3)
    finish_zoom(ax_z[2], Pe, "e", "Low-gap region (O2 hits)")
    lab_e = sorted(map(int, e["nodes"]), key=lambda v: gap_all[v])[:7]
    all_texts.append(add_labels(ax_z[2], pe, lab_e, df, Pe, unresolved=unresolved))

    bad = check_overlaps(fig, all_texts)
    print("overlapping labels:", bad or "none")
    print("labels without a CSD refcode (QMOF ID used):", sorted(set(unresolved)) or "none")
    for ext in ("pdf", "svg", "png", "tiff"):
        kw = {"dpi": 600}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"Figure2.{ext}", facecolor="white", **kw)
    print("wrote", OUT / "Figure2.pdf")


if __name__ == "__main__":
    main()
