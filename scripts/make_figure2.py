"""Figure 2: the QMOF similarity network at phi* (3,000-MOF sample) with zoom panels and MOF labels.

    python scripts/make_figure2.py   -> results/figures/Figure2.{pdf,svg,png,tiff}, results/figures/fig2_layout.csv

Data are the pipeline's: the phi* network, Leiden partition, max-normalized betweenness, PBE gaps and the same
3,000-node BFS sample and Fruchterman-Reingold layout as stage 8 (plots._giant_sample, seed 0). The layout is saved
to fig2_layout.csv and reused on later runs.

Panels: (a) Leiden communities, (b) PBE band gap, (c) the largest building-block group (identical linker and metal)
in the sample, (d) the neighborhood of the highest-betweenness MOF, (e) the region with the most O2 hits.
Zoom regions are chosen by these rules in code; the chosen nodes are printed.

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
def box_around(pts: np.ndarray, span: float, margin: float = 0.18, min_half: float = 0.045) -> tuple[float, float, float, float]:
    lo, hi = pts.min(0), pts.max(0)
    c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2 * (1 + margin), min_half * span)
    half[:] = half.max()  # square boxes so zooms keep the aspect ratio
    return c[0] - half[0], c[1] - half[1], c[0] + half[0], c[1] + half[1]


def select(D: dict) -> dict:
    df, nodes, lay, bc, comm, hit = D["df"], D["nodes"], D["lay"], D["bc"], D["comm"], D["hit"]
    span = float(np.ptp(lay, axis=0).max())
    pos = {int(v): lay[k] for k, v in enumerate(nodes)}
    sub = D["sub"]
    # c: largest building-block group (identical linker + metal) inside the sample; ties -> larger betweenness sum
    d = df.iloc[nodes].assign(node=nodes, bc=bc[nodes])
    grp = d.groupby("bb_group").agg(n=("node", "size"), bcs=("bc", "sum")).sort_values(["n", "bcs"], ascending=False)
    g = int(grp.index[0])
    c_nodes = d[d.bb_group == g].node.to_numpy()
    # d: highest-betweenness MOF and its 1-hop neighborhood
    k_b = int(np.argmax(bc[nodes]))
    bridge = int(nodes[k_b])
    d_nb = np.array([int(nodes[k]) for k in sub.neighbors(k_b)])
    # e: O2 hit with the most other O2 hits within 6% of the layout span; ties -> smaller mean distance
    H = np.array([int(v) for v in nodes if hit[v]])
    P = np.array([pos[v] for v in H])
    dd = np.linalg.norm(P[:, None] - P[None], axis=-1)
    R = 0.06 * span
    cnt = (dd <= R).sum(1)
    best = np.lexsort((np.where(dd <= R, dd, 0).sum(1) / cnt, -cnt))[0]
    e_hits = H[dd[best] <= R]
    sel = {"c": {"nodes": c_nodes, "box": box_around(np.array([pos[v] for v in c_nodes]), span, margin=0.9,
                                                       min_half=0.02), "group": g},
           "d": {"bridge": bridge, "nodes": d_nb,
                 # window: the bridge and its neighbors up to the 85th percentile of distance (far ones stay as edges)
                 "box": box_around(np.array([pos[bridge]] + [pos[v] for v in d_nb if np.linalg.norm(pos[v] - pos[bridge])
                                   <= np.quantile([np.linalg.norm(pos[u] - pos[bridge]) for u in d_nb], 0.85)]),
                                   span, margin=0.12)},
           "e": {"nodes": e_hits, "box": box_around(np.array([pos[v] for v in e_hits]), span, margin=0.6)}}
    print("c: building-block group", g, "|", len(c_nodes), "MOFs:", ", ".join(df.qmof_id.to_numpy()[c_nodes][:10]), "...")
    print("d: bridge", df.qmof_id[bridge], label_of(df, bridge)[0], "| 1-hop neighbors:", len(d_nb),
          "| communities:", pd.Series(comm[d_nb]).value_counts().to_dict())
    print("e: O2 hits in region:", len(e_hits), ":", ", ".join(df.qmof_id.to_numpy()[e_hits]), f"(sample has {len(H)})")
    return sel


# --------------------------------------------------------------------------------------------- drawing helpers
def top_communities(comm_s: np.ndarray) -> list[int]:
    return list(pd.Series(comm_s).value_counts().index[:10])


def comm_color(c: int, top10: list[int]) -> str:
    return plots.PALETTE[top10.index(c)] if c in top10 else OTHER


def draw_net(ax, D, colors, sizes, keep=None, edge_lw=0.2, edge_alpha=0.15):
    lay, sub = D["lay"], D["sub"]
    el = np.asarray(sub.get_edgelist())
    if keep is not None:
        el = el[keep[el[:, 0]] & keep[el[:, 1]]]
    ax.add_collection(LineCollection(lay[el], colors=EDGE, linewidths=edge_lw, alpha=edge_alpha, zorder=1, rasterized=True))
    m = np.ones(len(lay), bool) if keep is None else keep
    ax.scatter(lay[m, 0], lay[m, 1], s=sizes[m], c=np.asarray(colors, dtype=object)[m].tolist() if isinstance(colors, list)
               else colors[m], linewidths=0, zorder=2, rasterized=True)


def place_labels(ax, items, fig, box, obstacles=()):
    """Greedy, overlap-free label placement. items: list of (xy, text, bold). Candidate offsets around each point are
    tried in order; a candidate is accepted if its text box overlaps no placed label, no marked point and stays inside
    the panel. Leader lines connect label and point."""
    renderer = fig.canvas.get_renderer()
    placed = [o.get_window_extent(renderer).expanded(1.05, 1.1) for o in obstacles]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    pts_disp = [ax.transData.transform(xy) for xy, *_ in items]
    ax_bb = ax.get_window_extent(renderer)
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
    failed = []
    for (xy, text, bold), pdisp in zip(items, pts_disp):
        done = False
        for r in (0.10, 0.17, 0.25, 0.33, 0.42):
            for dx, dy in dirs:
                tx, ty = xy[0] + dx * r * w, xy[1] + dy * r * h * 0.8
                if not (x0 + 0.02 * w < tx < x1 - 0.02 * w and y0 + 0.03 * h < ty < y1 - 0.03 * h):
                    continue
                ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
                va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
                t = ax.text(tx, ty, text, fontsize=LABEL_PT, ha=ha, va=va, fontweight="bold" if bold else "normal",
                            zorder=6, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
                bb = t.get_window_extent(renderer).expanded(1.04, 1.12)
                inside = (bb.x0 >= ax_bb.x0 and bb.x1 <= ax_bb.x1 and bb.y0 >= ax_bb.y0 and bb.y1 <= ax_bb.y1)
                clash = any(bb.overlaps(p) for p in placed) or any(bb.contains(*q) for q in pts_disp)
                if inside and not clash:
                    placed.append(bb)
                    ax.plot([xy[0], tx], [xy[1], ty], color="#37424a", lw=0.35, zorder=5)
                    done = True
                    break
                t.remove()
            if done:
                break
        if not done:
            failed.append(text)
    return failed


def _inside(xy, box) -> bool:
    return box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]


def zoom_axes(ax, box, letter, title):
    x0, y0, x1, y1 = box
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9aa4ae")
        s.set_linewidth(0.6)
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    ax.text(0.07, 1.03, title, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom", color="#263238")


# --------------------------------------------------------------------------------------------- figure
def main() -> None:
    D = load()
    S = select(D)
    df, nodes, lay, bc, comm, hit = D["df"], D["nodes"], D["lay"], D["bc"], D["comm"], D["hit"]
    idx = {int(v): k for k, v in enumerate(nodes)}
    cs = comm[nodes]
    top10 = top_communities(cs)
    ccols = [comm_color(c, top10) for c in cs]
    gap = df.pbe_gap.to_numpy()[nodes]
    vmax = float(np.quantile(gap, 0.99))
    size_main = 1.5 + 30 * bc[nodes] / bc[nodes].max()

    fig = plt.figure(figsize=(7.2, 5.6), facecolor="white")
    ax_a = fig.add_axes([0.01, 0.43, 0.45, 0.55])
    ax_b = fig.add_axes([0.49, 0.43, 0.42, 0.55])
    cax = fig.add_axes([0.925, 0.50, 0.011, 0.40])
    ax_z = [fig.add_axes([0.035 + k * 0.325, 0.025, 0.285, 0.34]) for k in range(3)]
    for ax in (ax_a, ax_b):
        ax.set_aspect("equal")
        ax.set_axis_off()
    draw_net(ax_a, D, ccols, size_main)
    el = np.asarray(D["sub"].get_edgelist())
    ax_b.add_collection(LineCollection(lay[el], colors=EDGE, linewidths=0.2, alpha=0.15, zorder=1, rasterized=True))
    sc = ax_b.scatter(lay[:, 0], lay[:, 1], s=size_main, c=gap, cmap=GAPMAP, vmin=0, vmax=vmax, linewidths=0, zorder=2,
                      rasterized=True)
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label("PBE band gap (eV)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2)
    cb.outline.set_linewidth(0.5)
    for ax in (ax_a, ax_b):
        ax.autoscale()
    ax_b.set_xlim(ax_a.get_xlim())
    ax_b.set_ylim(ax_a.get_ylim())
    for x, letter, title in ((0.012, "a", "Leiden communities"), (0.49, "b", "PBE band gap")):
        fig.text(x, 0.985, letter, fontsize=8.5, fontweight="bold", va="top")
        fig.text(x + 0.022, 0.983, title, fontsize=6.5, va="top", color="#263238")

    # zoom boxes on panel a, connectors to the zoom panels
    for (key, sel), axz in zip(S.items(), ax_z):
        x0, y0, x1, y1 = sel["box"]
        ax_a.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.02",
                                      fc="none", ec="#37424a", lw=0.6, zorder=4, mutation_aspect=1))
        lx, ly, ha, va = (x1, y0, "left", "top") if key == "d" else (x0, y1, "right", "bottom")
        ax_a.text(lx, ly, key, fontsize=6.5, fontweight="bold", ha=ha, va=va, color="#263238", zorder=5,
                  bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
        for xa, xz in ((x0, 0.0), (x1, 1.0)):
            fig.add_artist(ConnectionPatch(xyA=(xa, y0), coordsA=ax_a.transData, xyB=(xz, 1.0), coordsB=axz.transAxes,
                                           color="#9aa4ae", lw=0.35, alpha=0.6, zorder=0))

    unresolved = []

    def zoom(axz, sel, color_mode, title, letter, label_nodes, bold=(), focus=None, star_from=None, notes=()):
        x0, y0, x1, y1 = sel["box"]
        keep = (lay[:, 0] >= x0) & (lay[:, 0] <= x1) & (lay[:, 1] >= y0) & (lay[:, 1] <= y1)
        foc = np.zeros(len(lay), bool)
        if focus is not None:
            foc[[idx[v] for v in focus]] = True
        zsize = 7 + 40 * bc[nodes] / bc[nodes].max()
        el = np.asarray(D["sub"].get_edgelist())
        el = el[keep[el[:, 0]] & keep[el[:, 1]]]
        if focus is None:
            axz.add_collection(LineCollection(lay[el], colors="#c9d1d9", linewidths=0.15, alpha=0.12, zorder=1))
        if focus is not None:
            fe = el[foc[el[:, 0]] & foc[el[:, 1]]]
            axz.add_collection(LineCollection(lay[fe], colors="#7a8794", linewidths=0.3, alpha=0.35, zorder=1))
        if star_from is not None:  # edges from the bridge to all its neighbors (clipped at the panel border)
            kb = idx[star_from]
            segs = [[lay[kb], lay[idx[v]]] for v in sel["nodes"]]
            axz.add_collection(LineCollection(segs, colors="#37424a", linewidths=0.4, alpha=0.6, zorder=2))
        if color_mode == "community":
            cc = np.asarray(ccols, dtype=object)
            bg = keep & ~foc if focus is not None else np.zeros(len(lay), bool)
            fg = keep & foc if focus is not None else keep
            axz.scatter(lay[bg, 0], lay[bg, 1], s=zsize[bg] * 0.6, c="#e9ecef", linewidths=0, zorder=2)
            axz.scatter(lay[fg, 0], lay[fg, 1], s=zsize[fg], c=cc[fg].tolist(), linewidths=0.4, edgecolors="white",
                        zorder=3)
        else:
            axz.scatter(lay[keep, 0], lay[keep, 1], s=zsize[keep], c=gap[keep], cmap=GAPMAP, vmin=0, vmax=vmax,
                        linewidths=0.3, edgecolors="white", zorder=2)
            hk = np.array([idx[v] for v in sel["nodes"]])
            axz.scatter(lay[hk, 0], lay[hk, 1], s=zsize[hk] * 2.2, facecolors="none", edgecolors="k", linewidths=0.8,
                        zorder=3)
        zoom_axes(axz, sel["box"], letter, title)
        items = []
        for v in label_nodes:
            lab, ok = label_of(df, v)
            if not ok:
                unresolved.append(lab)
            items.append((lay[idx[v]], f"{lab} · {df.pbe_gap[v]:.2f} eV", v in bold))
        obst = [axz.text(*xy, txt, transform=axz.transAxes, fontsize=LABEL_PT, ha=ha, va=va, color=col, zorder=7,
                         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=0.5, alpha=0.95))
                for xy, txt, ha, va, col in notes]
        fig.canvas.draw()
        return place_labels(axz, items, fig, sel["box"], obstacles=obst)

    # c: label up to 8 members of the group, spread over the group (highest betweenness first, distinct labels)
    sc_ = S["c"]
    members = sorted(sc_["nodes"], key=lambda v: -bc[v])
    lab_c, seen = [], set()
    for v in members:
        if label_of(df, v)[0] not in seen and len(lab_c) < 8:
            lab_c.append(v)
            seen.add(label_of(df, v)[0])
    r0 = df.iloc[sc_["nodes"][0]]
    tag = f"{r0.metal} | {linker_formula(r0.linker_smiles)}"
    fc = zoom(ax_z[0], sc_, "community", f"Building-block near-clique: {tag}, {len(sc_['nodes'])} MOFs", "c", lab_c,
              focus=sc_["nodes"])
    # d: bridge in bold + the highest-betweenness neighbor of each community, then the next highest, up to 7
    sd = S["d"]
    nb = sorted(sd["nodes"], key=lambda v: -bc[v])
    lab_d, comms = [], []
    for v in nb:
        if comm[v] not in comms:
            lab_d.append(v)
            comms.append(comm[v])
    for v in nb:
        if len(lab_d) >= 7:
            break
        if v not in lab_d:
            lab_d.append(v)
    lab_d = [v for v in lab_d if _inside(lay[idx[v]], sd["box"])]
    out = [v for v in sd["nodes"] if not _inside(lay[idx[v]], sd["box"])]
    notes = []
    if out:
        cnt = pd.Series(comm[out]).value_counts()
        for c, n in cnt.items():
            vec = np.array([lay[idx[v]] for v in out if comm[v] == c]).mean(0) - lay[idx[sd["bridge"]]]
            right, up = vec[0] > 0, vec[1] > 0
            col = comm_color(c, top10) if comm_color(c, top10) != OTHER else "#37424a"
            notes.append(((0.97 if right else 0.03, 0.96 if up else 0.04),
                          f"{n} neighbor{'s' if n > 1 else ''} in this family\n(outside view)",
                          "right" if right else "left", "top" if up else "bottom", col))
        print("d: neighbors outside the zoom window by community:", cnt.to_dict())
    fd = zoom(ax_z[1], sd, "community", "Bridge between families", "d", [sd["bridge"]] + lab_d, bold=(sd["bridge"],),
              focus=np.r_[sd["bridge"], sd["nodes"]], star_from=sd["bridge"], notes=notes)
    # e: label the O2 hits in the region (up to 10, lowest gap first)
    se = S["e"]
    lab_e = sorted(se["nodes"], key=lambda v: df.pbe_gap[v])[:10]
    fe = zoom(ax_z[2], se, "gap", "Low-gap region (O2 hits)", "e", lab_e)

    print("labels that could not be placed without overlap:", fc + fd + fe or "none")
    print("labels without a CSD refcode (QMOF ID used):", sorted(set(unresolved)) or "none")
    for ext in ("pdf", "svg", "png", "tiff"):
        kw = {"dpi": 600} if ext in ("png", "tiff") else {"dpi": 600}
        if ext == "tiff":
            kw["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(OUT / f"Figure2.{ext}", facecolor="white", **kw)
    print("wrote", OUT / "Figure2.pdf")


if __name__ == "__main__":
    main()
