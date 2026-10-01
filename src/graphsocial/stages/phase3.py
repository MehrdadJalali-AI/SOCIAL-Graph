"""Phase 3: QMOF-MOFGalaxyNet graphs at several phi, comparison topologies, communities and centralities."""

from __future__ import annotations

import time

import igraph as ig
import numpy as np
import pandas as pd

from .. import config as C
from .. import plots
from ..data import mofgalaxynet_2k
from ..graph import build, communities, topologies
from ..graph.topologies import Topology
from ..store import Store, graph_name
from . import GateResult


def graph_stats(top: Topology, comm: np.ndarray | None = None) -> dict:
    g = top.to_igraph()
    comps = g.connected_components()
    giant = g.induced_subgraph(max(comps, key=len)) if top.n else g
    deg = top.degree()
    out = {
        "graph": top.name,
        "nodes": top.n,
        "edges": top.m,
        "mean_degree": float(deg.mean()),
        "max_degree": int(deg.max()),
        "isolated": int((deg == 0).sum()),
        "components": len(comps),
        "giant_size": giant.vcount(),
        "giant_frac": giant.vcount() / top.n,
        "clustering_avg_local": float(g.transitivity_avglocal_undirected(mode="zero")),
        "transitivity_global": float(g.transitivity_undirected()),
        "avg_shortest_path_giant": float(giant.average_path_length(directed=False)) if giant.vcount() > 1 else 0.0,
    }
    if comm is not None:
        out["communities"] = int(comm.max() + 1)
        out["modularity"] = communities.modularity(top, comm)
    return out


def _gn_comparison(cfg: dict) -> pd.DataFrame:
    """Leiden vs Girvan-Newman on the original 2k MOFGalaxyNet (spec recipe), per phi."""
    p1, p3 = cfg["phase1"], cfg["phase3"]
    df = mofgalaxynet_2k.load(C.resolve(p1["table"]), p1["n_mofs"])
    rec = build.SimilarityRecipe.from_cfg(p1)
    desc = build.descriptors(df["linker_smiles"].tolist(), df["metal"].tolist(), rec)
    rows = []
    for phi in p3["phis"]:
        e = build.threshold_edges(desc, rec, phi)
        top = Topology.from_pairs(f"2k_phi{phi}", len(df), np.c_[e.row, e.col])
        t0 = time.time()
        gn, skipped = communities.girvan_newman(top, p3["gn_max_component_edges"])
        lei = communities.leiden(top, p3["leiden_resolution"], p3["seed"])
        rows.append({"phi": phi, "edges": top.m, "leiden_communities": int(lei.max() + 1),
                     "gn_communities": int(gn.max() + 1), "ARI_leiden_vs_gn": communities.ari(lei, gn),
                     "gn_components_kept_whole": skipped, "gn_seconds": round(time.time() - t0, 1)})
    return pd.DataFrame(rows)


def _write_graphml(top: Topology, df: pd.DataFrame, comm: np.ndarray, bc: np.ndarray, path) -> None:
    g = top.to_igraph()
    g.vs["qmof_id"] = df["qmof_id"].tolist()
    g.vs["metal"] = df["metal"].tolist()
    g.vs["pbe_gap"] = df["pbe_gap"].astype(float).tolist()
    g.vs["community"] = comm.astype(int).tolist()
    g.vs["betweenness"] = bc.astype(float).tolist()
    g.write_graphml(str(path))


def run(cfg: dict, force: bool = False) -> GateResult:
    t0 = time.time()
    p3 = cfg["phase3"]
    st = Store(cfg)
    tables, figs, reports = C.path(cfg, "tables"), C.path(cfg, "figures"), C.path(cfg, "reports")
    df = st.load_clean()
    bits, mvec = st.load_descriptors()
    rec = build.SimilarityRecipe.from_cfg(cfg["similarity"])
    desc = build.Descriptors(bits, bits.any(axis=1), mvec)
    sens = {**cfg["similarity"], **cfg["similarity"]["sensitivity_recipe"]}
    sens_rec = build.SimilarityRecipe.from_cfg(sens)
    sens_desc = build.descriptors(df["linker_smiles"].tolist(), df["metal"].tolist(), sens_rec)

    rows, degs, csizes, bc_methods = [], {}, {}, {}
    for phi in p3["phis"]:
        e = build.threshold_edges(desc, rec, phi, cfg["similarity"]["block_size"])
        base = Topology.from_pairs(graph_name(phi), len(df), np.c_[e.row, e.col])
        base.save(st.graph(base.name))
        comm = communities.leiden(base, p3["leiden_resolution"], p3["seed"])
        np.save(st.communities(phi), comm)
        degs[f"φ={phi}"] = base.degree()
        csizes[f"φ={phi}"] = np.bincount(comm)

        variants = [base]
        for rho in p3["rhos"]:
            t = topologies.add_long_range(base, rho, comm, p3["seed"])
            t.name = graph_name(phi, "rho", rho)
            variants.append(t)
        dp = topologies.degree_preserving(base, p3["seed"], p3["swaps_per_edge"])
        dp.name = graph_name(phi, "degrand")
        ws = topologies.watts_strogatz(len(df), base.degree().mean(), p3["seed"], p3["ws_p"], graph_name(phi, "ws"))
        variants += [dp, ws]
        es = build.threshold_edges(sens_desc, sens_rec, phi, cfg["similarity"]["block_size"])
        sens_top = Topology.from_pairs(graph_name(phi, "rdkit"), len(df), np.c_[es.row, es.col])

        for t in variants:
            t.save(st.graph(t.name))
            st.centralities(t.name, "all")  # precompute and cache
            rows.append({"phi": phi, **graph_stats(t, comm if t is base else None)})
        rows.append({"phi": phi, **graph_stats(sens_top)})
        bc_methods[phi] = "exact" if len(df) <= p3["betweenness_exact_max_n"] else f"sampled Brandes k={p3['betweenness_k']}"
        if p3.get("graphml"):
            _write_graphml(base, df, comm, st.centralities(base.name, "all")["betweenness"],
                           st.root / "graphs" / f"{base.name}.graphml")

    stats = pd.DataFrame(rows)
    stats.to_csv(tables / "phase3_graph_stats.csv", index=False)
    plots.degree_distributions(degs, figs / "phase3_degree_distribution")
    plots.community_sizes(csizes, figs / "phase3_community_sizes")
    gn = _gn_comparison(cfg)
    gn.to_csv(tables / "phase3_leiden_vs_gn_2k.csv", index=False)

    lines = [
        "# Phase 3 — QMOF-MOFGalaxyNet and comparison topologies",
        "",
        f"N = {len(df)} MOFs. Similarity recipe: {rec.label()}. Betweenness: "
        + ", ".join(f"φ={k}: {v}" for k, v in bc_methods.items()) + " (all centralities max-normalised to [0,1]).",
        "",
        "Topologies per φ: (a) `mgn_phi*`; (b) `*_rho*`, i.e. (a) plus ρ·|E| random edges between different Leiden "
        "communities; (c) `*_degrand`, degree-preserving rewiring with 10·|E| swaps; (d) `ws_phi*`, "
        "Watts–Strogatz with matched mean degree and p = 0.3. `rdkit_phi*` is the code-faithful sensitivity "
        "recipe (RDKit path fingerprint, 0.9/0.1); see DEVIATIONS D1.",
        "",
        stats.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Leiden vs Girvan–Newman on the original 2k MOFGalaxyNet",
        "",
        gn.to_markdown(index=False, floatfmt=".3f"),
        "",
        f"Girvan–Newman is run per connected component and cut at maximum modularity. Components with more than "
        f"{p3['gn_max_component_edges']} edges are kept whole because of GN's O(m²n) cost.",
        "",
        "Figures: `phase3_degree_distribution.png`, `phase3_community_sizes.png`. "
        f"Runtime {time.time() - t0:.0f} s.",
    ]
    (reports / "PHASE3_GRAPHS.md").write_text("\n".join(lines) + "\n")
    base_rows = stats[stats.graph.str.match(r"^mgn_phi[0-9.]+$")]
    summary = "; ".join(f"φ={r.phi}: {r.edges} edges, giant {r.giant_frac:.0%}, {r.communities} communities"
                        for r in base_rows.itertuples())
    return GateResult(3, True, "n/a", summary)
