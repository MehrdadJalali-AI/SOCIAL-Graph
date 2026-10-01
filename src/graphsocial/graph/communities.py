"""Community detection: Leiden (primary) and Girvan-Newman (2k comparison only)."""

from __future__ import annotations

import igraph as ig
import leidenalg
import numpy as np
from sklearn.metrics import adjusted_rand_score

from .topologies import Topology


def leiden(top: Topology, resolution: float = 1.0, seed: int = 0) -> np.ndarray:
    part = leidenalg.find_partition(top.to_igraph(), leidenalg.RBConfigurationVertexPartition,
                                    resolution_parameter=resolution, seed=seed)
    return relabel_by_size(np.asarray(part.membership, dtype=np.int64))


def relabel_by_size(memb: np.ndarray) -> np.ndarray:
    """Relabel communities 0..K-1 by decreasing size (ties by first occurrence) for stable ids."""
    labels, first, counts = np.unique(memb, return_index=True, return_counts=True)
    order = np.lexsort((first, -counts))
    remap = np.empty(labels.max() + 1, dtype=np.int64)
    remap[labels[order]] = np.arange(len(labels))
    return remap[memb]


def girvan_newman(top: Topology, max_component_edges: int = 3000) -> tuple[np.ndarray, int]:
    """Girvan-Newman per connected component, cut at maximum modularity.

    Components with more than ``max_component_edges`` edges are kept whole (edge-betweenness
    recomputation is O(m^2 n)); the number of such components is returned.
    """
    g = top.to_igraph()
    memb = np.empty(top.n, dtype=np.int64)
    nxt, skipped = 0, 0
    for comp in g.connected_components():
        sub = g.induced_subgraph(comp)
        if sub.ecount() == 0 or sub.ecount() > max_component_edges:
            skipped += sub.ecount() > max_component_edges
            labels = np.zeros(len(comp), dtype=np.int64)
        else:
            labels = np.asarray(sub.community_edge_betweenness(directed=False).as_clustering().membership)
        memb[comp] = labels + nxt
        nxt += labels.max() + 1
    return relabel_by_size(memb), skipped


def ari(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))


def modularity(top: Topology, memb: np.ndarray) -> float:
    return float(top.to_igraph().modularity(memb.tolist()))
