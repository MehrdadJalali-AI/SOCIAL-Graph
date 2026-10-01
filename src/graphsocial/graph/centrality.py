"""Node centralities, max-normalised to [0, 1] so they are commensurate with SOCIAL's influence term.

Betweenness is exact for N <= 5,000 and otherwise estimated with sampled Brandes (k source vertices
drawn with a fixed seed, rescaled by N/k).
"""

from __future__ import annotations

import numpy as np

from .topologies import Topology


def _maxnorm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mx = x.max() if len(x) else 0.0
    return x / mx if mx > 0 else np.zeros_like(x)


def betweenness(top: Topology, exact_max_n: int = 5000, k: int = 1000, seed: int = 0) -> tuple[np.ndarray, str]:
    g = top.to_igraph()
    if top.n <= exact_max_n:
        return _maxnorm(g.betweenness(directed=False)), "exact"
    sources = np.sort(np.random.default_rng(seed).choice(top.n, size=min(k, top.n), replace=False)).tolist()
    bc = np.asarray(g.betweenness(directed=False, sources=sources), dtype=np.float64) * top.n / len(sources)
    return _maxnorm(bc), f"sampled Brandes k={len(sources)}"


def degree(top: Topology) -> np.ndarray:
    return _maxnorm(top.degree())


def pagerank(top: Topology) -> np.ndarray:
    return _maxnorm(top.to_igraph().pagerank(directed=False))


def all_centralities(top: Topology, exact_max_n: int, k: int, seed: int) -> dict[str, np.ndarray]:
    bc, how = betweenness(top, exact_max_n, k, seed)
    return {"betweenness": bc, "degree": degree(top), "pagerank": pagerank(top), "_betweenness_method": how}
