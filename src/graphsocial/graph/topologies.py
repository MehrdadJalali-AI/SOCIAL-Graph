"""Search topologies over the MOF node set and hop-neighbourhood queries.

(a) MOFGalaxyNet(phi); (b) MOFGalaxyNet(phi) + rho*|E| long-range random edges (endpoints in different
Leiden communities); (c) degree-preserving randomisation (10*|E| double-edge swaps); (d) Watts-Strogatz
with matching mean degree, p = 0.3.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import igraph as ig
import numpy as np
import scipy.sparse as sp


@dataclass
class Topology:
    name: str
    n: int
    edges: np.ndarray  # (m, 2) int32, u < v, unique
    _csr: sp.csr_matrix | None = field(default=None, repr=False)

    @classmethod
    def from_pairs(cls, name: str, n: int, pairs: np.ndarray) -> "Topology":
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = np.sort(pairs, axis=1)
        pairs = np.unique(pairs, axis=0)
        return cls(name, n, pairs.astype(np.int32))

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def csr(self) -> sp.csr_matrix:
        if self._csr is None:
            u, v = self.edges[:, 0], self.edges[:, 1]
            data = np.ones(2 * self.m, dtype=np.int8)
            self._csr = sp.csr_matrix((data, (np.r_[u, v], np.r_[v, u])), shape=(self.n, self.n))
        return self._csr

    def degree(self) -> np.ndarray:
        return np.diff(self.csr.indptr)

    def neighbors(self, i: int) -> np.ndarray:
        c = self.csr
        return c.indices[c.indptr[i]:c.indptr[i + 1]]

    def within_hops(self, src: int, h: int) -> set[int]:
        """All nodes at hop distance 1..h from ``src``."""
        seen = {src}
        frontier = [src]
        for _ in range(h):
            nxt = []
            for u in frontier:
                for v in self.neighbors(u):
                    v = int(v)
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        seen.discard(src)
        return seen

    def to_igraph(self) -> ig.Graph:
        return ig.Graph(n=self.n, edges=self.edges.tolist(), directed=False)

    def subgraph(self, nodes: np.ndarray, name: str | None = None) -> "Topology":
        """Induced subgraph on ``nodes`` (relabelled 0..len(nodes)-1 in the given order)."""
        remap = -np.ones(self.n, dtype=np.int64)
        remap[nodes] = np.arange(len(nodes))
        e = remap[self.edges]
        e = e[(e >= 0).all(axis=1)]
        return Topology.from_pairs(name or self.name, len(nodes), e)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, name=self.name, n=self.n, edges=self.edges)

    @classmethod
    def load(cls, path: Path) -> "Topology":
        z = np.load(path)
        return cls(str(z["name"]), int(z["n"]), z["edges"])


def _seed_igraph(seed: int) -> None:
    ig.set_random_number_generator(random.Random(seed))


def add_long_range(base: Topology, rho: float, communities: np.ndarray, seed: int) -> Topology:
    """Add round(rho*|E|) random edges between nodes in different communities (no duplicates)."""
    rng = np.random.default_rng(seed)
    target = int(round(rho * base.m))
    existing = set(map(tuple, base.edges.tolist()))
    new: set[tuple[int, int]] = set()
    while len(new) < target:
        u = rng.integers(0, base.n, size=4 * (target - len(new)) + 16)
        v = rng.integers(0, base.n, size=len(u))
        for a, b in zip(u.tolist(), v.tolist()):
            if a == b or communities[a] == communities[b]:
                continue
            e = (a, b) if a < b else (b, a)
            if e in existing or e in new:
                continue
            new.add(e)
            if len(new) == target:
                break
    pairs = np.vstack([base.edges, np.array(sorted(new), dtype=np.int32).reshape(-1, 2)])
    return Topology.from_pairs(f"{base.name}_rho{rho:.2f}", base.n, pairs)


def degree_preserving(base: Topology, seed: int, swaps_per_edge: int = 10) -> Topology:
    g = base.to_igraph()
    _seed_igraph(seed)
    g.rewire(n=swaps_per_edge * base.m, allowed_edge_types="simple")
    return Topology.from_pairs(f"{base.name}_degrand", base.n, np.array(g.get_edgelist()))


def watts_strogatz(n: int, mean_degree: float, seed: int, p: float = 0.3, name: str = "ws") -> Topology:
    nei = max(1, int(round(mean_degree / 2)))
    _seed_igraph(seed)
    g = ig.Graph.Watts_Strogatz(dim=1, size=n, nei=nei, p=p)
    return Topology.from_pairs(name, n, np.array(g.get_edgelist()))
