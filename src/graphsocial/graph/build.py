"""MOFGalaxyNet graph construction: SIM = w * Tanimoto(linker) + (1 - w) * cosine(metal), keep SIM >= phi."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from ..data import linkers, metals


@dataclass
class SimilarityRecipe:
    fingerprint: str = "morgan"
    morgan_radius: int = 2
    fp_bits: int = 2048
    linker_weight: float = 0.7
    metal_source: str = "mendeleev"  # mendeleev | mordred | original

    @classmethod
    def from_cfg(cls, cfg: dict) -> "SimilarityRecipe":
        return cls(**{k: cfg[k] for k in cls.__dataclass_fields__ if k in cfg})

    def label(self) -> str:
        fp = f"morgan-r{self.morgan_radius}" if self.fingerprint == "morgan" else "rdkit"
        return f"{fp}-{self.fp_bits} | w_linker={self.linker_weight} | metal={self.metal_source}"


@dataclass
class Descriptors:
    """Per-MOF inputs to the similarity: fingerprint bits and metal vectors."""

    bits: np.ndarray
    valid_smiles: np.ndarray
    metal_vec: np.ndarray
    notes: dict = field(default_factory=dict)


def descriptors(
    smiles: Sequence[str | None],
    metal_symbols: Sequence[str],
    recipe: SimilarityRecipe,
    original_metal_vec: np.ndarray | None = None,
) -> Descriptors:
    bits, valid = linkers.fingerprint_matrix(smiles, recipe.fingerprint, recipe.morgan_radius, recipe.fp_bits)
    if recipe.metal_source == "original":
        if original_metal_vec is None:
            raise ValueError("metal_source='original' needs the original normalised vectors")
        mvec, notes = np.asarray(original_metal_vec, dtype=np.float64), {}
    else:
        mvec, notes = metals.normalized_vectors(list(metal_symbols), recipe.metal_source)
    return Descriptors(bits, valid, mvec, {"filled_metal_props": notes, "n_invalid_smiles": int((~valid).sum())})


def _metal_unit(mvec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mvec, axis=1, keepdims=True)
    return np.divide(mvec, norm, out=np.zeros_like(mvec), where=norm > 0)


def similarity_matrix(desc: Descriptors, recipe: SimilarityRecipe) -> np.ndarray:
    """Dense combined similarity (diagonal set to 0). Use only for small N (Phase 1, tests)."""
    w = recipe.linker_weight
    mu = _metal_unit(desc.metal_vec)
    sim = w * linkers.tanimoto_matrix(desc.bits).astype(np.float64) + (1 - w) * (mu @ mu.T)
    np.fill_diagonal(sim, 0.0)
    return sim


def threshold_edges(desc: Descriptors, recipe: SimilarityRecipe, phi: float, block: int = 2048) -> sp.coo_matrix:
    """Upper-triangular sparse matrix of edges with SIM >= phi, computed block-wise (memory O(block * N))."""
    w = recipe.linker_weight
    n = len(desc.bits)
    mu = _metal_unit(desc.metal_vec).astype(np.float32)
    counts = desc.bits.sum(axis=1).astype(np.float32)
    rows, cols, vals = [], [], []
    for sl in linkers.iter_blocks(n, block):
        s = w * linkers.tanimoto_block(desc.bits, sl, counts) + (1 - w) * (mu[sl] @ mu.T)
        r, c = np.nonzero(s >= phi - 1e-7)
        r = r + sl.start
        keep = c > r
        rows.append(r[keep])
        cols.append(c[keep])
        vals.append(s[r[keep] - sl.start, c[keep]])
    r, c, v = (np.concatenate(x) for x in (rows, cols, vals))
    return sp.coo_matrix((v.astype(np.float32), (r, c)), shape=(n, n))


def edges_from_dense(sim: np.ndarray, phi: float) -> sp.coo_matrix:
    upper = np.triu(sim, 1)
    r, c = np.nonzero(upper >= phi - 1e-7)
    return sp.coo_matrix((upper[r, c], (r, c)), shape=sim.shape)


def edge_stats(edges: sp.coo_matrix) -> dict[str, float]:
    n = edges.shape[0]
    m = int(edges.nnz)
    deg = np.bincount(np.concatenate([edges.row, edges.col]), minlength=n)
    return {
        "nodes": n,
        "edges": m,
        "mean_degree": 2 * m / n if n else 0.0,
        "isolated": int((deg == 0).sum()),
        "non_isolated": int((deg > 0).sum()),
        "mean_degree_non_isolated": float(2 * m / max(1, (deg > 0).sum())),
        "max_degree": int(deg.max()) if n else 0,
    }
