"""Continuous embedding of MOFs and the snap-to-nearest-unevaluated operator.

Embedding: standardised linker fingerprint bits (weight 0.7) and standardised metal vector (weight 0.3),
concatenated and reduced to 32 dimensions by PCA. Each block is divided by sqrt(#columns) after
standardisation so that it carries unit total variance; the 0.7/0.3 weights then set the blocks'
relative contribution (otherwise 2,048 bit columns would swamp 6 metal columns). See DEVIATIONS.md.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std(axis=0)
    keep = sd > 0
    z = (x[:, keep] - x[:, keep].mean(axis=0)) / sd[keep]
    return z / np.sqrt(max(1, z.shape[1]))


def build_embedding(bits: np.ndarray, metal_vec: np.ndarray, linker_weight: float = 0.7,
                    dim: int = 32, seed: int = 0) -> np.ndarray:
    x = np.hstack([linker_weight * _standardize(bits), (1 - linker_weight) * _standardize(metal_vec)])
    dim = min(dim, x.shape[1], x.shape[0])
    return PCA(n_components=dim, random_state=seed, svd_solver="full").fit_transform(x).astype(np.float64)


class Snapper:
    """Nearest unevaluated MOF to a query point (Euclidean), via a KD-tree with masking.

    The tree is queried for k neighbours with k doubling until an unevaluated MOF appears; ties are
    broken by the lower MOF index, so the result is deterministic.
    """

    def __init__(self, X: np.ndarray):
        self.X = X
        self.tree = cKDTree(X)
        self.n = len(X)

    def nearest(self, x: np.ndarray, evaluated: np.ndarray) -> int:
        if evaluated.all():
            raise RuntimeError("all MOFs evaluated")
        k = 8
        while True:
            k = min(k, self.n)
            d, idx = self.tree.query(x, k=k)
            d, idx = np.atleast_1d(d), np.atleast_1d(idx)
            free = ~evaluated[idx]
            if free.any():
                dmin = d[free].min()
                cand = idx[free & (d <= dmin + 1e-12)]
                return int(cand.min())
            if k == self.n:
                raise RuntimeError("no unevaluated MOF found")
            k *= 4

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X.min(axis=0), self.X.max(axis=0)
