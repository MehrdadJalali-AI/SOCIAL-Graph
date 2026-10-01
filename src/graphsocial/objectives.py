"""Objectives (all minimised), the non-negative shift used by SOCIAL's influence, and hit sets.

O1 max PBE gap:      f = -gap
O2 min PBE gap:      f = gap
O3 window [1.5,2.5]: f = max(0, 1.5 - gap, gap - 2.5)
O4 max HSE06 gap:    f = -gap_HSE   (universe = MOFs with an HSE06 gap)

f' = f - min_f over the universe (precomputed) is what SOCIAL's influence formula sees.
Hits: the ceil(1%) MOFs with the lowest f (ties broken by index). For O3 the hit set is every MOF inside
the window if that is fewer than 1% of the universe, otherwise the 1% closest to the window centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW = (1.5, 2.5)
NAMES = {"O1": "max PBE gap", "O2": "min PBE gap", "O3": "PBE gap in [1.5, 2.5] eV", "O4": "max HSE06 gap"}


@dataclass
class Objective:
    name: str
    universe: np.ndarray  # indices into the clean QMOF table
    f: np.ndarray         # raw minimised objective on the universe
    shift: float          # min f over the universe
    hits: np.ndarray      # boolean mask over the universe
    gap: np.ndarray       # underlying band gap (for reporting)

    @property
    def f_shifted(self) -> np.ndarray:
        return self.f - self.shift

    @property
    def n(self) -> int:
        return len(self.f)

    @property
    def f_best(self) -> float:
        return float(self.f.min())


def _top_fraction(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, math.ceil(frac * len(score)))
    order = np.lexsort((np.arange(len(score)), score))
    mask = np.zeros(len(score), dtype=bool)
    mask[order[:k]] = True
    return mask


def make(name: str, df: pd.DataFrame, hit_frac: float = 0.01) -> Objective:
    if name == "O4":
        universe = np.flatnonzero(df["hse_gap"].notna().to_numpy())
        gap = df["hse_gap"].to_numpy()[universe]
    else:
        universe = np.arange(len(df))
        gap = df["pbe_gap"].to_numpy()
    gap = gap.astype(np.float64)
    if name in ("O1", "O4"):
        f = -gap
    elif name == "O2":
        f = gap.copy()
    elif name == "O3":
        f = np.maximum(0.0, np.maximum(WINDOW[0] - gap, gap - WINDOW[1]))
    else:
        raise ValueError(name)
    if name == "O3":
        inside = f == 0
        k = max(1, math.ceil(hit_frac * len(f)))
        hits = inside if inside.sum() < k else _top_fraction(np.abs(gap - sum(WINDOW) / 2), hit_frac)
    else:
        hits = _top_fraction(f, hit_frac)
    return Objective(name, universe, f, float(f.min()), hits, gap)
