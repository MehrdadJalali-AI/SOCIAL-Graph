"""Objectives (all minimised), the non-negative shift used by SOCIAL's influence, and hit sets.

O1 max PBE gap:      f = -gap
O2 min PBE gap:      f = gap
O3 target value:     f = |gap - 2.0|   (eV)
O4 max HSE06 gap:    f = -gap_HSE   (universe = MOFs with an HSE06 gap)

f' = f - min_f over the universe (precomputed) is what SOCIAL's influence formula sees.
Hits: the ceil(1%) MOFs with the lowest f (ties broken by index), for every objective.

O3 was previously a window objective, max(0, 1.5 - gap, gap - 2.5). It is zero for every MOF inside the
window, so the search signal could not tell the hits (the MOFs closest to 2.0 eV) from the other in-window
MOFs. The target-value form keeps the same hit set and gives a non-flat signal (DEVIATIONS D9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

O3_TARGET = 2.0  # eV
NAMES = {"O1": "max PBE gap", "O2": "min PBE gap", "O3": "PBE gap closest to 2.0 eV", "O4": "max HSE06 gap"}


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
        f = np.abs(gap - O3_TARGET)
    else:
        raise ValueError(name)
    hits = _top_fraction(f, hit_frac)
    return Objective(name, universe, f, float(f.min()), hits, gap)
