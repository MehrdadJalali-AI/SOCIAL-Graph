"""Metal descriptors: the six MOFGalaxyNet properties, min-max normalisation and cosine similarity.

The 2023 MOFGalaxyNet paper computed atomic number (AN), atomic weight (AW), atomic radius (AR),
Mulliken electronegativity (ME), polarizability (P) and electron affinity (EA) with Mordred.
Mordred is unmaintained, so the primary source here is ``mendeleev``; ``mordredcommunity``
tables are offered as a diagnostic alternative (Phase 1).
"""

from __future__ import annotations

import functools
import re
from typing import Iterable

import numpy as np
import pandas as pd

PROPERTIES: tuple[str, ...] = ("AN", "AW", "AR", "ME", "P", "EA")

# Elements that never count as the "metal" of a MOF.
NONMETALS: frozenset[str] = frozenset(
    "H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Te I Xe At Rn".split()
)

# Range used by the original 2k table: AN_norm = (Z - 3) / 89, i.e. Li..U.
_ORIG_Z_MIN, _ORIG_Z_SPAN = 3, 89


@functools.lru_cache(maxsize=None)
def _mendeleev_element(symbol: str):
    from mendeleev import element

    return element(symbol)


@functools.lru_cache(maxsize=None)
def properties_mendeleev(symbol: str) -> tuple[dict[str, float], tuple[str, ...]]:
    """Six raw properties from mendeleev, plus the names of properties that were missing.

    Missing electron affinities (elements with an unbound anion, e.g. Zn, Mg) are set to 0 eV;
    missing radii/polarizabilities raise, since no physically neutral fill-in exists.
    """
    e = _mendeleev_element(symbol)
    missing: list[str] = []
    ea = e.electron_affinity
    if ea is None:
        ea = 0.0
        missing.append("EA")
    ie1 = e.ionenergies.get(1)
    if e.atomic_radius is None or e.dipole_polarizability is None or ie1 is None:
        raise ValueError(f"mendeleev lacks radius/polarizability/IE1 for {symbol}")
    props = {
        "AN": float(e.atomic_number),
        "AW": float(e.atomic_weight),
        "AR": float(e.atomic_radius),
        "ME": (float(ie1) + float(ea)) / 2.0,
        "P": float(e.dipole_polarizability),
        "EA": float(ea),
    }
    return props, tuple(missing)


@functools.lru_cache(maxsize=None)
def properties_mordred(symbol: str) -> tuple[dict[str, float], tuple[str, ...]]:
    """Six properties using mordredcommunity atomic tables where they exist.

    Mordred has no electron-affinity table, so EA (and hence the Mulliken term) comes from mendeleev.
    """
    from mordred import _atomic_property as ap

    z = _mendeleev_element(symbol).atomic_number
    base, missing = properties_mendeleev(symbol)
    ie1 = float(ap.ionization_potentials[z])
    return (
        {
            "AN": float(z),
            "AW": float(ap.mass[z]),
            "AR": float(ap.vdw_radii[z]),
            "ME": (ie1 + base["EA"]) / 2.0,
            "P": float(ap.polarizability94[z]),
            "EA": base["EA"],
        },
        missing,
    )


def raw_table(symbols: Iterable[str], source: str = "mendeleev") -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Raw property table indexed by element symbol and a log of filled-in values."""
    fn = {"mendeleev": properties_mendeleev, "mordred": properties_mordred}[source]
    rows, notes = {}, {}
    for s in sorted(set(symbols)):
        rows[s], miss = fn(s)
        if miss:
            notes[s] = list(miss)
    return pd.DataFrame.from_dict(rows, orient="index")[list(PROPERTIES)], notes


def minmax(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise min-max normalisation; constant columns map to 0."""
    lo, hi = df.min(), df.max()
    span = (hi - lo).replace(0, 1.0)
    return (df - lo) / span


def cosine_matrix(x: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors (zero vectors get similarity 0)."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    xn = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    return xn @ xn.T


def normalized_vectors(symbols: list[str], source: str = "mendeleev") -> tuple[np.ndarray, dict[str, list[str]]]:
    """Per-MOF min-max normalised six-vectors (normalised over the distinct metals present)."""
    raw, notes = raw_table(symbols, source)
    norm = minmax(raw)
    return norm.loc[symbols].to_numpy(dtype=np.float64), notes


def symbol_from_original_an(an_norm: float) -> str:
    """Recover the element from the original 2k table's min-max normalised atomic number."""
    z = _ORIG_Z_MIN + int(round(an_norm * _ORIG_Z_SPAN))
    return _mendeleev_element(z).symbol


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def elements_in_formula(formula: str) -> list[str]:
    """Element symbols appearing in a chemical formula string (order of first appearance)."""
    seen: list[str] = []
    for sym, _ in _FORMULA_TOKEN.findall(formula):
        if sym not in seen:
            seen.append(sym)
    return seen


def metals_in_formula(formula: str) -> list[str]:
    return [s for s in elements_in_formula(formula) if s not in NONMETALS]
