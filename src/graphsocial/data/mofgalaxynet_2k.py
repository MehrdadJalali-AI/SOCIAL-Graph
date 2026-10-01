"""Loader for the original MOFGalaxyNet 2,000-MOF table.

File: ``MOFGalaxyNet/Data/SMILES_METAL_2000_NoPLD.csv`` (no header, 2,004 rows). Inspected columns:
six min-max normalised metal properties (AN, AW, AR, ME, P, EA), a row index, the CSD refcode,
a single linker SMILES, and a PLD class label (0-3). The metal symbol is not stored; it is
recovered from the normalised atomic number (see ``metals.symbol_from_original_an``).
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .metals import PROPERTIES, symbol_from_original_an

COLUMNS = [*PROPERTIES, "idx", "refcode", "linker_smiles", "pld_class"]


def load(table: str | Path, n_mofs: int | None = 2000) -> pd.DataFrame:
    df = pd.read_csv(table, header=None, names=COLUMNS, dtype={"refcode": str, "linker_smiles": str})
    if n_mofs is not None:
        df = df.iloc[:n_mofs].copy()
    df["metal"] = [symbol_from_original_an(x) for x in df["AN"]]
    return df.reset_index(drop=True)


def original_metal_vectors(df: pd.DataFrame) -> np.ndarray:
    return df[list(PROPERTIES)].to_numpy(dtype=np.float64)


def load_released_matrix(zip_path: str | Path, n: int = 2000) -> np.ndarray:
    """The authors' released weighted adjacency matrix (similarities, 0 below ~0.1, zero diagonal)."""
    with zipfile.ZipFile(zip_path) as z:
        name = next(m for m in z.namelist() if m.endswith(".csv") and "__MACOSX" not in m)
        with z.open(name) as fh:
            mat = np.loadtxt(io.TextIOWrapper(fh), delimiter=",")
    return mat[:n, :n]
