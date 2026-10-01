"""QMOF loader and the Phase 2 filters (one organic linker, one metal), with per-reason exclusion counts.

Fields used (see reports/DATA_INSPECTION.md): ``qmof_id``, ``name``, ``info.formula``,
``info.mofid.smiles_linkers`` (stringified list), ``info.mofid.smiles_nodes``,
``outputs.pbe.bandgap`` and ``outputs.hse06.bandgap``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import linkers, metals

COLS = {
    "qmof_id": "qmof_id",
    "name": "name",
    "info.formula": "formula",
    "info.mofid.smiles_linkers": "smiles_linkers",
    "info.mofid.smiles_nodes": "smiles_nodes",
    "info.mofid.mofid": "mofid",
    "info.pld": "pld",
    "info.lcd": "lcd",
    "info.density": "density",
    "info.source": "source",
    "outputs.pbe.bandgap": "pbe_gap",
    "outputs.hse06.bandgap": "hse_gap",
}


def load_raw(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv, usecols=list(COLS), low_memory=False).rename(columns=COLS)
    return df


def _parse_list(s) -> list[str] | None:
    if not isinstance(s, str):
        return None
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return None
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else None


def _is_organic(smiles: str) -> bool:
    mol = linkers.mol_from_smiles(smiles)
    return mol is not None and any(a.GetSymbol() == "C" for a in mol.GetAtoms())


@dataclass
class FilterLog:
    steps: list[tuple[str, int, int]] = field(default_factory=list)  # (reason, excluded, remaining)

    def apply(self, df: pd.DataFrame, keep: pd.Series, reason: str) -> pd.DataFrame:
        out = df[keep.values]
        self.steps.append((reason, int((~keep).sum()), len(out)))
        return out

    def table(self, n0: int) -> pd.DataFrame:
        rows = [("start (all QMOF entries)", 0, n0)] + self.steps
        return pd.DataFrame(rows, columns=["filter / exclusion reason", "excluded", "remaining"])


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, FilterLog]:
    """Apply the 2023-paper filters: exactly one distinct organic linker and exactly one metal element."""
    log = FilterLog()
    df = raw.copy()
    df["linker_list"] = df["smiles_linkers"].map(_parse_list)
    df = log.apply(df, df["linker_list"].notna(), "no MOFid linker information")
    df = log.apply(df, df["linker_list"].map(len) > 0, "empty linker list")

    canon = df["linker_list"].map(lambda xs: sorted({linkers.canonical(x) or x for x in xs}))
    df = df.assign(linker_set=canon)
    df = log.apply(df, df["linker_set"].map(len) == 1, "more than one distinct linker")
    df = df.assign(linker_smiles=df["linker_set"].map(lambda xs: xs[0]))
    df = log.apply(df, df["linker_smiles"].map(lambda s: linkers.mol_from_smiles(s) is not None),
                   "linker SMILES not parseable by RDKit")
    df = log.apply(df, df["linker_smiles"].map(_is_organic), "linker is not organic (no carbon)")

    df = df.assign(metal_list=df["formula"].map(metals.metals_in_formula))
    df = log.apply(df, df["metal_list"].map(len) > 0, "no metal in composition")
    df = log.apply(df, df["metal_list"].map(len) == 1, "more than one metal element")
    df = df.assign(metal=df["metal_list"].map(lambda xs: xs[0]))

    def has_props(sym: str) -> bool:
        try:
            metals.properties_mendeleev(sym)
            return True
        except Exception:
            return False

    ok = {s: has_props(s) for s in df["metal"].unique()}
    df = log.apply(df, df["metal"].map(ok), "metal lacks mendeleev descriptors")
    df = log.apply(df, df["pbe_gap"].notna(), "missing PBE band gap")

    df = df.assign(refcode=df["name"].str.split("_").str[0])
    key = df["linker_smiles"] + "|" + df["metal"]
    df = df.assign(bb_group=pd.factorize(key)[0])
    df = df.assign(bb_group_size=df.groupby("bb_group")["qmof_id"].transform("size"))
    keep = ["qmof_id", "name", "refcode", "formula", "linker_smiles", "metal", "pbe_gap", "hse_gap",
            "bb_group", "bb_group_size", "pld", "lcd", "density", "source", "mofid"]
    return df[keep].reset_index(drop=True), log


def subsample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Deterministic subsample (used by --smoke); keeps HSE-labelled rows proportionally."""
    if n >= len(df):
        return df
    idx = np.sort(np.random.default_rng(seed).choice(len(df), size=n, replace=False))
    out = df.iloc[idx].reset_index(drop=True)
    out["bb_group"] = pd.factorize(out["bb_group"])[0]
    out["bb_group_size"] = out.groupby("bb_group")["qmof_id"].transform("size")
    return out
