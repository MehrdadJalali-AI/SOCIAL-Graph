"""Linker fingerprints and Tanimoto similarity.

Fingerprints are stored as packed bit matrices so that all-pairs Tanimoto can be computed in
blocks with integer matrix products, which scales to tens of thousands of MOFs on a CPU.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def mol_from_smiles(smiles: str | None):
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def fingerprint_matrix(
    smiles: Sequence[str | None], kind: str = "morgan", radius: int = 2, n_bits: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(bits, valid)``: an (N, n_bits) uint8 0/1 matrix and a mask of parseable SMILES.

    Unparseable SMILES get an all-zero row; their linker similarity to anything is 0, matching
    the behaviour of the original MOFGalaxyNet ``Similarity.py`` (which caught the exception).
    """
    if kind == "morgan":
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        make = gen.GetFingerprintAsNumPy
    elif kind == "rdkit":
        gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
        make = gen.GetFingerprintAsNumPy
    else:
        raise ValueError(f"unknown fingerprint kind {kind!r}")
    bits = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for i, s in enumerate(smiles):
        mol = mol_from_smiles(s)
        if mol is None:
            continue
        bits[i] = make(mol).astype(np.uint8)
        valid[i] = True
    return bits, valid


def tanimoto_block(bits: np.ndarray, rows: slice, counts: np.ndarray | None = None) -> np.ndarray:
    """Tanimoto similarity of ``bits[rows]`` against all rows; 0/0 is defined as 0."""
    a = bits[rows].astype(np.float32)
    b = bits.astype(np.float32)
    inter = a @ b.T
    if counts is None:
        counts = bits.sum(axis=1).astype(np.float32)
    union = counts[rows][:, None] + counts[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def tanimoto_matrix(bits: np.ndarray) -> np.ndarray:
    return tanimoto_block(bits, slice(0, len(bits)))


def iter_blocks(n: int, block: int) -> Iterator[slice]:
    for start in range(0, n, block):
        yield slice(start, min(n, start + block))


def canonical(smiles: str | None) -> str | None:
    mol = mol_from_smiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol)
