"""Phase 1 unit tests: descriptors, similarity and thresholding."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs

from graphsocial import config as C
from graphsocial.data import linkers, metals, mofgalaxynet_2k
from graphsocial.graph import build

SMILES = ["OC(=O)c1ccncc1", "OC(=O)c1ccc(cc1)C(=O)O", "OC(=O)c1cc(cc(c1)C(=O)O)C(=O)O", "not-a-smiles", "c1ccccc1"]


def test_tanimoto_matches_rdkit():
    bits, valid = linkers.fingerprint_matrix(SMILES, "morgan", 2, 2048)
    assert valid.tolist() == [True, True, True, False, True]
    sim = linkers.tanimoto_matrix(bits)
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in (SMILES[0], SMILES[1])]
    assert sim[0, 1] == pytest.approx(DataStructs.TanimotoSimilarity(*fps), abs=1e-6)
    assert sim[3].max() == 0.0  # unparseable SMILES -> 0
    assert np.allclose(np.diag(sim)[valid], 1.0)


def test_rdkit_generator_matches_rdkfingerprint():
    bits, _ = linkers.fingerprint_matrix(SMILES[:2], "rdkit", n_bits=2048)
    legacy = [Chem.RDKFingerprint(Chem.MolFromSmiles(s)) for s in SMILES[:2]]
    ref = DataStructs.TanimotoSimilarity(*legacy)
    assert linkers.tanimoto_matrix(bits)[0, 1] == pytest.approx(ref, abs=1e-6)


def test_metal_vectors_and_cosine():
    vec, notes = metals.normalized_vectors(["Zn", "Cu", "Zn", "Zr"], "mendeleev")
    assert vec.shape == (4, 6)
    assert vec.min() >= 0 and vec.max() <= 1
    assert "Zn" in notes and notes["Zn"] == ["EA"]
    cos = metals.cosine_matrix(vec)
    assert cos[0, 2] == pytest.approx(1.0)
    assert np.allclose(cos, cos.T)


def test_mulliken_definition():
    p, _ = metals.properties_mendeleev("Co")
    assert p["ME"] == pytest.approx((7.88101 + p["EA"]) / 2, abs=1e-4)


def test_formula_metals():
    assert metals.metals_in_formula("Ba2CuC6H14O16") == ["Ba", "Cu"]
    assert metals.metals_in_formula("ZnC8H4O4") == ["Zn"]


def test_blockwise_threshold_equals_dense():
    rng = np.random.default_rng(0)
    smiles = ["OC(=O)c1ccncc1", "OC(=O)c1ccc(cc1)C(=O)O", "OC(=O)c1cc(cc(c1)C(=O)O)C(=O)O", "c1ccccc1"] * 6
    syms = list(rng.choice(["Zn", "Cu", "Co", "Zr"], size=len(smiles)))
    rec = build.SimilarityRecipe()
    desc = build.descriptors(smiles, syms, rec)
    dense = build.edges_from_dense(build.similarity_matrix(desc, rec), 0.7)
    blocked = build.threshold_edges(desc, rec, 0.7, block=5)
    assert dense.nnz == blocked.nnz
    assert set(zip(dense.row, dense.col)) == set(zip(blocked.row, blocked.col))


@pytest.mark.skipif(not C.resolve("data/raw/MOFGalaxyNet/Data/SMILES_METAL_2000_NoPLD.csv").exists(),
                    reason="MOFGalaxyNet repository not downloaded")
def test_metal_recovery_from_original_table():
    df = mofgalaxynet_2k.load(C.resolve("data/raw/MOFGalaxyNet/Data/SMILES_METAL_2000_NoPLD.csv"), 2000)
    assert len(df) == 2000 and df.metal.nunique() == 48
    # Atomic weights recomputed from the recovered elements reproduce the stored normalised AW column.
    raw, _ = metals.raw_table(df.metal.unique(), "mendeleev")
    aw = (raw.AW - raw.AW.min()) / (raw.AW.max() - raw.AW.min())
    stored = df.drop_duplicates("metal").set_index("metal").AW
    assert np.allclose(aw.loc[stored.index], stored, atol=2e-3)
