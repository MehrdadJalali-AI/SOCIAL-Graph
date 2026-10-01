"""Phase 2: QMOF data preparation (GATE 2, soft).

Filters QMOF to MOFs with exactly one distinct organic linker and one metal element, logs every exclusion
reason, flags building-block duplicate groups, and caches descriptors (fingerprints, metal vectors) and
the 32-d search embedding.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import config as C
from .. import embedding as E
from ..data import qmof
from ..graph import build
from ..store import Store
from . import GateResult


def run(cfg: dict, force: bool = False) -> GateResult:
    t0 = time.time()
    p2 = cfg["phase2"]
    st = Store(cfg)
    tables, reports = C.path(cfg, "tables"), C.path(cfg, "reports")
    raw = qmof.load_raw(C.resolve(p2["qmof_csv"]))
    df, flog = qmof.clean(raw)
    n_full = len(df)
    if p2.get("smoke_subsample"):
        df = qmof.subsample(df, p2["smoke_subsample"], seed=0)
    df.to_parquet(st.clean, index=False)
    ftab = flog.table(len(raw))
    ftab.to_csv(tables / "phase2_filters.csv", index=False)

    rec = build.SimilarityRecipe.from_cfg(cfg["similarity"])
    desc = build.descriptors(df["linker_smiles"].tolist(), df["metal"].tolist(), rec)
    np.savez_compressed(st.descriptors, bits=np.packbits(desc.bits, axis=1), n_bits=desc.bits.shape[1],
                        metal_vec=desc.metal_vec, valid=desc.valid_smiles)
    em = cfg["embedding"]
    X = E.build_embedding(desc.bits, desc.metal_vec, em["linker_weight"], em["dim"], em["seed"])
    np.save(st.embedding, X)

    groups = df.groupby("bb_group").agg(size=("qmof_id", "size"), gap_std=("pbe_gap", "std"))
    dup = groups[groups["size"] > 1]
    n_hse = int(df["hse_gap"].notna().sum())
    metals_tab = df["metal"].value_counts().rename_axis("metal").reset_index(name="count")
    metals_tab.to_csv(tables / "phase2_metals.csv", index=False)

    n = len(df)
    ok = n_full >= p2["min_mofs"]
    gate = "PASS" if ok else "SOFT-FAIL"
    summary = f"{n_full} MOFs after filtering (threshold {p2['min_mofs']}); HSE06 subset {n_hse}" + (
        f"; smoke subsample {n}" if n != n_full else "")
    lines = [
        "# Phase 2 — QMOF data preparation",
        "",
        f"**Gate 2 (soft): {gate}** — {summary}",
        "",
        "Source: QMOF Figshare v18 (`configs/data_versions.yaml`). Linkers come from the MOFid fields; the metal "
        "is taken from the composition (`info.formula`), excluding H, B, C, N, O, Si, P, S, Ge, As, Se, Te, "
        "the halogens and the noble gases.",
        "",
        "## Filters and exclusion counts",
        "",
        ftab.to_markdown(index=False),
        "",
        "Distinct linkers are compared after RDKit canonicalisation. 'Organic' means the linker contains carbon.",
        "",
        "## Resulting dataset",
        "",
        f"- MOFs: {n_full}" + (f" (smoke run uses a deterministic subsample of {n})" if n != n_full else ""),
        f"- Distinct linkers: {df['linker_smiles'].nunique()}; distinct metals: {df['metal'].nunique()}",
        f"- PBE gap: mean {df.pbe_gap.mean():.3f} eV, std {df.pbe_gap.std():.3f}, "
        f"range {df.pbe_gap.min():.3f}–{df.pbe_gap.max():.3f}",
        f"- HSE06 gap available: {n_hse} MOFs (O4 runs if ≥ {cfg['objectives']['o4_min_mofs']})",
        f"- Metal properties set to 0 (missing EA in mendeleev): "
        f"{', '.join(sorted(desc.notes['filled_metal_props'])) or 'none'}",
        "",
        "## Building-block duplicates (identical linker + metal)",
        "",
        f"- Groups: {groups.shape[0]}; groups with more than one MOF: {len(dup)}, covering "
        f"{int(dup['size'].sum())} MOFs ({dup['size'].sum() / n:.1%}). Largest group: {int(groups['size'].max())}.",
        f"- Mean within-group PBE-gap std (groups > 1): {dup['gap_std'].mean():.3f} eV vs global std "
        f"{df.pbe_gap.std():.3f} eV. All entries are kept; `bb_group` / `bb_group_size` flag them.",
        "",
        "## Top metals",
        "",
        metals_tab.head(15).to_markdown(index=False),
        "",
        f"Embedding: {X.shape[1]}-d PCA of [0.7 × standardised linker bits | 0.3 × standardised metal vector]. "
        f"Runtime {time.time() - t0:.0f} s.",
    ]
    (reports / "PHASE2_DATA.md").write_text("\n".join(lines) + "\n")
    return GateResult(2, ok or cfg["smoke"], gate, summary)
