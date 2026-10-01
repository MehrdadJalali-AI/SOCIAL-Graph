"""Phase 1: reproduce the MOFGalaxyNet 2k graph (GATE 1).

Primary recipe (from the specification): Morgan r=2/2048 Tanimoto for linkers, cosine on min-max
normalised mendeleev metal vectors, SIM = 0.7 * linker + 0.3 * metal. Gate 1 compares the edge count
and mean degree at phi = 0.9 with the published 19,266 / 19.256 (+-5%).

Because the gate may fail, the stage always runs a diagnostic sweep (fingerprint type, linker weight,
metal descriptor source) and compares every recipe against the authors' released adjacency matrix.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as C
from ..data import metals, mofgalaxynet_2k
from ..graph import build
from . import GateResult


def _recipe_rows(df: pd.DataFrame, recipe: build.SimilarityRecipe, phis: list[float], released: np.ndarray | None):
    desc = build.descriptors(df["linker_smiles"].tolist(), df["metal"].tolist(), recipe,
                             mofgalaxynet_2k.original_metal_vectors(df))
    sim = build.similarity_matrix(desc, recipe)
    rows = []
    for phi in phis:
        st = build.edge_stats(build.edges_from_dense(sim, phi))
        rows.append({"recipe": recipe.label(), **recipe.__dict__, "phi": phi, **st})
    fit = {}
    if released is not None:
        iu = np.triu_indices(len(sim), 1)
        a, s = released[iu], sim[iu]
        mask = a > 0
        fit = {
            "corr_vs_released": float(np.corrcoef(s[mask], a[mask])[0, 1]),
            "mae_vs_released": float(np.abs(s[mask] - a[mask]).mean()),
        }
    return rows, fit, sim, desc


def _phi_matching(sim: np.ndarray, target_edges: int) -> float:
    """Threshold phi at which the graph has (approximately) ``target_edges`` edges."""
    v = np.sort(np.triu(sim, 1)[np.triu_indices(len(sim), 1)])[::-1]
    return float(v[min(target_edges, len(v)) - 1])


def _metal_check(df: pd.DataFrame) -> pd.DataFrame:
    """Per-property agreement between the original (Mordred-era) vectors and mendeleev/mordred ones."""
    orig = pd.DataFrame(mofgalaxynet_2k.original_metal_vectors(df), columns=metals.PROPERTIES)
    orig["metal"] = df["metal"].values
    orig = orig.drop_duplicates("metal").set_index("metal").sort_index()
    out = []
    for source in ("mendeleev", "mordred"):
        raw, _ = metals.raw_table(orig.index, source)
        norm = metals.minmax(raw).loc[orig.index]
        for p in metals.PROPERTIES:
            out.append({
                "source": source, "property": p,
                "pearson_vs_original": float(np.corrcoef(norm[p], orig[p])[0, 1]),
                "max_abs_diff": float((norm[p] - orig[p]).abs().max()),
            })
    return pd.DataFrame(out)


def run(cfg: dict, force: bool = False) -> GateResult:
    t0 = time.time()
    p1 = cfg["phase1"]
    tables, reports = C.path(cfg, "tables"), C.path(cfg, "reports")
    df = mofgalaxynet_2k.load(C.resolve(p1["table"]), p1["n_mofs"])
    phis = sorted(set(p1["phis"]) | {p1["gate_phi"], 0.8})
    released_path = C.resolve(p1["released_matrix_zip"])
    released = mofgalaxynet_2k.load_released_matrix(released_path, len(df)) if released_path.exists() else None

    primary = build.SimilarityRecipe.from_cfg(p1)
    prim_rows, prim_fit, prim_sim, prim_desc = _recipe_rows(df, primary, phis, released)
    prim = pd.DataFrame(prim_rows)
    prim[["phi", "nodes", "edges", "mean_degree", "isolated", "mean_degree_non_isolated", "max_degree"]].to_csv(
        tables / "phase1_graph_stats.csv", index=False)

    # Diagnostic sweep.
    diag_rows, fits = [], []
    for fp, w, ms in itertools.product(("morgan", "rdkit"), (0.7, 0.9), ("mendeleev", "mordred", "original")):
        rec = build.SimilarityRecipe(fp, p1["morgan_radius"], p1["fp_bits"], w, ms)
        rows, fit, sim, _ = _recipe_rows(df, rec, phis, released)
        diag_rows += rows
        fits.append({"recipe": rec.label(), **fit, "phi_for_published_edges": _phi_matching(sim, p1["published_edges"])})
    if released is not None:
        for phi in phis:
            st = build.edge_stats(build.edges_from_dense(released, phi))
            diag_rows.append({"recipe": "released adjacency matrix (authors)", "phi": phi, **st})
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(tables / "phase1_diagnostics.csv", index=False)
    fit_df = pd.DataFrame(fits).sort_values("mae_vs_released" if released is not None else "recipe")
    fit_df.to_csv(tables / "phase1_recipe_fit.csv", index=False)
    mcheck = _metal_check(df)
    mcheck.to_csv(tables / "phase1_metal_check.csv", index=False)

    # Gate 1.
    g = prim[prim.phi == p1["gate_phi"]].iloc[0]
    rel_e = g.edges / p1["published_edges"] - 1
    rel_d = g.mean_degree / p1["published_mean_degree"] - 1
    ok = abs(rel_e) <= p1["tolerance"] and abs(rel_d) <= p1["tolerance"]
    if cfg["smoke"]:
        gate, passed = "SKIPPED (smoke)", True
    else:
        gate, passed = ("PASS" if ok else "FAIL"), ok
    summary = (f"phi={p1['gate_phi']}: {int(g.edges)} edges ({rel_e:+.1%}), mean degree {g.mean_degree:.3f} "
               f"({rel_d:+.1%}) vs published {p1['published_edges']} / {p1['published_mean_degree']}")

    _write_report(reports / "PHASE1_REPRODUCTION.md", cfg, df, primary, prim, prim_desc, prim_fit, diag, fit_df,
                  mcheck, gate, summary, time.time() - t0)
    return GateResult(1, passed, gate, summary)


def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def _write_report(out: Path, cfg, df, primary, prim, desc, prim_fit, diag, fit_df, mcheck, gate, summary, secs):
    p1 = cfg["phase1"]
    at_gate = diag[diag.phi == p1["gate_phi"]]
    lo, hi = at_gate.edges.min(), at_gate.edges.max()
    rel = at_gate[at_gate.recipe.str.startswith("released")]
    interp = [
        f"- Across all recipes, φ = {p1['gate_phi']} yields {lo:.0f}–{hi:.0f} edges "
        f"(published {p1['published_edges']}).",
        f"- The authors' released matrix yields {int(rel.edges.iloc[0]) if len(rel) else 'n/a'} edges at "
        f"φ = {p1['gate_phi']}.",
        f"- Best-fitting recipe for the released matrix: {fit_df.iloc[0]['recipe']}; the published edge count is "
        f"reached at φ ≈ {fit_df.phi_for_published_edges.min():.2f}–{fit_df.phi_for_published_edges.max():.2f}.",
    ]
    lines = [
        "# Phase 1 — Reproduction of the MOFGalaxyNet 2k graph",
        "",
        f"**Gate 1: {gate}** — {summary}",
        "",
        "## Input",
        f"- Table: `{p1['table']}` — first {len(df)} rows used (the published graph and the released matrix use 2000).",
        f"- Distinct linker SMILES: {df.linker_smiles.nunique()}; unparseable by RDKit: {desc.notes['n_invalid_smiles']} "
        "(linker similarity set to 0, as in the original code).",
        f"- Metals recovered from the normalised atomic number (Z = 3 + 89·AN_norm): {df.metal.nunique()} elements.",
        f"- Metal properties filled with 0 (missing electron affinity in mendeleev): "
        f"{', '.join(sorted(desc.notes['filled_metal_props'])) or 'none'}.",
        "",
        f"## Primary recipe ({primary.label()})",
        "",
        _md(prim[["phi", "edges", "mean_degree", "isolated", "mean_degree_non_isolated", "max_degree"]]),
        "",
        f"Published reference at φ = 0.9: {p1['published_edges']} edges, mean degree {p1['published_mean_degree']}.",
        "",
        "## Diagnostic sweep: edges at each φ",
        "",
        _md(diag.pivot_table(index="recipe", columns="phi", values="edges", aggfunc="first").reset_index(), ".0f"),
        "",
        "## Which recipe produced the authors' released matrix?",
        "",
        "Correlation and mean absolute error against the released weighted adjacency matrix (pairs where the "
        "released value is non-zero), and the φ at which each recipe yields the published edge count.",
        "",
        _md(fit_df, ".4f"),
        "",
        "## Metal descriptors vs the original (Mordred-era) normalised vectors",
        "",
        _md(mcheck, ".3f"),
        "",
        "## Interpretation",
        "",
        *interp,
        "",
        "See `reports/DEVIATIONS.md` for the decision record. Runtime: " f"{secs:.0f} s.",
    ]
    out.write_text("\n".join(lines) + "\n")
