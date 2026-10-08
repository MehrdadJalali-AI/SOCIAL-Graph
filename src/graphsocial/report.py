"""Assemble RESULTS.md (tables inline, figures linked, gate outcomes, deviations) and results_bundle.zip."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import zipfile
from pathlib import Path

from . import config as C

FIGURES = [
    ("F1", "F1_graph_overview", "Giant component of MOFGalaxyNet(φ*) (3,000-MOF sample), sized by betweenness: (a) Leiden community, (b) PBE band gap"),
    ("F2", "F2_homophily", "Band-gap homophily: observed vs degree-preserving null distributions"),
    ("F3", "F3_recall_curves", "Top-1% recall vs evaluations (mean, 95% CI), one panel per objective"),
    ("F4", "F4_coverage_vs_recall", "Family coverage vs final recall (method means)"),
    ("F5", "F5_critical_difference", "Critical-difference diagram (Friedman + Nemenyi)"),
    ("F6", "F6_ablations", "Ablations of Graph-SOCIAL (final recall, 95% CI)"),
    ("F7", "F7_phi_rho_heatmap", "φ × ρ sensitivity (mean final recall)"),
    ("F8", "F8_h3_bridge_nodes", "H3: bridge-node analysis"),
    ("F9", "F9_runtime", "Runtime per run, excluding oracle lookups"),
    ("F10", "F10_network_full", "MOFGalaxyNet(φ*), components with at least 5 MOFs: components laid out separately and packed by size, coloured by Leiden community"),
]


def _rel(p: Path, base: Path) -> str:
    return str(p.relative_to(base)) if p.is_relative_to(base) else str(p)


def build(cfg: dict, sections: dict[str, str], choices: dict) -> Path:
    root = C.PROJECT_ROOT
    reports, tables, figs = C.path(cfg, "reports"), C.path(cfg, "tables"), C.path(cfg, "figures")
    out_md = (root / "RESULTS.md") if not cfg["smoke"] else (reports / "RESULTS.md")
    state = C.State(C.path(cfg, "state")).data["stages"]
    gates = "\n".join(f"| {s} | {v.get('gate')} | {v.get('summary', '')} |" for s, v in sorted(state.items()))
    dev = (root / "reports" / "DEVIATIONS.md")
    fig_lines = "\n".join(f"- **{k}** — {desc}: [{name}.png]({_rel(figs / (name + '.png'), out_md.parent)})"
                          for k, name, desc in FIGURES if (figs / f"{name}.png").exists())
    text = f"""# Graph-SOCIAL — Results

Generated {dt.datetime.now().isoformat(timespec='seconds')}{' (SMOKE RUN — not scientific results)' if cfg['smoke'] else ''}.
φ* = {choices['phi_star']}; Graph-SOCIAL search topology = `{choices['best_topology']}`.
Every number is a mean ± std over seeds {cfg['seeds_full'][0]}–{cfg['seeds_full'][1]} unless stated otherwise.

## Gate outcomes

| stage | gate | summary |
|---|---|---|
{gates}

Phase reports: {', '.join(f'[{p.name}]({_rel(p, out_md.parent)})' for p in sorted(reports.glob('PHASE*.md')))}.

## Hypothesis H1 (topology)

**H1 — a chemically meaningful MOFGalaxyNet topology beats a degree-preserving random topology — is not supported at pilot scale** (Gate 6 FAIL; `reports/PHASE6_PILOT.md`). By user decision, Phases 7–8 were run anyway with unchanged hyperparameters and defaults. The 30-seed topology comparison is reported below as observed.

{sections['topology_power']}

Columns: one-sided paired Wilcoxon on final top-1% recall (A > B); Cliff's delta (A vs B); d_z = mean/sd of the paired differences; achieved power of a one-sided paired t-test at α = 0.05 for the observed d_z (`power_t`), and the share of 2,000 bootstrap resamples in which the one-sided Wilcoxon test rejects (`power_wilcoxon_boot`); `n_for_80pct` = pairs needed for 80% t-test power at the observed d_z.

## Evaluation accounting and hit sets

- Every method spends exactly the budget B. The P initial-design MOFs (identical for all methods for a given seed) are evaluations 1…P. They **count toward the budget and toward every metric** (recall curves, AUC, final recall, enrichment, coverage, first-hit index). Non-population methods also start from the same P MOFs.
- Hit sets (top 1% of each objective's universe):

{sections['hits']}

### Centre bias (post hoc; DEVIATIONS D13)

Position of the hit sets relative to the embedding centroid; a centroid-only policy that evaluates the MOFs closest
to the centroid; and the distance-to-centroid percentile of each method's evaluations (early and late thirds of the
run, 2% budget). The benchmark-function shift test is in `reports/BENCH23_CENTER_BIAS.md`.

{sections['center']}

### Unique-building-block recall (post hoc)

Hits are counted per distinct building block (identical linker + metal), so several duplicates of one group count once.

{sections['bb_composition']}

{sections['bb_recall']}

Rank agreement between ordinary and unique-building-block recall (Spearman over method means):

{sections['bb_agreement']}

### Random search vs its analytical expectation

Without replacement, the expected recall of random search after B evaluations is B/N, and the number of hits found per run is hypergeometric(N, K, B). The p-value is exact for the sum over seeds.

{sections['random_check']}

## T1 — Dataset and graph statistics

{sections['T1']}

## T2 — Homophily

{sections['T2']}

## T3 — Main results

{sections['T3']}

## T4 — Graph-SOCIAL vs baselines (paired Wilcoxon by seed, Holm-corrected per objective × budget × metric)

{sections['T4']}

### Friedman test and mean ranks

{sections['friedman']}

## T5 — Ablations (O2, O3; 2% budget)

{sections['T5']}

### φ × ρ grid (mean final recall)

{sections['F7']}

## H3 — Bridge nodes

{sections['H3']}

## T6 — Runtime (seconds per run, excluding oracle lookups)

{sections['T6']}

## Figures

{fig_lines}

## Deviations

{dev.read_text() if dev.exists() else 'none recorded'}
"""
    out_md.write_text(text)
    _bundle(cfg, out_md)
    return out_md


def _bundle(cfg: dict, results_md: Path) -> Path:
    root = C.PROJECT_ROOT
    zpath = (root / "results_bundle.zip") if not cfg["smoke"] else (C.path(cfg, "reports") / "results_bundle.zip")
    env = {p.metadata["Name"]: p.version for p in md.distributions()}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(results_md, "RESULTS.md")
        for key in ("tables", "figures", "reports"):
            d = C.path(cfg, key)
            for f in sorted(d.rglob("*")):
                if f.is_file() and f != zpath:
                    z.write(f, f"{key}/{f.relative_to(d)}")
        for f in sorted((root / "configs").glob("*.yaml")):
            z.write(f, f"configs/{f.name}")
        for name in ("reports/DEVIATIONS.md", "reports/DATA_INSPECTION.md"):
            if (root / name).exists() and name not in z.namelist():
                z.write(root / name, name)
        z.writestr("environment_freeze.json", json.dumps(dict(sorted(env.items())), indent=1))
    return zpath
