"""Assemble RESULTS.md (tables inline, figures linked, gate outcomes, deviations) and results_bundle.zip."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import zipfile
from pathlib import Path

from . import config as C

FIGURES = [
    ("F1", "F1_graph_overview", "Giant component of MOFGalaxyNet(φ*), coloured by Leiden community, sized by betweenness"),
    ("F2", "F2_homophily", "Band-gap homophily: observed vs degree-preserving null distributions"),
    ("F3", "F3_recall_curves", "Top-1% recall vs evaluations (mean, 95% CI), one panel per objective"),
    ("F4", "F4_coverage_vs_recall", "Family coverage vs final recall (method means)"),
    ("F5", "F5_critical_difference", "Critical-difference diagram (Friedman + Nemenyi)"),
    ("F6", "F6_ablations", "Ablations of Graph-SOCIAL (final recall, 95% CI)"),
    ("F7", "F7_phi_rho_heatmap", "φ × ρ sensitivity (mean final recall)"),
    ("F8", "F8_h3_bridge_nodes", "H3: bridge-node analysis"),
    ("F9", "F9_runtime", "Runtime per run, excluding oracle lookups"),
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
            if (root / name).exists():
                z.write(root / name, name)
        z.writestr("environment_freeze.json", json.dumps(dict(sorted(env.items())), indent=1))
    return zpath
