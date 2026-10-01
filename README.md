# Graph-SOCIAL

Centrality-guided search over metal–organic framework (MOF) similarity networks for budget-limited discovery.
SOCIAL is used as a model-free, structure-aware acquisition policy: it decides which MOF to evaluate next under a fixed evaluation budget, and its communication topology is the MOFGalaxyNet similarity graph.

**Status (2026-10-01):** all eight stages are implemented and tested: 24 unit tests, plus a smoke run of all stages in a few minutes.

| Gate | Result |
|---|---|
| 1 — reproduce MOFGalaxyNet 2k | **FAIL**, overridden by decision: the published φ=0.9 edge count is not reproducible, even from the authors' released matrix (`reports/DEVIATIONS.md` D1) |
| 2 — ≥3,000 QMOF MOFs | **PASS**: 8,697 MOFs; 5,359 have HSE06 gaps |
| 4 — band-gap homophily | **PASS**: PBE r = 0.51 vs null 0.00 at φ = 0.7 (z ≈ 46); significant at every φ |
| 5 — method invariants | **PASS**: all 10 methods |
| 6 — H1 pilot | **FAIL**: Graph-SOCIAL beats random search (recall 0.063 vs 0.011, p ≈ 0.001), but MOFGalaxyNet does not significantly beat the degree-preserving random topology (p = 0.20; +ρ=0.10: p = 0.087). See `reports/PHASE6_PILOT.md` |

Phases 7–8 were run with `--force` by user decision (DEVIATIONS D7), with unchanged hyperparameters and two added ablations (`no_neighbor` and a decoupled geometric embedding). Because gate 6 is FAIL, re-running stage 7 or 8 needs `--force`.

**Main findings** (full details in [RESULTS.md](RESULTS.md)):
- H1 is **not supported at pilot scale**. In the 30-seed runs, MOFGalaxyNet's advantage over a degree-preserving random topology is small and objective-dependent (Cliff's δ ≤ 0.22, power ≤ 0.64, not significant after correction).
- H3 (bridge nodes) is not supported: 42.7% of new-family hits had a top-decile-betweenness neighbour, against a base rate of 51.3%.
- Graph-SOCIAL ranks 5th of 10 (Friedman). Ensemble Thompson sampling ranks 1st, followed by PSO, DE and GA. Graph-SOCIAL beats random search only on O2, and that advantage comes from its chemistry-aware embedding (the decoupled embedding removes it).

Both reference papers in `docs/` are open access (CC BY 4.0).

## Manuscript

`manuscript/` holds the Applied Soft Computing (elsarticle) manuscript, with `main.pdf` and `highlights.pdf` included. Every table is generated from `results/tables/`, and the figures come from `results/figures/`:

```bash
python manuscript/make_tables.py
cd manuscript && latexmk -pdf main.tex
```

Placeholders shown in red (CRediT, competing interests, funding, AI-use statement) must be completed by the authors.

## Install (datalab: Python 3.13, CPU only)

Every dependency ships Python 3.13 wheels or conda-forge builds; nothing is compiled.

```bash
git clone <repo-url> graph-social && cd graph-social
python -m pip install -r requirements.txt      # or: conda env create -f environment.yml
python scripts/check_env.py                    # versions, CPUs, RAM, internet access
python scripts/fetch_data.py                   # pinned MOFGalaxyNet commit + QMOF v18 (392 MB), checksum-verified
```

## Run

### From an IPython console

```
!git clone <repo-url> graph-social
%cd graph-social
!pip install -r requirements.txt
!python scripts/check_env.py
!python scripts/fetch_data.py
!python -m pytest
!python -m graphsocial.run --stage all --config configs/smoke.yaml --smoke
!mkdir -p logs
!nohup python -m graphsocial.run --stage all --config configs/default.yaml --n-jobs <cores> > logs/full_run.log 2>&1 &
!tail -n 40 logs/full_run.log
!python -m graphsocial.run --status
```

### From a shell

```bash
python -m graphsocial.run --stage all --config configs/smoke.yaml --smoke
mkdir -p logs
nohup python -m graphsocial.run --stage all --config configs/default.yaml --n-jobs 32 > logs/full_run.log 2>&1 &
tail -f logs/full_run.log
python -m graphsocial.run --status
```

- `--stage N` runs one stage (1–8); `--stage all` runs them in order.
- `--smoke` runs on tiny subsets and writes to `results/smoke/`. Gates are evaluated but not enforced.
- `--status` lists every stage as pending or done, with the last gate result.
- A FAIL at a hard gate (1, 4, 6) writes the report, prints one PASS/FAIL line, and exits non-zero. Later stages refuse to start until you pass `--force`.

### Resuming after a disconnection

Runs started with `nohup … &` survive SSH or IPython disconnection. Each completed stage is recorded in `results/state.json`. In the benchmark stages, every (method, objective, budget, seed, config) run writes its own file to `results/runs/`. Re-running the same command skips finished work. Use `--status` and `tail logs/full_run.log` to see where it stopped.

## Layout

```
configs/      default.yaml, smoke.yaml, data_versions.yaml (pinned data + checksums)
docs/         SOCIAL (ASOC 2026) and MOFGalaxyNet (J Cheminf 2023) papers
scripts/      check_env.py, fetch_data.py
src/graphsocial/
  data/       metals.py (mendeleev/mordred descriptors), linkers.py (fingerprints, Tanimoto), mofgalaxynet_2k.py
  graph/      build.py (block-wise SIM >= phi thresholding)
  graph/      topologies.py, centrality.py, communities.py
  methods/    graph_social.py + 8 baselines (social_ws is a preset)
  oracle.py, embedding.py, objectives.py, metrics.py, stats.py, plots.py, report.py
  experiment.py (resumable runs), plan.py (run plans), store.py (cached artifacts)
  stages/     phase1.py … phase8.py
  run.py      CLI
reports/      PHASE*_*.md, DATA_INSPECTION.md, DEVIATIONS.md
results/      tables/, figures/, runs/
```
