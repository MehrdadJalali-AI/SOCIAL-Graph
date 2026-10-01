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

Phases 7–8 wait on a decision about Gate 6. To run them anyway, add `--force`.
The ASOC SOCIAL paper is not redistributed in `docs/`; place `SOCIAL_ASOC_2026.pdf` there yourself.

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
