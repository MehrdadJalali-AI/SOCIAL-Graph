# Graph-SOCIAL

Centrality-guided search over metal–organic framework (MOF) similarity networks for budget-limited discovery.
SOCIAL is used as a model-free, structure-aware acquisition policy: it decides which MOF to evaluate next under a fixed evaluation budget, and its communication topology is the MOFGalaxyNet similarity graph.

**Status (2026-10-03, tag `v0.3-revision`):** the JCIM revision is complete. `manuscript/main.pdf` and `manuscript/SI.pdf` are built from the results files, and `python manuscript/check_numbers.py` verifies every number. Submission documents are in `submission/`, and the change log against v0.2 is in `reports/REVISION_DIFF.md`.

Main findings:
- Band gaps are strongly homophilous on the MOF similarity network, but the communication topology has only a small, objective-dependent effect.
- Graph-SOCIAL's gains concentrate on the lowest band gaps (O2), whose top MOFs lie near the center of the chemistry-aware embedding. A centroid-only policy matches it there, and SOCIAL's population contracts toward the centroid.
- Ensemble Thompson sampling ranks first (13 of 16 cells). The strengthened ARD GP-EI ranks third, and Graph-SOCIAL ranks sixth of ten.
- Several analyses were added after the results were known; they are documented as deviations D9–D13 in `reports/DEVIATIONS.md`:
  - O3 redefined as a target-value objective;
  - ARD GP-EI with hyperparameter refits every 10 evaluations;
  - unique-building-block recall;
  - the post hoc center-bias analyses, including a shifted-optimum control on the classic benchmark functions.

Re-running stage 7 or 8 requires `--force` because gate 6 is recorded as FAIL. Stage 8 refuses to run on incomplete results unless `--provisional` is given.

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
