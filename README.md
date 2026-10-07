# Graph-SOCIAL

Centrality-guided search over metal–organic framework (MOF) similarity networks for budget-limited discovery.
SOCIAL is used as a model-free, structure-aware acquisition policy: it decides which MOF to evaluate next under a fixed evaluation budget, and its communication topology is the MOFGalaxyNet similarity graph.

This repository accompanies the manuscript "Location over Links: Search-Space Geometry Shapes Social-Network Optimization for Metal–Organic Framework Discovery", submitted to the Journal of Chemical Information and Modeling (release v1.2-jcim).

## Main findings

- Band gaps are strongly homophilous on the MOF similarity network (r = 0.515 vs ≈0 for degree-preserving null graphs), but the communication topology has only a small, objective-dependent effect.
- Graph-SOCIAL's gains concentrate on the lowest band gaps (O2), whose top MOFs lie near the centroid of the chemistry-aware embedding; a centroid-only policy reaches similar or higher recall there, and SOCIAL's evaluations concentrate near the centroid from the first iterations. The pattern persists on a pool deduplicated to one MOF per building block.
- Ensemble Thompson sampling ranks first (highest recall in 13 of 16 objective–budget cells). Over 11 methods, PSO ranks 2nd, GP-EI 3rd, DE 4th, GA 5th, CMA-ES 6th and Graph-SOCIAL 7th (mean Friedman ranks 1.25, 3.16, 4.28, 4.31, 5.09, 5.75 and 6.69; `results/tables/T4b_friedman_ranks.csv`).
- Analyses added after the results were known are documented as deviations D9–D21 in `reports/DEVIATIONS.md`:
  - O3 redefined as a target-value objective;
  - ARD GP-EI with hyperparameter refits every 10 evaluations;
  - unique-building-block recall;
  - the post hoc center-bias analyses, including a shifted-optimum control on the classic benchmark functions;
  - the CMA-ES baseline with the snap operator;
  - the centroid-only policy that includes the shared ten-MOF initial design;
  - revision 2: confirmatory topology seeds 10–29, rank-biserial effect sizes with bootstrap CIs, explicit Holm families, rank sensitivity, a run-level bridge analysis and duplicate controls (`scripts/revision2_analyses.py`, `results/revision2/`).

Re-running stage 7 or 8 requires `--force` because gate 6 is recorded as FAIL. Stage 8 refuses to run on incomplete results unless `--provisional` is given.

## Relation to the original SOCIAL code

This repository implements SOCIAL as specified in Algorithm 1 of the Applied Soft Computing paper (Jalali et al., *Appl. Soft Comput.* 2026, 194, 114914), not the separately released SOCIAL code. The differences between the two are listed in Table S4 of the Supporting Information. Graph-SOCIAL (`src/graphsocial/methods/graph_social.py`) keeps the Algorithm 1 learning rules and adapts them to a finite pool of MOFs; the SOCIAL-WS baseline is the same implementation with a Watts–Strogatz agent graph.

## Data

QMOF version 18 is not redistributed here. `scripts/fetch_data.py` downloads it from Figshare (DOI [10.6084/m9.figshare.13147324](https://doi.org/10.6084/m9.figshare.13147324)) together with a pinned commit of the [MOFGalaxyNet repository](https://github.com/MehrdadJalali-KIT/MOFGalaxyNet), and verifies the SHA-256 checksums listed in `configs/data_versions.yaml`.

If you use the data, please cite QMOF:

- Rosen, A. S.; Iyer, S. M.; Ray, D.; Yao, Z.; Aspuru-Guzik, A.; Gagliardi, L.; Notestein, J. M.; Snurr, R. Q. Machine learning the quantum-chemical properties of metal–organic frameworks for accelerated materials discovery. *Matter* **2021**, *4*, 1578–1597. https://doi.org/10.1016/j.matt.2021.02.015
- Rosen, A. S.; Fung, V.; Huck, P.; O'Donnell, C. T.; Horton, M. K.; Truhlar, D. G.; Persson, K. A.; Notestein, J. M.; Snurr, R. Q. High-throughput predictions of metal–organic framework electronic properties: theoretical challenges, graph neural networks, and data exploration. *npj Comput. Mater.* **2022**, *8*, 112. https://doi.org/10.1038/s41524-022-00796-6

The two papers this work builds on are listed in `docs/REFERENCES.md`.

## Reproducing the paper

The full pipeline is `python -m graphsocial.run --stage all` (see Run below). Stages write tables to `results/tables/` and figures to `results/figures/`. `python scripts/check_numbers.py` then recomputes the Friedman ranks and the topology power analysis from the raw runs and compares them with the tables.

| Manuscript item | Output file(s) (under `results/`) | Produced by |
|---|---|---|
| Figure 1 (method overview) | `figures/F0_graph_social_scheme.*` | `scripts/make_scheme.py` |
| Figure 2 (QMOF network) | `figures/F1_graph_overview.*` | stage 8 |
| Figure 3 (homophily) | `figures/F2_homophily.*` | stage 8 (data from stage 4) |
| Figure 4 (recall curves) | `figures/F3_recall_curves.*` | stage 8 |
| Figure 5 (critical-difference diagram) | `figures/F5_critical_difference.*` | stage 8; CD bar from `scripts/revision2_analyses.py stats` |
| Figure 6 (ablations) | `figures/F6_ablations.*` | stage 8 |
| Table 1 (related work) | — | literature summary, no code |
| Table 2 (adaptation of SOCIAL) | — | method description, no code |
| Table 3 (final recall, 2% and 5%) | `tables/T3_main_results.csv` | stage 8 |
| Table S1 (decision criteria) | `reports/PHASE*_*.md`, `results/state.json` | gates of stages 1, 2, 4 and 6 |
| Table S2 (SOCIAL hyperparameters) | `configs/default.yaml` | configuration |
| Table S3 (QMOF filtering) | `tables/phase2_filters.csv` | stage 2 |
| Table S4 (released SOCIAL code vs Algorithm 1) | — | comparison of the two implementations, no code |
| Table S5 (implementation details) | — | method description, no code |
| Table S6 (hit sets) | `tables/hit_sets.csv` | stage 8 |
| Table S7 (Holm families) | `revision2/holm_families.csv` | `scripts/revision2_analyses.py stats` |
| Table S8 (GP-EI refit schedule) | `tables/gp_refit_check.csv` | `scripts/gp_ei_checks.py` |
| Table S9 (MOFGalaxyNet reconstruction) | `tables/phase1_recipe_fit.csv`, `tables/phase1_diagnostics.csv` | stage 1 |
| Table S10 (search topologies) | `tables/phase3_graph_stats.csv` | stage 3 |
| Table S11 (homophily) | `tables/phase4_homophily.csv` | stage 4 |
| Table S12 (duplicate controls) | `revision2/homophily_collapsed.csv`, `revision2/dedup_summary.csv` | `scripts/revision2_analyses.py homophily`, `... dedup` |
| Table S13 (topology comparisons, confirmatory seeds, power) | `revision2/topology_tests.csv` | `scripts/revision2_analyses.py stats` |
| Table S14 (random search vs expectation) | `tables/random_search_check.csv` | stage 8 |
| Table S15 (benchmark: recall, EF, coverage) | `revision2/full_benchmark.csv` (also SI_full_benchmark.xlsx/.csv) | `scripts/revision2_analyses.py stats` |
| Table S16 (rank sensitivity) | `revision2/friedman_sensitivity*.csv` | `scripts/revision2_analyses.py stats` |
| Table S17 (pairwise tests, Friedman ranks) | `revision2/pairwise_rrb.csv`, `tables/T4b_friedman_ranks.csv` | `scripts/revision2_analyses.py stats`, stage 8 |
| Table S18 (ablations) | `tables/T5_ablations.csv`, `revision2/ablations_rrb.csv` | stage 8, `scripts/revision2_analyses.py stats` |
| Table S19 (position of hits) | `tables/center_hits.csv` | stage 8 (`center_analysis.py`) |
| Table S20 (centroid-only policy) | `tables/center_baseline.csv` | stage 8 (`center_analysis.py`) |
| Table S21 (distance to centroid by objective) | `revision2/contraction_by_objective.csv` | `scripts/revision2_analyses.py stats` |
| Table S22 (distance to centroid under ablations) | `tables/center_contraction_ablations.csv` | stage 8 (`center_analysis.py`) |
| Table S23 (shifted-optimum control) | `tables/bench23_runs.csv` | `analysis/center_bias_benchmarks/run_benchmarks.py` |
| Table S24 (building-block composition) | `tables/bb_hit_composition.csv` | stage 8 |
| Table S25 (unique-building-block recall) | `tables/bb_recall.csv` | stage 8 |
| Table S26 (run-level bridge analysis) | `revision2/rq2_cluster.csv` | `scripts/revision2_analyses.py stats` |
| Table S27 (runtime) | `tables/T6_runtime.csv` | stage 8 |
| Figure S1 (φ × ρ sensitivity) | `figures/F7_phi_rho_heatmap.*` | stage 8 |
| Figure S2 (coverage vs recall) | `figures/F4_coverage_vs_recall.*` | stage 8 |
| Figure S3 (bridge MOFs) | `figures/F8_h3_bridge_nodes.*` | stage 8 |

## Install (Python 3.13, CPU only)

Every dependency ships Python 3.13 wheels or conda-forge builds; nothing is compiled.

```bash
git clone https://github.com/MehrdadJalali-AI/SOCIAL-Graph.git graph-social && cd graph-social
python -m pip install -r requirements.txt      # or: conda env create -f environment.yml
python scripts/check_env.py                    # versions, CPUs, RAM, internet access
python scripts/fetch_data.py                   # pinned MOFGalaxyNet commit + QMOF v18 (392 MB), checksum-verified
```

## Run

### From an IPython console

```
!git clone https://github.com/MehrdadJalali-AI/SOCIAL-Graph.git graph-social
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
analysis/     center_bias_benchmarks/ (shifted-optimum control on the classic benchmark functions)
configs/      default.yaml, smoke.yaml, data_versions.yaml (pinned data + checksums)
docs/         REFERENCES.md, MOFGalaxyNet paper (J. Cheminform. 2023, open access)
scripts/      check_env.py, fetch_data.py, check_numbers.py, make_scheme.py (Figure 1), gp_ei_checks.py,
              revision_phase1.py, revision2_analyses.py, revision_diff.py
src/graphsocial/
  data/       metals.py (mendeleev/mordred descriptors), linkers.py (fingerprints, Tanimoto), mofgalaxynet_2k.py
  graph/      build.py (block-wise SIM >= phi thresholding), topologies.py, centrality.py, communities.py
  methods/    graph_social.py + 9 baselines (random, greedy walk, GP-EI, ensemble TS, DE, PSO, GA, CMA-ES,
              static diverse); SOCIAL-WS is a preset of graph_social.py
  center_analysis.py, oracle.py, embedding.py, objectives.py, metrics.py, stats.py, plots.py, report.py
  experiment.py (resumable runs), plan.py (run plans), store.py (cached artifacts)
  stages/     phase1.py … phase8.py
  run.py      CLI
reports/      PHASE*_*.md, DATA_INSPECTION.md, DEVIATIONS.md, GP_EI_CHECKS.md, BENCH23_CENTER_BIAS.md
results/      tables/, figures/, revision/, revision2/ (runs/ is created by the pipeline and not tracked)
tests/        pytest suite
```
