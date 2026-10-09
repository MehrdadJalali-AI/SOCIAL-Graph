# Deviations and decision log

Each entry gives the date, the phase, what differs from the specification, why, and its status.

## D1 — Gate 1 FAIL: the published φ = 0.9 edge count is not reproducible (Phase 1, 2026-10-01) — **RESOLVED by user override**

**Decision (user, 2026-10-01):** keep the specified recipe (Morgan r=2/2048, 0.7·linker + 0.3·metal, mendeleev metals) as primary for QMOF. Gate 1 stays recorded as FAIL and the pipeline continues past it (`gate1_override: true` in `configs/default.yaml`, equivalent to `--force` for gate 1 only). The code-faithful recipe (RDKit path fingerprint, 0.9/0.1) is a Phase 3 sensitivity graph.


- With the specified recipe (Morgan r=2/2048, SIM = 0.7·linker + 0.3·metal, mendeleev metals), φ = 0.9 gives **13,196 edges** (−31.5%; mean degree 13.20 vs 19.256).
- None of the 12 diagnostic recipes (Morgan/RDKit × w ∈ {0.7, 0.9} × metal ∈ {mendeleev, mordredcommunity, original vectors}) gets within ±5% at φ = 0.9. The range is 13,034–14,757.
- **The authors' own released adjacency matrix gives 15,901 edges at φ = 0.9 (−17.5%).** The specification therefore cannot be met even with the original similarity values.
- The published 19,266 edges is reached at φ ≈ 0.68–0.76, depending on the recipe: 0.679 for the code-faithful recipe (RDKit, 0.9/0.1) and 0.749 for the spec recipe (Morgan, 0.7/0.3). The released matrix has 19,753 edges at φ = 0.7 (+2.5%).
- Metal source matters little: at most ±2% in edge count. Mordred vs mendeleev is not the cause.
- **Conclusion:** the reported "φ = 0.9 → 19,266 edges" is most consistent with a graph thresholded at about 0.7. The paper's description (Morgan, 0.7/0.3) also differs from its released code (RDKit path fingerprint, 0.9/0.1).
- Per the gate rule, the pipeline stopped after Phase 1. Later phases were not started.

## D2 — Metal descriptor source (Phase 1)

mendeleev 1.3.0 replaces Mordred, as specified. Electron affinity is missing in mendeleev for Cd, Er, Gd, Hg, Ho, Mg, Mn, Sm, Th, U and Zn. These elements have unbound anions, so EA is set to 0 eV, which also enters the Mulliken term (IE1 + EA)/2. Agreement with the original vectors is r = 1.00 (AN, AW), 0.99 (P), 0.93 (ME), 0.82 (AR) and 0.50 (EA). `mordredcommunity` 2.0.7 installs on Python 3.13 but has no electron-affinity table, so its variant borrows EA from mendeleev.

## D3 — Unparseable linker SMILES in the 2k table (Phase 1)

16 of the first 2,000 SMILES do not parse in RDKit 2026.3. Their linker similarity is set to 0, which is what the original `Similarity.py` does via its exception handler.

## D4 — SOCIAL implementation source (planned, Phase 5)

The SOCIAL-OPTIMIZATION repository's `optimizer.py` (commit c20de23) no longer matches the paper. It adds presets, LOTUS hybrids, rank-based influence and schedule variants. As the specification requires, SOCIAL-MGN will port the paper's Eqs. 8–17, Algorithm 1 and Table 2 directly, including the log-ratio influence formula. Only the `BudgetedObjective` counting/forbidding pattern is reused.

## D5 — Stage code layout

Stage logic lives in `src/graphsocial/stages/phaseN.py`; `run.py` dispatches to it. The module list in Section 2 of the spec is otherwise unchanged.

## D6 — Implementation choices where the specification leaves room (Phases 2–8)

| # | Topic | Choice | Reason |
|---|---|---|---|
| a | Embedding block weights | After standardisation, each block (2,048 fingerprint bits; 6 metal properties) is divided by √(#columns), then weighted 0.7 / 0.3 before PCA-32. | Without this, 2,048 bit columns would swamp the 6 metal columns and the 0.7/0.3 weights would mean nothing. |
| b | Centrality scale | Betweenness, degree and PageRank are max-normalised to [0, 1]. | SOCIAL's weight α·c_j + β·I_j needs c and I on comparable scales. Raw normalised betweenness is about 10⁻³. |
| c | Sampled betweenness | For N > 5,000: igraph subset betweenness from k = 1,000 seeded random sources, rescaled by N/k. | Sampled Brandes, as specified. |
| d | "Long-range" edges (topology b) | Uniformly random node pairs whose endpoints lie in different Leiden communities of MOFGalaxyNet(φ). | Gives a concrete meaning to "long-range". |
| e | Reference communities | All search runs, both for mutation and for family coverage, use the Leiden partition of MOFGalaxyNet(φ*), including runs on other topologies or φ. | Keeps family coverage comparable across topologies. |
| f | Agent with no neighbours | x ← (1−γ_t−δ_t)·x + γ_t·x_gbest + δ_t·x_elite (then sync, mutation). | As the specification states. Algorithm 1 would keep x unchanged. |
| g | Mutation | Worse-than-median agents with probability p_m jump to a random unevaluated MOF in the community with the fewest evaluated MOFs (ties broken at random). The periodic perturbation of Algorithm 1 (t mod 10 = 0, probability 0.05, U(−0.5, 0.5)^D) is kept in embedding units. `social_ws` uses the paper's continuous mutation (s_t schedule). | Specification plus paper. |
| h | Iterations and leftovers | The initial design counts as iteration 0. T = ⌊B/P⌋ iterations in total; B − T·P leftover evaluations go to the unevaluated MOFs nearest the elite. The same rule applies to DE, PSO and GA. | Keeps every method at exactly B evaluations. |
| i | Minimum budget | budget = max(⌈frac·N⌉, 40). On the full data this applies only to O4 at 0.5% (N = 5,359: 27 → 40 evaluations, i.e. 0.75%). All other budgets are exact; the smallest is 44. | So that P = 20 has at least one update iteration. |
| j | Greedy walk | Best-first: take a random unevaluated neighbour of the best evaluated MOF that still has one. If none of the evaluated MOFs has one, restart at a random unevaluated MOF. | Interpretation of "best-first graph walk". |
| k | Ensemble TS | Each step, only the sampled forest (100 trees) is fitted on its own bootstrap resample. | Same in distribution as refitting all 10 members, and 10× cheaper. GNN ensemble replaced, as the specification allows. |
| l | GP-EI | Constant × Matern(ν=2.5, isotropic) + White; normalize_y; one optimiser start; EI with ξ = 0. | CPU cost. |
| m | H3 event | "Hit in a previously unvisited community" = the first hit found in that Leiden community during the run. Base rate = share of all SOCIAL-MGN update evaluations whose top-weighted neighbour is in the top betweenness decile. One-sided binomial test, aggregated over all benchmark runs. | |
| n | O3 hit set | 1% of MOFs closest to the window centre (2.0 eV) when more than 1% lie inside the window. | As specified. In QMOF the window holds far more than 1%. |
| o | Smoke mode | Deterministic 600-MOF subsample, 2–3 seeds, 2 budgets, 20 null graphs. Gates are evaluated but not enforced. | Under 10 minutes. |
| p | Run caching | Run files are keyed by (method, variant, objective, budget, P, community φ, seed), so configurations shared between the benchmark, ablations and φ×ρ grid run once. | Resumability and no duplicate work. |

## D7 — Gate 6 FAIL; Phases 7–8 forced by user decision (2026-10-01)

- Pilot (O2, 2% budget, seeds 0–9, φ* = 0.7): MOFGalaxyNet vs degree-preserving random topology gave +0.014 final recall (one-sided Wilcoxon p = 0.195). With +ρ = 0.10 the difference was +0.015 (p = 0.087). Gate 6 is **FAIL** and stays recorded as FAIL. **H1 is not supported at pilot scale.**
- **Decision (user):** run Phases 7–8 with `--force`, with no change to any hyperparameter or default. The topology carried forward is still chosen by the pre-registered Phase 6 rule (highest mean pilot recall among MOFGalaxyNet variants): `mgn_phi0.70_rho0.10`.
- **Ablations added at the user's request** (O2, O3, 2% budget, seeds 0–29):
  1. `no_neighbor`: SOCIAL-MGN with α = β = 0 throughout, so the neighbour term is off.
  2. Decoupled embedding: the snap/update space is built from geometric/structural descriptors only (PLD, LCD, density, log volume, log atom count from QMOF; standardised, 5-d, no PCA), with no linker fingerprints or metal descriptors. It is run with neighbourhoods from (a) MOFGalaxyNet(φ*) and (c) the degree-preserving random topology. Mutation still uses the reference Leiden communities. These descriptors were available for every MOF, so the random-projection fallback was not needed.
- **Additional reporting** (RESULTS.md):
  - random-search recall checked against its analytical expectation B/N (exact hypergeometric test);
  - sizes of the top-1% hit sets;
  - how initial-design evaluations are counted (they count toward the budget and all metrics);
  - Cliff's delta and achieved power (paired t at the observed d_z, plus bootstrap Wilcoxon) for the topology comparisons.

## D8 — Tie handling in the Wilcoxon test (Phase 8, 2026-10-01)

Recall values are multiples of 1/K. Phase 6 computed the pilot tests from in-memory values, Phase 8 from CSV, and floating-point noise broke exact ties differently (pilot p = 0.195 / 0.087 in Phase 6 vs 0.166 / 0.076 in Phase 8). Paired differences are now rounded to 10 decimals before ranking. Phases 6 and 8 were re-run from the cached runs, which did not need recomputing. The Gate 6 outcome (FAIL) is unchanged.

## D9 — O3 redefined as a target-value objective (revision, 2026-10-01)

The window form f = max(0, 1.5 − gap, gap − 2.5) is zero for all 2,251 MOFs inside the window, while the hit set is the 87 MOFs closest to 2.0 eV. The search signal therefore could not separate hits from other in-window MOFs, and simple regret was 0 for every method. O3 is now f = |gap − 2.0 eV|. The hit set is the 1% lowest f (ties broken by index), which is verified to be the same 87 MOFs. All O3 runs were repeated: 10 methods × 4 budgets × 30 seeds, plus the O3 ablations, topology comparisons and φ×ρ grid. The v0.2 run files are archived in `results/v0.2/runs_replaced/`.

## D10 — GP-EI strengthened (revision, 2026-10-01)

- **Kernel:** Constant × Matérn(ν = 2.5, ARD with 32 length scales) + White, with normalised targets.
- **Bounds:** constant 10⁻³–10³, length scales 10⁻²–10³, noise 10⁻⁸–1. Not specified in the brief; chosen so the bounds cover the standardised targets and the embedding scale.
- **Optimiser:** 5 restarts. Each refit is warm-started from the previous step's fitted hyperparameters, which is the first optimiser start; the 5 random restarts follow.
- **Acquisition:** EI for minimisation with ξ = 0.01. The incumbent is the best observed f in original units (sklearn returns de-normalised predictions).
- **Checks:** a unit test of the EI sign was added.
- **Scope:** all GP-EI runs were repeated for every objective, budget and seed. The v0.2 isotropic runs are archived.
- **Cost:** estimated before starting at about 179 CPU-hours, ~22 h wall with 8 workers. 87% of this is the 5%-budget runs, because fit cost grows as about n^2.1 per refit over ~425 sequential refits.

## D11 — Unique-building-block recall (revision, post hoc)

This is computed from the existing evaluation logs, with no new runs. A hit counts once per distinct (linker, metal) group. Duplicate-group membership of hits is counted within each objective's pool.

## D12 — GP-EI hyperparameters re-optimized every 10 evaluations (revision, 2026-10-03)

The ARD GP-EI of D10 re-optimized all 34 hyperparameters (with 5 restarts) after every evaluation. That made a 5%-budget run take about 2.5 h, and the full set about 30 h of wall time. **Decision (user, 2026-10-03):** keep the ARD kernel, 5 restarts, normalized targets and ξ = 0.01, but re-optimize hyperparameters only at the first step and then every 10 evaluations (warm-started from the previous fit). In between, the GP is conditioned on all evaluated MOFs with the current hyperparameters, so every acquisition uses all observations. This is standard practice in Bayesian optimization.

- **Speed:** one O1 2% run (174 evaluations) took 29 s, against about 9 min with refits at every step.
- **Same-seed check:** that run found 16 hits; the every-step run with the same seed found 13.
- **Rerun:** all 480 GP-EI runs are redone with this schedule for one consistent protocol.
- **Archive:** the 360 completed every-step runs are kept in `results/archive/gp_ei_refit_every_step/` and are compared with the new runs for the 0.5–2% budgets in `reports/GP_EI_CHECKS.md`.

## D13 — Post hoc center-bias analyses (revision, 2026-10-03)

These analyses were not pre-specified. They were added after a separate pilot (SOCIAL on VSA process design, outside this repository's scope) showed that SOCIAL's population contracts toward the centroid of its search box. They are reported as exploratory.

1. **Hit position:** where each objective's hits lie relative to the embedding centroid, in the chemistry-aware and geometry-only embeddings (`center_hits.csv`).
2. **Centroid-only policy:** evaluate the B MOFs closest to the centroid, without using property values (`center_baseline.csv`).
3. **Contraction:** the distance-to-centroid percentile of each method's evaluations, early versus late in each run, at the 2% budget (`center_contraction.csv`).
4. **Synthetic control on classic benchmark functions:** the functions and bounds of the SOCIAL repository are evaluated with the optimum at its usual position and with the optimum shifted by ±0.6 of the half-range. Settings: D = 30, 30,000 evaluations, 10 seeds. Methods: the published SOCIAL implementation (default preset), SOCIAL Algorithm 1, CMA-ES, DE, PSO and random search. Code is in `analysis/center_bias_benchmarks/`; results are in `results/tables/bench23_runs.csv` and `reports/BENCH23_CENTER_BIAS.md`. Schwefel 2.26 is excluded from the error comparison because its reference optimum in the repository is 0 while its true minimum depends on D.

**Consequences for the manuscript:**
- The interpretation changed from "the chemistry-aware embedding drives efficiency" to "search-space geometry governs efficiency". The data underlying the earlier statement are unchanged.
- The title and abstract were revised accordingly, and a Results subsection, Discussion paragraphs and SI tables were added.

## D14 — v0.4 revision: CMA-ES baseline and verification analyses (2026-10-03)

None of these were pre-specified. No pre-specified hyperparameter, seed set, hit definition or evaluation rule was changed, and no existing run was redone.

1. **CMA-ES baseline (`methods/cmaes_snap.py`).** Settings:
   - `cma` 4.5.0 with default settings, except popsize = 10 and sigma0 = 0.3 × the mean range of the embedding;
   - the shared initial design is injected as the first generation;
   - each point snaps to the nearest unevaluated MOF, and the remainder rule is the same as for DE, PSO and GA.

   It ran on 4 objectives × 4 budgets × seeds 0–29 (480 runs), and none of them stopped early (`results/revision/cmaes_early_stop.csv`). Friedman ranks, pairwise tests and the CD diagram now use 11 methods.
2. **Centroid-only policy.** The main variant now includes the shared ten-MOF initial design; the variant without it is still reported (`results/tables/center_baseline.csv`, `results/revision/centroid_policy_variants.csv`).
3. **Contraction under ablations** (`results/tables/center_contraction_ablations.csv`): the default plus the sync off, no-neighbor, elite off and mutation off ablations, on O2 and O3 at 2%.
4. **Verification outputs** in `results/revision/`:
   - the pilot random-search check;
   - the iteration accounting;
   - the stage 4 rerun, whose homophily output is identical to the earlier one.

## D15–D21 — Revision 2 (post hoc, 2026-10-07)

None of these analyses were pre-specified. They were added at the second revision. Pre-specified hyperparameters, seed sets, hit definitions and existing results are unchanged.

- **Code and outputs:** `scripts/revision2_analyses.py` (subcommands `stats`, `homophily`, `dedup`); outputs in `results/revision2/`.
- **In the manuscript:** every analysis below is labelled as added at revision.

### D15 — Confirmatory topology seeds
- **What:** the RQ1 topology comparisons are recomputed on seeds 10–29 only.
- **Why:** pilot seeds 0–9 were used to select ρ = 0.10.
- **Reported:** alongside the 30-seed values.
- **Output:** `topology_tests.csv`.

### D16 — Paired effect sizes
- **What:** the matched-pairs rank-biserial correlation r_rb replaces Cliff's δ.
  - r_rb = (W+ − W−)/(W+ + W−), with zero differences dropped as in the signed-rank test.
  - Each r_rb comes with a 95% percentile bootstrap CI of the paired mean difference (10,000 resamples of seeds).
- **Kept:** Cliff's δ remains in one labelled SI column (topology table) for continuity.
- **Outputs:** `topology_tests.csv`, `pairwise_rrb.csv`, `ablations_rrb.csv`.

### D17 — Holm families defined explicitly
There are four families:
- **(a) SOCIAL-MGN vs each baseline:** one family per baseline over its 16 objective–budget cells. Previously the family was one per cell over the 10 baselines; this changes 6 cell-level calls.
- **(b)** the 13 topology tests.
- **(c)** the 36 non-representation ablation tests.
- **(d)** the 16 random-search checks.

Output: `holm_families.csv`.

### D18 — Friedman ranks
- **Status:** ranks are declared descriptive.
- **Added:**
  - per-budget sensitivity (4 blocks each);
  - the Nemenyi critical difference, also drawn in Figure 5 (`plots.cd_diagram(..., cd=)`).
- **Outputs:** `friedman_sensitivity*.csv`.

### D19 — RQ2 at the run level
- **What:** the evaluation-level binomial test is replaced by a run-level analysis.
- **Method:** per SOCIAL-MGN run, the new-family top-decile share minus the run's base rate, tested with a cluster bootstrap over runs (10,000 resamples).
- **Output:** `rq2_cluster*.csv`.

### D20 — Duplicate controls
- **(a) Collapsed building blocks:** homophily at φ = 0.7 on the network with building blocks collapsed (mean gap per block; blocks linked if any members are linked), against 100 degree-preserving nulls. Output: `homophily_collapsed.csv`.
- **(b) Deduplicated pool, new runs:**
  - one random MOF per building block per seed, chosen with RNG seed 10,000 + seed;
  - hit sets recomputed as the top 1% of the reduced pool;
  - SOCIAL-MGN (default configuration and topology), random search, ensemble TS, CMA-ES and the centroid policy with the shared initial design;
  - O1–O4 at 2% and 5%, seeds 0–29;
  - centralities recomputed on the induced subgraph.
  - Outputs: `dedup_*.csv` and `dedup_runs/`.

### D21 — Reporting additions
- **Enrichment factor** (final recall / (B/N)): per cell, in `full_benchmark.csv`, which is also the separate SI file.
- **Contraction by objective:** distance-to-centroid percentiles by objective in `contraction_by_objective.csv`.
- **Figure 1:** in the embedding panel, the rose family is placed around agents i and j.
- **TOC graphic:** redrawn.

## D22 — Influence shift computed from evaluated MOFs only (correction, 2026-10-07)

### The problem
- **Leak:** SOCIAL's influence (eq 3) needs a non-negative objective, f′ = f − shift. The implementation took the shift as the minimum of f over the *whole pool* (`Objective.shift`), which includes MOFs that had not been evaluated.
- **Consequence:** the neighbor weights of SOCIAL-MGN and SOCIAL-WS depended on the value of the best MOF in the pool before that MOF was found. This is information that a real campaign does not have (oracle leakage).
- **Scope:** no other method used the shift. Global best and elite memory always used evaluated values only.

### The correction
- **Code:** `methods/graph_social.py` now uses f′ = f − (smallest value among the MOFs evaluated so far in the run). `Objective.shift` is documented as analysis-only.
- **Test:** `test_graph_social_does_not_use_pool_minimum` passes on the corrected code and fails on the previous code.

### Rerun
- **What was rerun:** every SOCIAL-MGN and SOCIAL-WS run: pilot, benchmark, ablations, φ × ρ grid and deduplicated pool.
- **Run keys:** corrected runs carry the tag `__fobs`. The previous pool-shift runs (5,040 files without the tag) are kept on disk in `results/runs/` for comparison and are no longer loaded.
- **Downstream:** stages 6–8, `scripts/revision2_analyses.py` (outputs in `results/revision2/`; deduplicated runs in `dedup_runs_fobs/`), all tables, numbers and figures.
- **Pilot selection:** unchanged; it still selects ρ = 0.10, and gate 6 is still FAIL.

### History (from git)
| | Commit | Date |
|---|---|---|
| Pool-minimum shift introduced (`fshift = prob.objective.shift`; `fs = f - fshift`) | `7a8215c` | 2026-10-01 |
| Replaced by the running minimum (`fs = f - min(oracle.values)`) | `c2177bf` | 2026-10-07 |

### Pilot and the topology selection
The topology was selected on the **original** pilot, i.e. with the pool-minimum shift. Those outputs are preserved in `results/revision2/pilot_original_*.csv`, extracted from commit `c2177bf^`.

| Comparison (vs degree-preserving random unless noted) | Original pilot | Corrected pilot |
|---|---|---|
| MOFGalaxyNet | +0.0138, p = 0.195 | +0.0080, p = 0.308 |
| ρ = 0.10 variant | +0.0149, p = 0.087 | +0.0103, p = 0.156 |
| MOFGalaxyNet vs Watts–Strogatz | +0.0046, p = 0.299 | −0.0023, p = 0.531 |

- **Original selection:** ρ = 0.10 had the highest pilot mean (0.064).
- **Corrected pilot:**
  - MOFGalaxyNet's recall is unchanged (0.063 ± 0.027).
  - ρ = 0.10 still has the highest mean among the MOFGalaxyNet variants (57 vs 55 hits over 10 seeds), so the selection rule gives the same choice.
  - Overall it is tied with Watts–Strogatz (57 hits).
- **Gate 6:** FAIL in both versions.

### Effect
- **SOCIAL-MGN O2 at 2%:** final recall 0.063 → 0.067.
- **Benchmark recall:** SOCIAL-MGN and SOCIAL-WS final recall changes by at most 0.014 in any cell (`results/revision2/pool_shift_comparison.csv`).
- **Ranks:** SOCIAL-MGN's mean Friedman rank is unchanged (6.69, 7th of 11).
- **Conclusions:** unchanged. No topology or ablation test survives Holm correction, and the O2 centroid pattern persists.

## D23 — Renaming, verification of the D22 disclosure and submission formatting (2026-10-09)

### Renaming (display labels only)
- **Method:** Graph-SOCIAL is now called **SOCIAL-MGN** (SOCIAL optimization on MOFGalaxyNet). The baseline keeps the name SOCIAL-WS (SOCIAL optimization on a Watts–Strogatz small-world graph).
- **Where:** figure labels (`plots.DISPLAY`, `scripts/make_scheme.py`), report text, docstrings and comments.
- **Unchanged:** internal identifiers (`graphsocial` package, `graph_social` method key, run keys and file names), so every run file still loads.
- **Regenerated from their scripts:** Figure 1 (`make_scheme.py`), Figure 2 (`make_figure2.py`), and stage 8 figures F3–F10. Every table in `results/tables/` is byte-identical after the rerun. In `results/revision2/`, `scripts/revision2_analyses.py stats` changed only the family label in `holm_families.csv`.

### D22 re-verification from the raw run files
Pilot: O2, 2% budget, seeds 0–9, one-sided paired Wilcoxon against the degree-preserving random topology unless noted. Original = pool-minimum shift (run files without `__fobs`); corrected = running minimum (`__fobs`).

| | Original | Corrected |
|---|---|---|
| MOFGalaxyNet | +0.0138, p = 0.195 | +0.0080, p = 0.308 |
| ρ = 0.10 variant | +0.0149, p = 0.087 | +0.0103, p = 0.156 |
| MOFGalaxyNet vs Watts–Strogatz | +0.0046, p = 0.299 | −0.0023, p = 0.531 |
| Highest MOFGalaxyNet-variant mean | ρ = 0.10 (0.064; 56 hits) | ρ = 0.10 (0.066; 57 hits), tied with Watts–Strogatz (57) |

- **MOFGalaxyNet:** recall is 0.063 ± 0.027 in both versions. Its recall curves are identical seed by seed.
- **Pilot r_rb:** 0.20 for MOFGalaxyNet vs degree-preserving random (corrected).
- **Conclusions:** no conclusion changed.

### Separate full-benchmark file
- **Problem:** `SI_full_benchmark.csv/.xlsx` had been exported before the D22 rerun. Its 31 SOCIAL-MGN and SOCIAL-WS rows held pre-correction values.
- **Fix:** the file is now generated by `manuscript/make_benchmark_file.py` from `results/revision2/full_benchmark.csv`.
- **Unaffected:** the tables and numbers in the manuscript and SI were already based on the corrected runs.

### Figure 2 layout
Changes to `scripts/make_figure2.py`:
- tighter panel layout;
- the neighbors in zoom c are drawn as open circles in their community color, not as gray background nodes;
- community captions in zoom d are placed so that they avoid nodes and leader lines.

The data and selection rules are unchanged.
