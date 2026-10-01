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

The SOCIAL-OPTIMIZATION repository's `optimizer.py` (commit c20de23) no longer matches the paper. It adds presets, LOTUS hybrids, rank-based influence and schedule variants. As the specification requires, Graph-SOCIAL will port the paper's Eqs. 8–17, Algorithm 1 and Table 2 directly, including the log-ratio influence formula. Only the `BudgetedObjective` counting/forbidding pattern is reused.

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
| i | Minimum budget | budget = max(⌈frac·N⌉, 40). This only triggers for smoke runs; at N = 8,697 the smallest budget is 44. | So that P = 20 has at least one update iteration. |
| j | Greedy walk | Best-first: take a random unevaluated neighbour of the best evaluated MOF that still has one. If none of the evaluated MOFs has one, restart at a random unevaluated MOF. | Interpretation of "best-first graph walk". |
| k | Ensemble TS | Each step, only the sampled forest (100 trees) is fitted on its own bootstrap resample. | Same in distribution as refitting all 10 members, and 10× cheaper. GNN ensemble replaced, as the specification allows. |
| l | GP-EI | Constant × Matern(ν=2.5, isotropic) + White; normalize_y; one optimiser start; EI with ξ = 0. | CPU cost. |
| m | H3 event | "Hit in a previously unvisited community" = the first hit found in that Leiden community during the run. Base rate = share of all Graph-SOCIAL update evaluations whose top-weighted neighbour is in the top betweenness decile. One-sided binomial test, aggregated over all benchmark runs. | |
| n | O3 hit set | 1% of MOFs closest to the window centre (2.0 eV) when more than 1% lie inside the window. | As specified. In QMOF the window holds far more than 1%. |
| o | Smoke mode | Deterministic 600-MOF subsample, 2–3 seeds, 2 budgets, 20 null graphs. Gates are evaluated but not enforced. | Under 10 minutes. |
| p | Run caching | Run files are keyed by (method, variant, objective, budget, P, community φ, seed), so configurations shared between the benchmark, ablations and φ×ρ grid run once. | Resumability and no duplicate work. |
