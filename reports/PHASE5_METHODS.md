# Phase 5 — Graph-SOCIAL and baselines

**Checks: PASS** — 10 methods; invariants hold (O2, budget 174, φ=0.7)

Single-run checks on the real problem (seed 0). The full unit tests are in `tests/` (`python -m pytest`).

| method         |   budget |   evaluations | unique   | budget_exact   | initial_design_first   | deterministic   |   final_recall |   seconds |
|:---------------|---------:|--------------:|:---------|:---------------|:-----------------------|:----------------|---------------:|----------:|
| graph_social   |      174 |           174 | True     | True           | True                   | True            |          0.057 |     0.130 |
| random         |      174 |           174 | True     | True           | True                   | True            |          0.011 |     0.000 |
| greedy_walk    |      174 |           174 | True     | True           | True                   | True            |          0.046 |     0.010 |
| gp_ei          |      174 |           174 | True     | True           | True                   | True            |          0.034 |     5.850 |
| ensemble_ts    |      174 |           174 | True     | True           | True                   | True            |          0.115 |    15.940 |
| de             |      174 |           174 | True     | True           | True                   | True            |          0.057 |     0.020 |
| pso            |      174 |           174 | True     | True           | True                   | True            |          0.011 |     0.030 |
| ga             |      174 |           174 | True     | True           | True                   | True            |          0.069 |     0.030 |
| static_diverse |      174 |           174 | True     | True           | True                   | True            |          0.011 |     0.070 |
| social_ws      |      174 |           174 | True     | True           | True                   | True            |          0.080 |     0.030 |

Methods: `graph_social` (SOCIAL over the MOF topology, h-hop agent neighbourhoods, community-aware mutation), `random`, `greedy_walk` (best-first walk on the MOF graph), `gp_ei`, `ensemble_ts` (10 bootstrap random forests), `de`/`pso`/`ga` with the snap operator, `social_ws` (original SOCIAL with a Watts–Strogatz agent graph and continuous mutation, plus snap) and `static_diverse` (k-center greedy).
