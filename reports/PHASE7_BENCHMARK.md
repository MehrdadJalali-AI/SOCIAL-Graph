# Phase 7 — Full benchmark, ablations, sensitivity

5280 benchmark runs, 1380 ablation and 720 sensitivity specs (6840 unique runs); objectives ['O1', 'O2', 'O3', 'O4'].

**Gate 6 was FAIL** (H1 not supported at pilot scale). By user decision this stage runs with `--force`; hyperparameters and defaults are unchanged. Added ablations (O2, O3, 2% budget, all seeds): `no_neighbor` (α = β = 0 throughout) and a decoupled geometric embedding (PLD, LCD, density, log volume, log atom count; no linker or metal information) with neighbourhoods from topologies (a) and (c).

- Graph-SOCIAL topology: `mgn_phi0.70_rho0.10` (from Phase 6); reference communities at φ* = 0.7.
- Budgets: [0.005, 0.01, 0.02, 0.05] of N (minimum 40 evaluations); seeds 0–29.
- O4 runs only if the HSE06 subset has ≥ 1500 MOFs: yes.
- Wall time of this stage: 11 s with n_jobs = 8.

Analysis, tables and figures are produced by stage 8 (`RESULTS.md`).
