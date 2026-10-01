# Phase 6 — H1 pilot

**Gate 6: FAIL** — no MOFGalaxyNet variant beats the random topology at p<0.05 (p = 0.195, 0.0869)

Objective O2 (min PBE gap), budget 2% of N (174 evaluations), seeds 0–9, φ* = 0.7, P = 10. Mean ± std over seeds.

| label                    |   runs | final_recall   | recall_auc    | family_coverage   | simple_regret   |
|:-------------------------|-------:|:---------------|:--------------|:------------------|:----------------|
| MOFGalaxyNet+ρ=0.10      |     10 | 0.064 ± 0.037  | 0.027 ± 0.019 | 0.067 ± 0.035     | 0.0038 ± 0.0069 |
| MOFGalaxyNet             |     10 | 0.063 ± 0.027  | 0.018 ± 0.007 | 0.069 ± 0.024     | 0.0032 ± 0.0087 |
| Watts–Strogatz           |     10 | 0.059 ± 0.015  | 0.029 ± 0.009 | 0.056 ± 0.014     | 0.0037 ± 0.0066 |
| degree-preserving random |     10 | 0.049 ± 0.031  | 0.019 ± 0.014 | 0.057 ± 0.033     | 0.0157 ± 0.0416 |
| random search            |     10 | 0.011 ± 0.008  | 0.006 ± 0.004 | 0.016 ± 0.011     | 0.0549 ± 0.0630 |

## Paired one-sided Wilcoxon on final top-1% recall (by seed)

| variant             | vs                       |   mean_diff |   wilcoxon_p_one_sided |   cliffs_delta |
|:--------------------|:-------------------------|------------:|-----------------------:|---------------:|
| MOFGalaxyNet        | degree-preserving random |    0.01379  |              0.1953    |           0.26 |
| MOFGalaxyNet+ρ=0.10 | degree-preserving random |    0.01494  |              0.08691   |           0.22 |
| MOFGalaxyNet        | Watts–Strogatz           |    0.004598 |              0.2988    |           0.14 |
| MOFGalaxyNet        | random search            |    0.05172  |              0.0009766 |           0.96 |

Topology carried forward to Phase 7 (highest mean final recall among MOFGalaxyNet variants): **MOFGalaxyNet+ρ=0.10** (`mgn_phi0.70_rho0.10`).

Figure: `phase6_recall_curves.png`.

## φ × ρ sensitivity at pilot scale (mean final recall)

|    phi |    0.0 |   0.05 |    0.1 |    0.2 |
|-------:|-------:|-------:|-------:|-------:|
| 0.7000 | 0.0632 | 0.0586 | 0.0644 | 0.0575 |
| 0.8000 | 0.0598 | 0.0655 | 0.0609 | 0.0759 |
| 0.9000 | 0.0609 | 0.0563 | 0.0701 | 0.0713 |

Gate 6 failed: the pipeline stops here unless re-run with `--force`.
