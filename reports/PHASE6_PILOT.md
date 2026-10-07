# Phase 6 — H1 pilot

**Gate 6: FAIL** — no MOFGalaxyNet variant beats the random topology at p<0.05 (p = 0.308, 0.156)

Objective O2 (min PBE gap), budget 2% of N (174 evaluations), seeds 0–9, φ* = 0.7, P = 10. Mean ± std over seeds.

| label                    |   runs | final_recall   | recall_auc    | family_coverage   | simple_regret   |
|:-------------------------|-------:|:---------------|:--------------|:------------------|:----------------|
| MOFGalaxyNet+ρ=0.10      |     10 | 0.066 ± 0.039  | 0.026 ± 0.017 | 0.069 ± 0.038     | 0.0037 ± 0.0069 |
| Watts–Strogatz           |     10 | 0.066 ± 0.022  | 0.033 ± 0.016 | 0.062 ± 0.019     | 0.0029 ± 0.0061 |
| MOFGalaxyNet             |     10 | 0.063 ± 0.027  | 0.018 ± 0.007 | 0.069 ± 0.024     | 0.0032 ± 0.0087 |
| degree-preserving random |     10 | 0.055 ± 0.032  | 0.021 ± 0.015 | 0.059 ± 0.032     | 0.0166 ± 0.0487 |
| random search            |     10 | 0.011 ± 0.008  | 0.006 ± 0.004 | 0.016 ± 0.011     | 0.0549 ± 0.0630 |

## Paired one-sided Wilcoxon on final top-1% recall (by seed)

| variant             | vs                       |   mean_diff |   wilcoxon_p_one_sided |   cliffs_delta |
|:--------------------|:-------------------------|------------:|-----------------------:|---------------:|
| MOFGalaxyNet        | degree-preserving random |    0.008046 |              0.3076    |           0.13 |
| MOFGalaxyNet+ρ=0.10 | degree-preserving random |    0.01034  |              0.1562    |           0.12 |
| MOFGalaxyNet        | Watts–Strogatz           |   -0.002299 |              0.5312    |          -0.03 |
| MOFGalaxyNet        | random search            |    0.05172  |              0.0009766 |           0.96 |

Topology carried forward to Phase 7 (highest mean final recall among MOFGalaxyNet variants): **MOFGalaxyNet+ρ=0.10** (`mgn_phi0.70_rho0.10`).

Figure: `phase6_recall_curves.png`.

## φ × ρ sensitivity at pilot scale (mean final recall)

|    phi |    0.0 |   0.05 |    0.1 |    0.2 |
|-------:|-------:|-------:|-------:|-------:|
| 0.7000 | 0.0632 | 0.0609 | 0.0655 | 0.0563 |
| 0.8000 | 0.0598 | 0.0655 | 0.0609 | 0.0759 |
| 0.9000 | 0.0609 | 0.0563 | 0.0701 | 0.0690 |

Gate 6 failed: the pipeline stops here unless re-run with `--force`.
