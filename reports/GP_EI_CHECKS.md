# GP-EI diagnostics (revision)

## 1. Sign of expected improvement
For minimisation with incumbent 1.0: EI(mean 0.0) = 0.9945 > EI(mean 2.0) = 0.0040. Candidates predicted below the incumbent are preferred, as required.

## 2–4. Normalisation, incumbent and model quality
A GP was fitted on random training sets of 44 and 174 MOFs for each objective. The table reports:
- the correlation of the posterior mean with the training targets (normalisation round-trip);
- the offset of held-out predictions from the training mean (predictions are in original units);
- the held-out Spearman correlation of the GP and of a random forest with the true objective;
- the hit rate among the 20 held-out MOFs with the highest EI, against the base rate.

| objective   |   n_train |   train_fit_r |   pred_mean_vs_train_mean | incumbent_is_min_observed   |   spearman_gp_heldout |   spearman_rf_heldout |   hit_rate_top20_EI |   hit_rate_random |   median_length_scale |   noise_level |
|:------------|----------:|--------------:|--------------------------:|:----------------------------|----------------------:|----------------------:|--------------------:|------------------:|----------------------:|--------------:|
| O1          |        44 |         0.997 |                     0.231 | True                        |                 0.597 |                 0.533 |               0.050 |             0.010 |               103.477 |         0.032 |
| O1          |       174 |         0.887 |                    -0.010 | True                        |                 0.693 |                 0.698 |               0.050 |             0.010 |               262.261 |         0.280 |
| O2          |        44 |         1.000 |                     0.050 | True                        |                 0.574 |                 0.662 |               0.000 |             0.010 |               351.729 |         0.005 |
| O2          |       174 |         0.953 |                     0.006 | True                        |                 0.696 |                 0.682 |               0.000 |             0.010 |               134.351 |         0.178 |
| O3          |        44 |         1.000 |                     0.068 | True                        |                -0.007 |                 0.001 |               0.050 |             0.010 |               123.955 |         0.000 |
| O3          |       174 |         0.974 |                    -0.054 | True                        |                 0.365 |                 0.389 |               0.000 |             0.010 |               107.028 |         0.196 |
| O4          |        44 |         1.000 |                     0.045 | True                        |                 0.221 |                 0.314 |               0.000 |             0.010 |                28.257 |         0.000 |
| O4          |       174 |         0.852 |                     0.020 | True                        |                 0.521 |                 0.507 |               0.200 |             0.010 |               103.339 |         0.425 |

## 5. GP-EI against random search and ensemble TS (final recall)

| objective   |   budget_frac |    gp_ei |   random |   ensemble_ts |   gp_minus_random |
|:------------|--------------:|---------:|---------:|--------------:|------------------:|
| O1          |        0.0050 |   0.0210 |   0.0040 |        0.0310 |            0.0170 |
| O1          |        0.0100 | nan      |   0.0090 |        0.0990 |          nan      |
| O1          |        0.0200 | nan      |   0.0220 |        0.2490 |          nan      |
| O1          |        0.0500 | nan      |   0.0520 |        0.5960 |          nan      |
| O2          |        0.0050 | nan      |   0.0050 |        0.0130 |          nan      |
| O2          |        0.0100 | nan      |   0.0100 |        0.0390 |          nan      |
| O2          |        0.0200 | nan      |   0.0220 |        0.0910 |          nan      |
| O2          |        0.0500 | nan      |   0.0590 |        0.2150 |          nan      |
| O3          |        0.0050 | nan      |   0.0040 |        0.0090 |          nan      |
| O3          |        0.0100 | nan      |   0.0110 |        0.0250 |          nan      |
| O3          |        0.0200 | nan      |   0.0180 |        0.0590 |          nan      |
| O3          |        0.0500 | nan      |   0.0470 |        0.1430 |          nan      |
| O4          |        0.0050 | nan      |   0.0080 |        0.0350 |          nan      |
| O4          |        0.0100 | nan      |   0.0110 |        0.0520 |          nan      |
| O4          |        0.0200 | nan      |   0.0250 |        0.1900 |          nan      |
| O4          |        0.0500 | nan      |   0.0600 |        0.5230 |          nan      |

GP-EI trails random search in 0 of 16 cells.

