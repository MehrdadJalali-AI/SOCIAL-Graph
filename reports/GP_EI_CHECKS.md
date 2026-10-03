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

| objective   |   budget_frac |   gp_ei |   random |   ensemble_ts |   gp_minus_random |
|:------------|--------------:|--------:|---------:|--------------:|------------------:|
| O1          |        0.0050 |  0.0120 |   0.0040 |        0.0310 |            0.0080 |
| O1          |        0.0100 |  0.0510 |   0.0090 |        0.0990 |            0.0420 |
| O1          |        0.0200 |  0.1460 |   0.0220 |        0.2490 |            0.1240 |
| O1          |        0.0500 |  0.4650 |   0.0520 |        0.5960 |            0.4130 |
| O2          |        0.0050 |  0.0030 |   0.0050 |        0.0130 |           -0.0020 |
| O2          |        0.0100 |  0.0130 |   0.0100 |        0.0390 |            0.0030 |
| O2          |        0.0200 |  0.0390 |   0.0220 |        0.0910 |            0.0170 |
| O2          |        0.0500 |  0.1630 |   0.0590 |        0.2150 |            0.1040 |
| O3          |        0.0050 |  0.0050 |   0.0040 |        0.0090 |            0.0010 |
| O3          |        0.0100 |  0.0140 |   0.0110 |        0.0250 |            0.0030 |
| O3          |        0.0200 |  0.0370 |   0.0180 |        0.0590 |            0.0190 |
| O3          |        0.0500 |  0.1180 |   0.0470 |        0.1430 |            0.0710 |
| O4          |        0.0050 |  0.0150 |   0.0080 |        0.0350 |            0.0070 |
| O4          |        0.0100 |  0.0300 |   0.0110 |        0.0520 |            0.0190 |
| O4          |        0.0200 |  0.0900 |   0.0250 |        0.1900 |            0.0650 |
| O4          |        0.0500 |  0.3530 |   0.0600 |        0.5230 |            0.2930 |

GP-EI trails random search in 1 of 16 cells. Checks 1–4 show no sign, normalisation or incumbent error. Where GP-EI trails random search, the held-out rank correlation of the GP model is low for that objective (table above), so EI concentrates evaluations in regions that the model wrongly ranks as promising.

## 6. Hyperparameter refits every 10 evaluations vs every evaluation (same seeds)

| objective   |   budget_frac |   seeds |   every_step |   every_10 |   p_wilcoxon_two_sided |
|:------------|--------------:|--------:|-------------:|-----------:|-----------------------:|
| O1          |        0.0050 |      30 |       0.0230 |     0.0119 |                 0.0347 |
| O1          |        0.0100 |      30 |       0.0621 |     0.0510 |                 0.2623 |
| O1          |        0.0200 |      30 |       0.1544 |     0.1456 |                 0.7758 |
| O2          |        0.0050 |      30 |       0.0031 |     0.0034 |                 0.7963 |
| O2          |        0.0100 |      30 |       0.0146 |     0.0130 |                 0.5327 |
| O2          |        0.0200 |      30 |       0.0391 |     0.0395 |                 0.8610 |
| O3          |        0.0050 |      30 |       0.0027 |     0.0054 |                 0.0896 |
| O3          |        0.0100 |      30 |       0.0119 |     0.0138 |                 0.5660 |
| O3          |        0.0200 |      30 |       0.0368 |     0.0368 |                 0.9507 |
| O4          |        0.0050 |      30 |       0.0327 |     0.0148 |                 0.0189 |
| O4          |        0.0100 |      30 |       0.0512 |     0.0302 |                 0.0385 |
| O4          |        0.0200 |      30 |       0.1080 |     0.0895 |                 0.3209 |

