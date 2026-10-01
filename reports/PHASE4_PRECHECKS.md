# Phase 4 — Landscape pre-checks

**Gate 4: PASS** — PBE homophily significant (p<0.01) at φ ∈ [0.7, 0.8, 0.9]; strongest at φ*=0.7

## Homophily

For each node with at least one neighbour: the correlation between its band gap and its neighbours' mean band gap, plus the value-assortativity coefficient. Null model: 100 degree-preserving randomisations of the same graph (10·|E| swaps each). Empirical one-sided p = (1 + #null ≥ observed)/(1 + n). For HSE06, the graph is induced on the MOFs that have an HSE06 gap.

|    phi | target   |   n_nodes |   pearson |   null_mean_pearson |   z_pearson |   p_pearson |   spearman |   z_spearman |   p_spearman |   assortativity |   null_mean_assortativity |   z_assortativity |   p_assortativity |
|-------:|:---------|----------:|----------:|--------------------:|------------:|------------:|-----------:|-------------:|-------------:|----------------:|--------------------------:|------------------:|------------------:|
| 0.7000 | PBE      |      7622 |    0.5148 |              0.0010 |     45.5947 |      0.0099 |     0.5120 |      39.3042 |       0.0099 |          0.1786 |                   -0.0005 |           92.1770 |            0.0099 |
| 0.7000 | HSE06    |      4518 |    0.6557 |             -0.0010 |     38.7211 |      0.0099 |     0.6391 |      34.8831 |       0.0099 |          0.3396 |                   -0.0009 |          100.6064 |            0.0099 |
| 0.8000 | PBE      |      6947 |    0.4848 |             -0.0027 |     38.9962 |      0.0099 |     0.4689 |      35.5196 |       0.0099 |          0.1837 |                   -0.0010 |           70.5776 |            0.0099 |
| 0.8000 | HSE06    |      4077 |    0.6574 |             -0.0010 |     39.2863 |      0.0099 |     0.6404 |      39.5600 |       0.0099 |          0.3604 |                   -0.0017 |           94.5865 |            0.0099 |
| 0.9000 | PBE      |      6584 |    0.4842 |             -0.0003 |     43.8731 |      0.0099 |     0.4632 |      41.7601 |       0.0099 |          0.2027 |                   -0.0016 |           62.0993 |            0.0099 |
| 0.9000 | HSE06    |      3837 |    0.6575 |             -0.0004 |     46.8178 |      0.0099 |     0.6456 |      41.9125 |       0.0099 |          0.3602 |                   -0.0029 |           77.8566 |            0.0099 |

φ* (strongest observed PBE homophily, used for the pilot): **0.7**.

## Degeneracy (identical linker + metal)

|   groups |   groups_size_gt1 |   mofs_in_dup_groups |   largest_group |   mean_within_group_std_pbe |   median_within_group_std_pbe |   global_std_pbe |
|---------:|------------------:|---------------------:|----------------:|----------------------------:|------------------------------:|-----------------:|
| 6065.000 |          1134.000 |             3766.000 |          82.000 |                       0.281 |                         0.179 |            1.165 |

Within-group spread well below the global spread means that much of the graph homophily comes from building-block duplicates (near-cliques of identical linker and metal).

Figure: `phase4_homophily.png`. Runtime 400 s.
