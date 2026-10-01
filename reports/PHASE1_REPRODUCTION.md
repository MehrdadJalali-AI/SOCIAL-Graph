# Phase 1 — Reproduction of the MOFGalaxyNet 2k graph

**Gate 1: FAIL** — phi=0.9: 13196 edges (-31.5%), mean degree 13.196 (-31.5%) vs published 19266 / 19.256

## Input
- Table: `data/raw/MOFGalaxyNet/Data/SMILES_METAL_2000_NoPLD.csv` — first 2000 rows used (the published graph and the released matrix use 2000).
- Distinct linker SMILES: 729; unparseable by RDKit: 16 (linker similarity set to 0, as in the original code).
- Metals recovered from the normalised atomic number (Z = 3 + 89·AN_norm): 48 elements.
- Metal properties filled with 0 (missing electron affinity in mendeleev): Cd, Er, Gd, Hg, Ho, Mg, Mn, Sm, Th, U, Zn.

## Primary recipe (morgan-r2-2048 | w_linker=0.7 | metal=mendeleev)

|   phi |       edges |   mean_degree |   isolated |   mean_degree_non_isolated |   max_degree |
|------:|------------:|--------------:|-----------:|---------------------------:|-------------:|
| 0.200 | 1871873.000 |      1871.873 |      0.000 |                   1871.873 |     1999.000 |
| 0.700 |   26249.000 |        26.249 |    282.000 |                     30.558 |      194.000 |
| 0.800 |   16588.000 |        16.588 |    374.000 |                     20.403 |      105.000 |
| 0.900 |   13196.000 |        13.196 |    439.000 |                     16.907 |      104.000 |

Published reference at φ = 0.9: 19266 edges, mean degree 19.256.

## Diagnostic sweep: edges at each φ

| recipe                                          |     0.2 |   0.7 |   0.8 |   0.9 |
|:------------------------------------------------|--------:|------:|------:|------:|
| morgan-r2-2048 | w_linker=0.7 | metal=mendeleev | 1871873 | 26249 | 16588 | 13196 |
| morgan-r2-2048 | w_linker=0.7 | metal=mordred   | 1889371 | 26470 | 16624 | 13416 |
| morgan-r2-2048 | w_linker=0.7 | metal=original  | 1894926 | 27458 | 16800 | 13359 |
| morgan-r2-2048 | w_linker=0.9 | metal=mendeleev |  896226 | 19270 | 16236 | 14714 |
| morgan-r2-2048 | w_linker=0.9 | metal=mordred   |  903002 | 19327 | 16262 | 14714 |
| morgan-r2-2048 | w_linker=0.9 | metal=original  |  919001 | 19394 | 16205 | 14757 |
| rdkit-2048 | w_linker=0.7 | metal=mendeleev     | 1830722 | 20595 | 15876 | 13034 |
| rdkit-2048 | w_linker=0.7 | metal=mordred       | 1860572 | 20671 | 15960 | 13223 |
| rdkit-2048 | w_linker=0.7 | metal=original      | 1864982 | 21297 | 16155 | 13137 |
| rdkit-2048 | w_linker=0.9 | metal=mendeleev     |  620933 | 18025 | 15650 | 14349 |
| rdkit-2048 | w_linker=0.9 | metal=mordred       |  625102 | 18065 | 15664 | 14351 |
| rdkit-2048 | w_linker=0.9 | metal=original      |  640511 | 18213 | 15693 | 14361 |
| released adjacency matrix (authors)             |  642051 | 19753 | 17233 | 15901 |

## Which recipe produced the authors' released matrix?

Correlation and mean absolute error against the released weighted adjacency matrix (pairs where the released value is non-zero), and the φ at which each recipe yields the published edge count.

| recipe                                          |   corr_vs_released |   mae_vs_released |   phi_for_published_edges |
|:------------------------------------------------|-------------------:|------------------:|--------------------------:|
| rdkit-2048 | w_linker=0.9 | metal=original      |             0.9765 |            0.0008 |                    0.6791 |
| rdkit-2048 | w_linker=0.9 | metal=mordred       |             0.9746 |            0.0068 |                    0.6762 |
| rdkit-2048 | w_linker=0.9 | metal=mendeleev     |             0.9748 |            0.0071 |                    0.6757 |
| morgan-r2-2048 | w_linker=0.9 | metal=mendeleev |             0.8470 |            0.0611 |                    0.7000 |
| morgan-r2-2048 | w_linker=0.9 | metal=mordred   |             0.8468 |            0.0614 |                    0.7000 |
| morgan-r2-2048 | w_linker=0.9 | metal=original  |             0.8494 |            0.0619 |                    0.7000 |
| rdkit-2048 | w_linker=0.7 | metal=mendeleev     |             0.9173 |            0.1329 |                    0.7168 |
| rdkit-2048 | w_linker=0.7 | metal=mordred       |             0.9239 |            0.1353 |                    0.7182 |
| rdkit-2048 | w_linker=0.7 | metal=original      |             0.9231 |            0.1447 |                    0.7261 |
| morgan-r2-2048 | w_linker=0.7 | metal=mendeleev |             0.8038 |            0.1648 |                    0.7491 |
| morgan-r2-2048 | w_linker=0.7 | metal=mordred   |             0.8084 |            0.1669 |                    0.7506 |
| morgan-r2-2048 | w_linker=0.7 | metal=original  |             0.8118 |            0.1760 |                    0.7580 |

## Metal descriptors vs the original (Mordred-era) normalised vectors

| source    | property   |   pearson_vs_original |   max_abs_diff |
|:----------|:-----------|----------------------:|---------------:|
| mendeleev | AN         |                 1.000 |          0.000 |
| mendeleev | AW         |                 1.000 |          0.000 |
| mendeleev | AR         |                 0.821 |          0.278 |
| mendeleev | ME         |                 0.931 |          0.288 |
| mendeleev | P          |                 0.992 |          0.147 |
| mendeleev | EA         |                 0.495 |          0.654 |
| mordred   | AN         |                 1.000 |          0.000 |
| mordred   | AW         |                 1.000 |          0.000 |
| mordred   | AR         |                 0.840 |          0.323 |
| mordred   | ME         |                 0.931 |          0.288 |
| mordred   | P          |                 0.998 |          0.070 |
| mordred   | EA         |                 0.495 |          0.654 |

## Interpretation

- Across all recipes, φ = 0.9 yields 13034–15901 edges (published 19266).
- The authors' released matrix yields 15901 edges at φ = 0.9.
- Best-fitting recipe for the released matrix: rdkit-2048 | w_linker=0.9 | metal=original; the published edge count is reached at φ ≈ 0.68–0.76.

See `reports/DEVIATIONS.md` for the decision record. Runtime: 38 s.
