# Phase 2 — QMOF data preparation

**Gate 2 (soft): PASS** — 8697 MOFs after filtering (threshold 3000); HSE06 subset 5359

Source: QMOF Figshare v18 (`configs/data_versions.yaml`). Linkers come from the MOFid fields; the metal is taken from the composition (`info.formula`), excluding H, B, C, N, O, Si, P, S, Ge, As, Se, Te, the halogens and the noble gases.

## Filters and exclusion counts

| filter / exclusion reason            |   excluded |   remaining |
|:-------------------------------------|-----------:|------------:|
| start (all QMOF entries)             |          0 |       20372 |
| no MOFid linker information          |       2828 |       17544 |
| empty linker list                    |          0 |       17544 |
| more than one distinct linker        |       7432 |       10112 |
| linker SMILES not parseable by RDKit |        441 |        9671 |
| linker is not organic (no carbon)    |        215 |        9456 |
| no metal in composition              |          0 |        9456 |
| more than one metal element          |        759 |        8697 |
| metal lacks mendeleev descriptors    |          0 |        8697 |
| missing PBE band gap                 |          0 |        8697 |

Distinct linkers are compared after RDKit canonicalisation. 'Organic' means the linker contains carbon.

## Resulting dataset

- MOFs: 8697
- Distinct linkers: 3430; distinct metals: 59
- PBE gap: mean 2.218 eV, std 1.165, range 0.002–6.271
- HSE06 gap available: 5359 MOFs (O4 runs if ≥ 1500)
- Metal properties set to 0 (missing EA in mendeleev): Cd, Er, Gd, Hg, Ho, Mg, Mn, Np, Pu, Sm, Th, U, Zn

## Building-block duplicates (identical linker + metal)

- Groups: 6065; groups with more than one MOF: 1134, covering 3766 MOFs (43.3%). Largest group: 82.
- Mean within-group PBE-gap std (groups > 1): 0.281 eV vs global std 1.165 eV. All entries are kept; `bb_group` / `bb_group_size` flag them.

## Top metals

| metal   |   count |
|:--------|--------:|
| Zn      |    1480 |
| Cu      |    1326 |
| Cd      |    1287 |
| Ag      |     591 |
| Co      |     408 |
| Mn      |     386 |
| Al      |     362 |
| Ni      |     263 |
| Ca      |     170 |
| K       |     147 |
| Na      |     140 |
| Fe      |     121 |
| Li      |     119 |
| Tb      |     117 |
| Sn      |     102 |

Embedding: 32-d PCA of [0.7 × standardised linker bits | 0.3 × standardised metal vector]. Runtime 16 s.
