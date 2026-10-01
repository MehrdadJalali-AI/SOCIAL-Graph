# Data inspection

Inspected 2026-10-01. Pins and checksums: `configs/data_versions.yaml`.

## 1. MOFGalaxyNet repository (commit f320754)

`Data/` contents:

| File | Shape | Content found |
|---|---|---|
| `SMILES_METAL_2000_NoPLD.csv` | 2,004 rows × 10 cols, **no header** | cols 0–5: min-max normalised metal properties (AN, AW, AR, ME, P, EA); col 6: row index; col 7: CSD refcode (unique); col 8: one linker SMILES; col 9: PLD class 0–3 (1062/425/271/246) |
| `MOF_Features.csv` | 2,004 rows | index + same six metal properties + PLD class (redundant with the above) |
| `Adjacency Matrix.zip` → `Wieghted_2000 (1).csv` | 2000 × 2000, symmetric | weighted similarity matrix, zero diagonal, smallest non-zero value 0.1 |
| `EdgesList_0.9.csv`, `EdgesList-0.7.csv`, `EdgesList-0.2.csv` | directed pairs, both directions listed | 0.9: 15,901 undirected edges on 1,625 nodes; 0.7: 19,753 on 1,737 nodes. The 0.2 file is truncated at Excel's 1,048,576-row limit. |

Findings:

- **No metal symbol column.** The metal is recoverable exactly: AN_norm·89 is an integer for all 48 distinct vectors, so Z = 3 + 89·AN_norm (normalisation range Li…U). Recomputed atomic weights reproduce the AW column (test `test_metal_recovery_from_original_table`).
- 729 distinct linker SMILES in the first 2,000 rows; 16 rows have SMILES that RDKit 2026.3 cannot parse.
- The ME column is constant (0.378) across all lanthanides, which suggests the Mordred-era table filled missing electronegativities with a constant.
- **Released code differs from the paper text.** `Similarity.py` uses `Chem.RDKFingerprint` (path-based, not Morgan) and `alpha = 0.1` for the metal term, i.e. SIM = 0.9·linker + 0.1·metal. The paper text says Morgan fingerprints with weight 0.7. Phase 1 confirms that the released matrix equals the code recipe (MAE 0.0008).
- `Normalization.py` applies `sklearn.preprocessing.normalize` (row L2), not min-max. Cosine similarity is invariant to that.

## 2. QMOF database, Figshare v18 (2025-11-15)

`qmof_database.zip` contains `qmof.csv` (20,372 × 94), `qmof.json`, `qmof_structure_data.json` (3.2 GB, not extracted), CIF zips, and a README. Only `qmof.csv` is used.

| Needed field | Column | Non-null |
|---|---|---|
| MOF ID | `qmof_id` (unique) | 20,372 |
| Name / refcode | `name` (e.g. `ABACUF01_FSR`; the CSD refcode is the prefix before `_FSR`/`_freeONLY`/…) | 20,372 |
| PBE band gap (eV) | `outputs.pbe.bandgap` (mean 2.09, std 1.13, range 0–6.45) | 20,372 |
| HSE06 band gap (eV) | `outputs.hse06.bandgap` (also `hse06_10hf`, `hle17`) | 10,810 |
| MOFid string | `info.mofid.mofid` | 7,462 |
| Linker SMILES (list as string) | `info.mofid.smiles_linkers` | 17,544 |
| Node SMILES (list as string) | `info.mofid.smiles_nodes` | 17,677 |
| Composition | `info.formula` | 20,372 |

**MOFid linker information is present**, so the Phase 2 stop condition ("MOFid absent") does not apply.

Preliminary counts, for planning only (Phase 2 will apply and log the formal filters):

- Distinct linkers per entry: 1 → 10,051; 2 → 5,506; 3 → 1,540; ≥4 → 447.
- Metals per formula, excluding H, B, C, N, O, Si, P, S, halogens, noble gases, Ge, As, Se, Te: 1 → 19,014; 2 → 1,336; 3 → 19; 0 → 3.
- Exactly one distinct linker **and** one metal: **9,268**, of which 5,725 have an HSE06 gap. This is above the Gate 2 threshold (3,000) and the O4 threshold (1,500).
- Sources: CSD 16,040; BoydWoo 1,806; GMOF 1,366; CoRE 823; others.
