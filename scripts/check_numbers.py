"""Check that the reported results tables agree with the raw per-run results.

Independent of any manuscript files. Recomputes from results/runs/*.json:

1. the Friedman mean ranks over all objective-budget blocks of the main benchmark (results/tables/T4b_friedman_ranks.csv),
   including the rank-sum identity k(k+1)/2 and the number of blocks;
2. the mean differences and analytical power of the topology comparisons (results/tables/topology_effect_power.csv).

The raw runs are not in the repository (results/runs/ is git-ignored); produce them with the pipeline first:

    python -m graphsocial.run --stage all
    python scripts/check_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsocial import config as C, plan  # noqa: E402
from graphsocial import stats as S  # noqa: E402


def main() -> int:
    cfg = C.load_config(None)
    runs = ROOT / cfg["paths"]["runs"]
    tables = ROOT / "results" / "tables"
    errs: list[str] = []

    rows, missing = [], []
    for spec in plan.main_specs(cfg):
        f = runs / f"{spec.key}.json"
        if not f.exists():
            missing.append(f.name)
            continue
        rows.append((spec.objective, spec.budget_frac, spec.method, json.loads(f.read_text())["final_recall"]))
    if missing:
        print(f"FAIL: {len(missing)} benchmark run files missing in {runs} (e.g. {missing[0]}); run the pipeline first")
        return 1
    r = pd.DataFrame(rows, columns=["o", "b", "m", "rec"])
    blocks = r.groupby(["o", "b", "m"]).rec.mean().unstack("m")
    k = len(cfg["phase7"]["methods"])
    _, _, ranks, _ = S.friedman_nemenyi(blocks, len(blocks), k)
    if not np.isclose(ranks.sum(), k * (k + 1) / 2) or len(blocks) != 16:
        errs.append(f"Friedman blocks {len(blocks)} / rank sum {ranks.sum():.2f}")
    t4b = pd.read_csv(tables / "T4b_friedman_ranks.csv").set_index("method").mean_rank
    if not np.allclose(ranks.loc[t4b.index], t4b, atol=1e-9):
        errs.append("Friedman ranks in results/tables differ from the recomputation from raw runs")
    else:
        print("Friedman ranks: " + ", ".join(f"{m} {v:.2f}" for m, v in t4b.sort_values().items()))

    from statsmodels.stats.power import TTestPower

    tp = pd.read_csv(tables / "topology_effect_power.csv")
    lab: dict = {}
    for label, spec in plan.ablation_specs(cfg):
        lab.setdefault((label, spec.objective), []).append(spec)
    n = 0
    for row in tp[tp.scope.str.startswith("ablation")].itertuples():
        a = {s.seed: json.loads((runs / f"{s.key}.json").read_text())["final_recall"] for s in lab[(row.A, row.objective)]}
        b = {s.seed: json.loads((runs / f"{s.key}.json").read_text())["final_recall"] for s in lab[(row.B, row.objective)]}
        dd = np.round(np.array([a[s] - b[s] for s in sorted(a)]), 10)
        pw = TTestPower().power(effect_size=dd.mean() / dd.std(ddof=1), nobs=len(dd), alpha=0.05, alternative="larger")
        n += 1
        if not np.isclose(pw, row.power_t, atol=1e-6) or not np.isclose(dd.mean(), row.mean_diff, atol=1e-9):
            errs.append(f"power/diff mismatch for {row.objective} {row.A} vs {row.B}: {pw:.4f} vs {row.power_t:.4f}")
    print(f"topology comparisons recomputed: {n}")

    for e in errs:
        print("FAIL:", e)
    print("all checks passed" if not errs else f"{len(errs)} check(s) failed")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
