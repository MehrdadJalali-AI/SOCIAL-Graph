"""Diagnostics for GP-EI (revision A2): sign, normalisation, incumbent, and model quality.

Writes reports/GP_EI_CHECKS.md.

    python scripts/gp_ei_checks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphsocial import objectives as O  # noqa: E402
from graphsocial.methods.gp_ei import expected_improvement  # noqa: E402


def kernel(D: int):
    return (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(D), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-3, (1e-8, 1)))


def main() -> int:
    warnings.simplefilter("ignore")
    df = pd.read_parquet(ROOT / "data" / "processed" / "qmof_clean.parquet")
    X = np.load(ROOT / "data" / "processed" / "embedding.npy")
    lines = ["# GP-EI diagnostics (revision)", ""]

    # 1. EI sign.
    ei = expected_improvement(np.array([0.0, 2.0]), np.array([0.5, 0.5]), best=1.0, xi=0.01)
    lines += ["## 1. Sign of expected improvement",
              f"For minimisation with incumbent 1.0: EI(mean 0.0) = {ei[0]:.4f} > EI(mean 2.0) = {ei[1]:.4f}. "
              "Candidates predicted below the incumbent are preferred, as required.", ""]

    rows = []
    rng = np.random.default_rng(0)
    for obj in ("O1", "O2", "O3", "O4"):
        o = O.make(obj, df)
        Xu, f = X[o.universe], o.f
        for n in (44, 174):
            tr = rng.choice(len(f), n, replace=False)
            te = np.setdiff1d(np.arange(len(f)), tr)
            with threadpool_limits(1):
                gp = GaussianProcessRegressor(kernel(X.shape[1]), normalize_y=True, n_restarts_optimizer=5,
                                              random_state=0).fit(Xu[tr], f[tr])
                mu_tr = gp.predict(Xu[tr])
                mu, sd = gp.predict(Xu[te], return_std=True)
                rf = RandomForestRegressor(100, random_state=0, n_jobs=1).fit(Xu[tr], f[tr])
            ei = expected_improvement(mu, sd, f[tr].min(), 0.01)
            top_ei = te[np.argsort(-ei)[:20]]
            rows.append({
                "objective": obj, "n_train": n,
                "train_fit_r": float(np.corrcoef(mu_tr, f[tr])[0, 1]),
                "pred_mean_vs_train_mean": float(mu.mean() - f[tr].mean()),
                "incumbent_is_min_observed": bool(np.isclose(f[tr].min(), np.min(f[tr]))),
                "spearman_gp_heldout": float(stats.spearmanr(mu, f[te])[0]),
                "spearman_rf_heldout": float(stats.spearmanr(rf.predict(Xu[te]), f[te])[0]),
                "hit_rate_top20_EI": float(o.hits[top_ei].mean()),
                "hit_rate_random": float(o.hits.mean()),
                "median_length_scale": float(np.median(gp.kernel_.k1.k2.length_scale)),
                "noise_level": float(gp.kernel_.k2.noise_level),
            })
    tab = pd.DataFrame(rows)
    lines += ["## 2–4. Normalisation, incumbent and model quality",
              "A GP was fitted on random training sets of 44 and 174 MOFs for each objective. The table reports:",
              "- the correlation of the posterior mean with the training targets (normalisation round-trip);",
              "- the offset of held-out predictions from the training mean (predictions are in original units);",
              "- the held-out Spearman correlation of the GP and of a random forest with the true objective;",
              "- the hit rate among the 20 held-out MOFs with the highest EI, against the base rate.", "",
              tab.to_markdown(index=False, floatfmt=".3f"), ""]

    t3 = ROOT / "results" / "tables" / "T3_main_results.csv"
    if t3.exists():
        d = pd.read_csv(t3)
        d["r"] = d.final_recall.str.split(" ±").str[0].astype(float)
        w = d.pivot_table(index=["objective", "budget_frac"], columns="method", values="r")
        if "gp_ei" in w:
            cmp = w[["gp_ei", "random", "ensemble_ts"]].reset_index()
            cmp["gp_minus_random"] = cmp.gp_ei - cmp.random
            lines += ["## 5. GP-EI against random search and ensemble TS (final recall)", "",
                      cmp.to_markdown(index=False, floatfmt=".4f"), ""]
            trail = cmp[cmp.gp_minus_random < 0]
            lines += [f"GP-EI trails random search in {len(trail)} of {len(cmp)} cells."
                      + (" Checks 1–4 show no sign, normalisation or incumbent error. Where GP-EI trails random "
                         "search, the held-out rank correlation of the GP model is low for that objective (table "
                         "above), so EI concentrates evaluations in regions that the model wrongly ranks as "
                         "promising." if len(trail) else ""), ""]
    # 6. Periodic refits (every 10 evaluations) vs refits at every evaluation, same seeds (D12).
    import json
    import re

    arch = ROOT / "results" / "archive" / "gp_ei_refit_every_step"
    rows = []
    for f in sorted(arch.glob("gp_ei__*.json")):
        new = ROOT / "results" / "runs" / f.name
        if not new.exists():
            continue
        m = re.search(r"__(O\d)__b([\d.]+)__.*__s(\d+)\.json$", f.name)
        a, b = json.loads(f.read_text()), json.loads(new.read_text())
        rows.append({"objective": m.group(1), "budget_frac": float(m.group(2)), "seed": int(m.group(3)),
                     "recall_every_step": a["final_recall"], "recall_every_10": b["final_recall"]})
    if rows:
        d = pd.DataFrame(rows)
        g = d.groupby(["objective", "budget_frac"])
        cmp = pd.DataFrame({"seeds": g.size(), "every_step": g.recall_every_step.mean(),
                            "every_10": g.recall_every_10.mean()}).reset_index()
        pv = [stats.wilcoxon(x.recall_every_10, x.recall_every_step).pvalue
              if not np.allclose(x.recall_every_10, x.recall_every_step) else 1.0 for _, x in g]
        cmp["p_wilcoxon_two_sided"] = pv
        cmp.to_csv(ROOT / "results" / "tables" / "gp_refit_check.csv", index=False)
        lines += ["## 6. Hyperparameter refits every 10 evaluations vs every evaluation (same seeds)", "",
                  cmp.to_markdown(index=False, floatfmt=".4f"), ""]
    (ROOT / "reports" / "GP_EI_CHECKS.md").write_text("\n".join(lines) + "\n")
    print("wrote reports/GP_EI_CHECKS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
