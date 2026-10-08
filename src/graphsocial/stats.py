"""Statistical tests: paired Wilcoxon, Holm correction, Cliff's delta, Friedman + Nemenyi."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> float:
    """Paired Wilcoxon signed-rank p-value (zero differences dropped); 1.0 if all differences are zero."""
    # Recall values are multiples of 1/K; rounding makes tied differences exactly equal regardless of how
    # the values were stored (e.g. after a CSV round-trip), so ranks and p-values are reproducible.
    d = np.round(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), 10)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox", alternative=alternative).pvalue)
    except ValueError:
        return 1.0


def holm(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all pairs."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a[:, None] - b[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


class IncompleteBlocks(ValueError):
    """Raised when some (block, method) cells are missing: ranking must never silently drop blocks."""


def friedman_nemenyi(block_scores: pd.DataFrame, expected_blocks: int | None = None,
                     expected_methods: int | None = None) -> tuple[float, float, pd.Series, pd.DataFrame]:
    """Friedman test and Nemenyi post hoc over blocks (rows) x methods (columns), higher score = better.

    Methods are ranked within each block on the score (rank 1 = best; ties get average ranks) and the ranks are
    averaged over blocks. Any missing cell raises ``IncompleteBlocks``; the expected numbers of blocks and methods
    can be enforced explicitly. Returns (chi2, p, mean ranks, Nemenyi p-value matrix).
    """
    import scikit_posthocs as sp

    x = block_scores
    if x.isna().any().any():
        missing = [(i, c) for i in x.index for c in x.columns if pd.isna(x.loc[i, c])]
        raise IncompleteBlocks(f"{len(missing)} missing (block, method) cells, e.g. {missing[:3]}")
    if expected_blocks is not None and len(x) != expected_blocks:
        raise IncompleteBlocks(f"{len(x)} blocks, expected {expected_blocks}")
    if expected_methods is not None and x.shape[1] != expected_methods:
        raise IncompleteBlocks(f"{x.shape[1]} methods, expected {expected_methods}")
    chi2, p = stats.friedmanchisquare(*[x[c].to_numpy() for c in x.columns])
    ranks = x.rank(axis=1, ascending=False, method="average").mean(axis=0)
    k = x.shape[1]
    assert np.isclose(ranks.sum(), k * (k + 1) / 2), "mean ranks must sum to k(k+1)/2"
    nem = sp.posthoc_nemenyi_friedman(x.to_numpy())
    nem.index = nem.columns = x.columns
    return float(chi2), float(p), ranks, nem


def mean_std(x: pd.Series, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1) if len(x) > 1 else 0.0)}"


def power_paired(diffs: np.ndarray, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> dict:
    """Achieved (post-hoc) power of a one-sided paired test for the observed differences.

    - ``power_t``: analytic paired t-test power at the observed standardised effect d_z = mean/sd.
    - ``power_wilcoxon_boot``: share of bootstrap resamples (n pairs, with replacement) in which the
      one-sided Wilcoxon signed-rank test rejects at ``alpha``.
    - ``n_for_80pct``: pairs needed for 80% t-test power at the observed d_z (NaN if d_z <= 0).
    """
    from statsmodels.stats.power import TTestPower

    d = np.round(np.asarray(diffs, dtype=float), 10)
    n = len(d)
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else np.nan
    tp = TTestPower()
    power_t = float(tp.power(effect_size=dz, nobs=n, alpha=alpha, alternative="larger")) if np.isfinite(dz) else np.nan
    rng = np.random.default_rng(seed)
    rej = 0
    for _ in range(n_boot):
        s = rng.choice(d, size=n, replace=True)
        if not np.allclose(s, 0):
            try:
                rej += stats.wilcoxon(s, zero_method="wilcox", alternative="greater").pvalue < alpha
            except ValueError:
                pass
    n80 = (float(tp.solve_power(effect_size=dz, power=0.8, alpha=alpha, alternative="larger"))
           if np.isfinite(dz) and dz > 0 else np.nan)
    return {"n_pairs": n, "d_z": dz, "power_t": power_t, "power_wilcoxon_boot": rej / n_boot, "n_for_80pct": n80}


def hypergeom_sum_test(total_found: int, n_runs: int, N: int, K: int, B: int) -> tuple[float, float]:
    """Expected total hits and exact two-sided p-value for the sum of ``n_runs`` independent
    hypergeometric(N, K, B) draws (random search without replacement)."""
    pmf1 = stats.hypergeom(N, K, B).pmf(np.arange(min(K, B) + 1))
    pmf = np.array([1.0])
    for _ in range(n_runs):
        pmf = np.convolve(pmf, pmf1)
    obs = pmf[total_found] if total_found < len(pmf) else 0.0
    p = float(min(1.0, pmf[pmf <= obs * (1 + 1e-9)].sum()))
    return float(n_runs * B * K / N), p


def nemenyi_cd(k: int, n_blocks: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference q_alpha * sqrt(k (k + 1) / (6 n)) for k methods ranked over n blocks."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(k * (k + 1) / (6 * n_blocks)))
