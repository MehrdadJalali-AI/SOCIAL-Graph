"""Statistical tests: paired Wilcoxon, Holm correction, Cliff's delta, Friedman + Nemenyi."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> float:
    """Paired Wilcoxon signed-rank p-value (zero differences dropped); 1.0 if all differences are zero."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, zero_method="wilcox", alternative=alternative).pvalue)
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


def friedman_nemenyi(block_scores: pd.DataFrame) -> tuple[float, float, pd.Series, pd.DataFrame]:
    """``block_scores``: rows = blocks, columns = methods, higher = better.

    Returns (chi2, p, mean ranks (1 = best), Nemenyi p-value matrix).
    """
    import scikit_posthocs as sp

    x = block_scores.dropna(axis=0)
    chi2, p = stats.friedmanchisquare(*[x[c].to_numpy() for c in x.columns])
    ranks = x.rank(axis=1, ascending=False).mean(axis=0)
    nem = sp.posthoc_nemenyi_friedman(x.to_numpy())
    nem.index = nem.columns = x.columns
    return float(chi2), float(p), ranks, nem


def mean_std(x: pd.Series, fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(x.mean())} ± {fmt.format(x.std(ddof=1) if len(x) > 1 else 0.0)}"
