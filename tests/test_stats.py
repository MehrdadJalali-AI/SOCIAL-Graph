"""Statistics tests: Friedman mean ranks over blocks and completeness guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graphsocial import stats as S


def _blocks(n_blocks: int = 16, k: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.linspace(0.1, 0.5, k)  # true method quality
    data = base[None, :] + rng.normal(scale=0.08, size=(n_blocks, k))
    return pd.DataFrame(data, columns=[f"m{i}" for i in range(k)])


def test_mean_ranks_sum_and_not_integer():
    chi2, p, ranks, nem = S.friedman_nemenyi(_blocks(), expected_blocks=16, expected_methods=10)
    assert ranks.sum() == pytest.approx(55.0)          # k(k+1)/2 for k = 10
    assert not np.allclose(ranks, np.round(ranks))     # averaging over 16 noisy blocks gives non-integer ranks
    assert ranks.min() >= 1 and ranks.max() <= 10
    assert ranks.idxmin() == "m9"                       # best method (highest score) has the lowest rank
    assert p < 0.05 and nem.shape == (10, 10)


def test_ranks_equal_manual_average():
    x = _blocks(5, 4, seed=1)
    _, _, ranks, _ = S.friedman_nemenyi(x)
    manual = np.mean([pd.Series(-row).rank(method="average").to_numpy() for row in x.to_numpy()], axis=0)
    assert np.allclose(ranks.to_numpy(), manual)


def test_missing_cells_raise_instead_of_dropping_blocks():
    x = _blocks()
    x.iloc[3, 2] = np.nan
    with pytest.raises(S.IncompleteBlocks):
        S.friedman_nemenyi(x)
    with pytest.raises(S.IncompleteBlocks):
        S.friedman_nemenyi(_blocks(15), expected_blocks=16)


def test_ties_give_average_ranks():
    x = pd.DataFrame({"a": [1.0, 1.0], "b": [1.0, 1.0], "c": [0.0, 0.0]})
    _, _, ranks, _ = S.friedman_nemenyi(x)
    assert ranks["a"] == ranks["b"] == 1.5 and ranks["c"] == 3.0  # integer only because genuinely tied
