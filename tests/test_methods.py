"""Phase 5 tests on a synthetic problem: oracle, snap, objectives, metrics and every search method."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graphsocial import methods, metrics, objectives
from graphsocial.embedding import Snapper
from graphsocial.graph import centrality, communities, topologies
from graphsocial.graph.topologies import Topology
from graphsocial.oracle import AlreadyEvaluated, BudgetedOracle, BudgetExceeded
from graphsocial.store import Problem

ALL = list(methods.REGISTRY) + list(methods.PRESETS)


@pytest.fixture(scope="module")
def prob() -> Problem:
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 8))
    gap = np.abs(X[:, 0] * 1.2 + rng.normal(scale=0.3, size=n)) + 0.1
    df = pd.DataFrame({"pbe_gap": gap, "hse_gap": np.where(rng.random(n) < 0.5, gap * 1.3, np.nan)})
    obj = objectives.make("O2", df)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d2, axis=1)[:, 1:6]
    top = Topology.from_pairs("knn", n, np.c_[np.repeat(np.arange(n), 5), nn.ravel()])
    comm = communities.leiden(top, 1.0, 0)
    cent = {"betweenness": centrality.betweenness(top)[0], "degree": centrality.degree(top),
            "pagerank": centrality.pagerank(top)}
    return Problem(obj, X, top, cent, comm, np.array([f"m{i}" for i in range(n)]), "knn")


def test_oracle_rules():
    o = BudgetedOracle(np.arange(5.0), budget=2)
    o(1)
    with pytest.raises(AlreadyEvaluated):
        o(1)
    o(3)
    with pytest.raises(BudgetExceeded):
        o(4)
    assert o.best() == (1, 1.0)


def test_snap_returns_unevaluated_nearest():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    s = Snapper(X)
    ev = np.array([True, True, False, False])
    assert s.nearest(np.array([0.1]), ev) == 2
    assert s.nearest(np.array([2.9]), ev) == 3


def test_objectives_and_hits():
    df = pd.DataFrame({"pbe_gap": np.linspace(0, 5, 200), "hse_gap": np.nan})
    o3 = objectives.make("O3", df)
    assert o3.f_shifted.min() == 0 and np.allclose(o3.f, np.abs(df.pbe_gap - 2.0))
    closest = np.argsort(np.abs(df.pbe_gap.to_numpy() - 2.0))[:2]
    assert o3.hits.sum() == 2 and set(np.flatnonzero(o3.hits)) == set(closest)
    o1 = objectives.make("O1", df)
    assert o1.hits.sum() == 2 and df.pbe_gap[o1.hits].min() >= 4.9


def test_metrics_perfect_trace():
    df = pd.DataFrame({"pbe_gap": np.arange(100.0), "hse_gap": np.nan})
    o = objectives.make("O2", df)
    s = metrics.summarize(np.array([0, 5, 6]), o, np.zeros(100, dtype=int))
    assert s["final_recall"] == 1.0 and s["first_hit_eval"] == 1 and s["simple_regret"] == 0.0


@pytest.mark.parametrize("name", ALL)
def test_method_invariants(prob, name):
    budget, P, seed = 45, 10, 3
    init = methods.initial_design(prob.n, P, seed)
    o = methods.make(name).run(prob, budget, seed, init)
    assert o.used == budget                                   # budget used exactly, never exceeded
    assert len(set(o.order)) == budget                        # no MOF evaluated twice
    assert o.order[:P] == [int(i) for i in init]              # identical initial design
    o2 = methods.make(name).run(prob, budget, seed, init)
    assert o.order == o2.order and o.values == o2.values      # determinism


def test_initial_design_shared_and_seeded():
    a = methods.initial_design(500, 10, 1)
    assert np.array_equal(a, methods.initial_design(500, 10, 1))
    assert not np.array_equal(a, methods.initial_design(500, 10, 2))


def test_graph_social_variants(prob):
    init = methods.initial_design(prob.n, 5, 0)
    for params in ({"neighbor_weights": "uniform"}, {"centrality": "pagerank"}, {"mutation": "off"},
                   {"mutation": "uniform"}, {"sync": False}, {"elite": False}, {"h": 1}):
        o = methods.make("graph_social", **params).run(prob, 40, 0, init)
        assert o.used == 40 and len(set(o.order)) == 40


def test_topologies(prob):
    base = prob.topology
    deg = np.sort(base.degree())
    dp = topologies.degree_preserving(base, 0)
    assert np.array_equal(np.sort(dp.degree()), deg) and dp.m == base.m
    lr = topologies.add_long_range(base, 0.1, prob.communities, 0)
    assert lr.m == base.m + round(0.1 * base.m)
    ws = topologies.watts_strogatz(base.n, base.degree().mean(), 0)
    assert abs(ws.degree().mean() - base.degree().mean()) <= 1.0
    assert prob.topology.within_hops(0, 1) == set(map(int, base.neighbors(0)))


def test_expected_improvement_sign():
    from graphsocial.methods.gp_ei import expected_improvement

    # Minimisation: a candidate predicted below the incumbent must have higher EI than one above it.
    ei = expected_improvement(np.array([0.0, 2.0]), np.array([0.5, 0.5]), best=1.0, xi=0.01)
    assert ei[0] > ei[1] > 0
    # More uncertainty at equal mean gives more EI.
    ei2 = expected_improvement(np.array([1.5, 1.5]), np.array([0.1, 1.0]), best=1.0)
    assert ei2[1] > ei2[0]
