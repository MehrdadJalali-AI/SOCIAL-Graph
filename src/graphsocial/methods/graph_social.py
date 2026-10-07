"""Graph-SOCIAL: the SOCIAL optimiser (Jalali et al., ASOC 2026, Eqs. 8-17, Algorithm 1) as an acquisition policy.

Agents live in the 32-d embedding. After each position update an agent snaps to the nearest unevaluated
MOF, which the oracle evaluates. Two agents are neighbours if their current MOFs are within ``h`` hops in
the search topology; the centrality c_j of a neighbour is the precomputed (max-normalised) centrality of
its current MOF. With ``agent_topology="ws"`` the agents instead communicate over a Watts-Strogatz graph
among themselves (the original SOCIAL; used as baseline 6).

Per iteration t (T = floor(budget / P) iterations including the initial design):
    alpha_t = alpha (1 - t/T), beta_t = beta (1 - t/T), gamma_t = gamma t/T, delta_t = delta t/T   (8-11)
    I_j = 1 - log(1 + |f'_j|) / (log(1 + max f') + 1e-6)          (influence, on shifted objective f')
    f' = f - (minimum objective value among the MOFs evaluated so far); only evaluated values are used (D22)
    w_ij = (alpha_t c_j + beta_t I_j) / sum_k (alpha_t c_k + beta_t I_k)                         (12)
    x_i <- (1 - a - b - g - d) x_i + (a + b) x_neigh + g x_gbest + d x_elite                      (13)
    x_i <- (1 - omega) x_i + omega x_mean                                                     (14-15)
    x_i <- clip(x_i, x_min, x_max)                                                               (16)
    worse-than-median agents mutate with probability p_m; every 10 iterations each agent receives
    U(-0.5, 0.5)^D noise with probability 0.05 (Algorithm 1, lines 33-39).
Leftover evaluations (budget - T*P) go to the unevaluated MOFs nearest to the elite (17).
"""

from __future__ import annotations


import igraph as ig
import numpy as np

from ..embedding import Snapper
from ..oracle import BudgetedOracle
from ..store import Problem
from .base import Method, evaluate_initial, method_rng, spend_leftover

DEFAULTS = dict(
    alpha=0.4, beta=0.4, gamma=0.2, delta=0.2, p_m=0.1, s_base=0.1, s_min=0.01, omega=0.05,
    h=2, centrality="betweenness", neighbor_weights="social", mutation="community",
    sync=True, elite=True, agent_topology="mof", ws_k=4, ws_p=0.3,
    periodic_every=10, periodic_prob=0.05, periodic_scale=0.5,
)


class GraphSOCIAL(Method):
    name = "graph_social"
    population = True

    def __init__(self, **params):
        super().__init__(**{**DEFAULTS, **params})

    # -- helpers -------------------------------------------------------------------------------
    def _influence(self, fs: np.ndarray) -> np.ndarray:
        fmax = np.max(fs)
        return 1.0 - np.log1p(np.abs(fs)) / (np.log1p(fmax) + 1e-6)

    def _agent_ws(self, P: int, rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
        import random

        ig.set_random_number_generator(random.Random(int(rng.integers(2**31))))
        k = min(self.params["ws_k"], P - 1)
        g = ig.Graph.Watts_Strogatz(dim=1, size=P, nei=max(1, k // 2), p=self.params["ws_p"])
        bc = np.asarray(g.betweenness(), dtype=float)
        bc = bc / bc.max() if bc.max() > 0 else bc
        return [np.asarray(g.neighbors(i)) for i in range(P)], bc

    def _mof_neighbors(self, prob: Problem, cur: np.ndarray) -> list[np.ndarray]:
        pos = {int(m): a for a, m in enumerate(cur)}
        out = []
        for a, m in enumerate(cur):
            reach = prob.topology.within_hops(int(m), self.params["h"])
            out.append(np.array(sorted(pos[r] for r in reach if r in pos), dtype=np.int64))
        return out

    def _mutation_target(self, prob: Problem, oracle: BudgetedOracle, rng: np.random.Generator) -> int:
        free = ~oracle.evaluated
        if self.params["mutation"] == "community":
            comm = prob.communities
            k = comm.max() + 1
            evaluated_per = np.bincount(comm[~free], minlength=k)
            has_free = np.bincount(comm[free], minlength=k) > 0
            counts = np.where(has_free, evaluated_per, np.iinfo(np.int64).max)
            lowest = np.flatnonzero(counts == counts.min())
            target_comm = rng.choice(lowest)
            cand = np.flatnonzero(free & (comm == target_comm))
        else:
            cand = np.flatnonzero(free)
        return int(rng.choice(cand))

    # -- main loop -----------------------------------------------------------------------------
    def run(self, prob: Problem, budget: int, seed: int, init: np.ndarray) -> BudgetedOracle:
        p = self.params
        rng = method_rng(seed, self.name)
        oracle = BudgetedOracle(prob.objective.f, budget)
        snap = Snapper(prob.X)
        lo, hi = snap.bounds()
        span = hi - lo
        P = len(init)
        T = budget // P
        cent_all = prob.cent[p["centrality"]] if p["agent_topology"] == "mof" else None
        ws_nbrs, ws_cent = self._agent_ws(P, rng) if p["agent_topology"] == "ws" else (None, None)

        cur = init.astype(np.int64).copy()
        f = evaluate_initial(oracle, cur)
        x = prob.X[cur].copy()
        elite_x, elite_f = x[np.argmin(f)].copy(), f.min()

        for t in range(1, T):
            frac = t / T
            a_t, b_t = p["alpha"] * (1 - frac), p["beta"] * (1 - frac)
            g_t = p["gamma"] * frac
            d_t = p["delta"] * frac if p["elite"] else 0.0
            gbest_x = x[np.argmin(f)].copy()
            x_mean = x.mean(axis=0)
            fs = f - min(oracle.values)  # shift from evaluated MOFs only, never from the unevaluated pool
            infl = self._influence(fs)
            if p["agent_topology"] == "mof":
                nbrs = self._mof_neighbors(prob, cur)
                c_agents = cent_all[cur]
            else:
                nbrs, c_agents = ws_nbrs, ws_cent
            median = np.median(f)
            new_x = np.empty_like(x)
            top_nb = np.full(P, -1, dtype=np.int64)
            for i in range(P):
                nb = nbrs[i]
                if len(nb):
                    if p["neighbor_weights"] == "uniform":
                        w = np.full(len(nb), 1.0 / len(nb))
                    else:
                        raw = a_t * c_agents[nb] + b_t * infl[nb]
                        w = raw / raw.sum() if raw.sum() > 0 else np.full(len(nb), 1.0 / len(nb))
                    x_neigh = w @ x[nb]
                    top_nb[i] = nb[int(np.argmax(w))]
                    xi = ((1 - a_t - b_t - g_t - d_t) * x[i] + (a_t + b_t) * x_neigh
                          + g_t * gbest_x + d_t * elite_x)
                else:
                    xi = (1 - g_t - d_t) * x[i] + g_t * gbest_x + d_t * elite_x
                if p["sync"]:
                    xi = (1 - p["omega"]) * xi + p["omega"] * x_mean
                new_x[i] = np.clip(xi, lo, hi)

            # Mutation and periodic perturbation, then snap + evaluate (agents in index order).
            new_cur = cur.copy()
            new_f = f.copy()
            for i in range(P):
                kind = "update"
                jump = None
                if p["mutation"] != "off" and f[i] > median and rng.random() < p["p_m"]:
                    kind = "mutation"
                    if p["mutation"] == "perturb":
                        s_t = p["s_base"] * (1 - frac) + p["s_min"]
                        new_x[i] = np.clip(new_x[i] + rng.uniform(-s_t * span, s_t * span), lo, hi)
                    else:
                        jump = self._mutation_target(prob, oracle, rng)
                if p["periodic_every"] and t % p["periodic_every"] == 0 and rng.random() < p["periodic_prob"]:
                    s = p["periodic_scale"]
                    new_x[i] = np.clip(new_x[i] + rng.uniform(-s, s, size=new_x.shape[1]), lo, hi)
                if oracle.exhausted():
                    break
                m = jump if jump is not None else snap.nearest(new_x[i], oracle.evaluated)
                nb_agent = int(top_nb[i])
                nb_mof = int(cur[nb_agent]) if nb_agent >= 0 else -1
                nb_c = float(c_agents[nb_agent]) if nb_agent >= 0 else np.nan
                new_f[i] = oracle(m, step=t, agent=i, kind=kind, top_nb_mof=nb_mof, top_nb_cent=nb_c)
                new_cur[i] = m
            cur, f = new_cur, new_f
            x = prob.X[cur].copy()
            if f.min() < elite_f:
                elite_f, elite_x = f.min(), x[np.argmin(f)].copy()

        anchor = elite_x if p["elite"] else x[np.argmin(f)]
        spend_leftover(oracle, snap, anchor, T)
        return oracle
