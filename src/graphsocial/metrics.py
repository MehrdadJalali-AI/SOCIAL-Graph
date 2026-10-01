"""Run-level metrics computed from an evaluation trace (indices into the objective's universe)."""

from __future__ import annotations

import numpy as np

from .objectives import Objective


def recall_curve(order: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """Fraction of the hit set found after 1..B evaluations."""
    return np.cumsum(hits[order]) / max(1, hits.sum())


def summarize(order: np.ndarray, obj: Objective, communities: np.ndarray) -> dict:
    order = np.asarray(order, dtype=np.int64)
    B, N = len(order), obj.n
    is_hit = obj.hits[order]
    curve = recall_curve(order, obj.hits)
    found = int(is_hit.sum())
    hit_comms = np.unique(communities[obj.hits])
    found_comms = np.unique(communities[order[is_hit]])
    first = int(np.argmax(is_hit)) + 1 if found else None
    return {
        "budget": B,
        "n_universe": N,
        "n_hits_total": int(obj.hits.sum()),
        "hits_found": found,
        "final_recall": float(curve[-1]),
        "recall_auc": float(curve.mean()),
        "enrichment_factor": float((found / B) / (obj.hits.sum() / N)),
        "simple_regret": float(obj.f[order].min() - obj.f_best),
        "family_coverage": float(len(found_comms) / max(1, len(hit_comms))),
        "families_with_hits": int(len(hit_comms)),
        "families_found": int(len(found_comms)),
        "first_hit_eval": first,
        "recall_curve": curve.round(6).tolist(),
    }


def new_family_hits(order: np.ndarray, hits: np.ndarray, communities: np.ndarray) -> np.ndarray:
    """Boolean per evaluation: a hit in a community where no hit had been found earlier in the run."""
    seen: set[int] = set()
    out = np.zeros(len(order), dtype=bool)
    for k, i in enumerate(order):
        if hits[i]:
            c = int(communities[i])
            if c not in seen:
                out[k] = True
                seen.add(c)
    return out
