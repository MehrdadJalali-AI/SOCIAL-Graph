"""Search methods. ``make(name, **params)`` returns a configured method instance."""

from __future__ import annotations

from .base import Method, initial_design
from .cmaes_snap import CMAESSnap
from .de_snap import DESnap
from .ensemble_ts import EnsembleTS
from .ga_snap import GASnap
from .gp_ei import GPEI
from .graph_social import GraphSOCIAL
from .greedy_walk import GreedyWalk
from .pso_snap import PSOSnap
from .random_search import RandomSearch
from .static_diverse import StaticDiverse

REGISTRY: dict[str, type[Method]] = {
    "graph_social": GraphSOCIAL,
    "random": RandomSearch,
    "greedy_walk": GreedyWalk,
    "gp_ei": GPEI,
    "ensemble_ts": EnsembleTS,
    "de": DESnap,
    "pso": PSOSnap,
    "ga": GASnap,
    "static_diverse": StaticDiverse,
    "cmaes": CMAESSnap,
}

# SOCIAL with a Watts-Strogatz agent topology: the original algorithm plus the snap operator.
PRESETS: dict[str, tuple[str, dict]] = {
    "social_ws": ("graph_social", {"agent_topology": "ws", "mutation": "perturb"}),
}


def make(name: str, **params) -> Method:
    if name in PRESETS:
        base, preset = PRESETS[name]
        m = REGISTRY[base](**{**preset, **params})
        m.name = name
        return m
    return REGISTRY[name](**params)


__all__ = ["make", "initial_design", "REGISTRY", "PRESETS", "Method"]
