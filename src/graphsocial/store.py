"""Locations of cached artifacts and the ``Problem`` bundle handed to every search method.

A Problem is everything a method may use: the embedding, a search topology with precomputed
centralities, the reference Leiden communities, and the objective (which only the oracle reads).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import objectives
from .graph import centrality
from .graph.topologies import Topology


def fmt_phi(phi: float) -> str:
    return f"{phi:.2f}"


def graph_name(phi: float, variant: str = "mgn", rho: float = 0.0) -> str:
    """variant: mgn | rho | degrand | ws | rdkit."""
    p = fmt_phi(phi)
    if variant == "rho" and rho == 0:
        variant = "mgn"
    return {
        "mgn": f"mgn_phi{p}",
        "rho": f"mgn_phi{p}_rho{rho:.2f}",
        "degrand": f"mgn_phi{p}_degrand",
        "ws": f"ws_phi{p}",
        "rdkit": f"rdkit_phi{p}",
    }[variant]


class Store:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = C.PROJECT_ROOT / cfg["paths"]["processed"]
        (self.root / "graphs").mkdir(parents=True, exist_ok=True)

    # --- files -----------------------------------------------------------------------------
    @property
    def clean(self) -> Path:
        return self.root / "qmof_clean.parquet"

    @property
    def descriptors(self) -> Path:
        return self.root / "descriptors.npz"

    @property
    def embedding(self) -> Path:
        return self.root / "embedding.npy"

    @property
    def embedding_geometric(self) -> Path:
        return self.root / "embedding_geometric.npy"

    def ensure_geometric_embedding(self) -> Path:
        """Build (once) the decoupled geometric embedding aligned with the clean table."""
        if not self.embedding_geometric.exists():
            from .embedding import GEOMETRIC_COLUMNS, build_geometric_embedding

            raw = pd.read_csv(C.resolve(self.cfg["phase2"]["qmof_csv"]), usecols=["qmof_id", *GEOMETRIC_COLUMNS])
            geom = self.load_clean()[["qmof_id"]].merge(raw, on="qmof_id", how="left")
            np.save(self.embedding_geometric, build_geometric_embedding(geom))
        return self.embedding_geometric

    def graph(self, name: str) -> Path:
        return self.root / "graphs" / f"{name}.npz"

    def cent(self, name: str, universe: str = "all") -> Path:
        return self.root / "graphs" / f"{name}__{universe}__centrality.npz"

    def communities(self, phi: float) -> Path:
        return self.root / "graphs" / f"communities_phi{fmt_phi(phi)}.npy"

    @property
    def choices(self) -> Path:
        return self.root / "choices.json"

    # --- loaders ---------------------------------------------------------------------------
    def load_clean(self) -> pd.DataFrame:
        return pd.read_parquet(self.clean)

    def load_descriptors(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.load(self.descriptors)
        bits = np.unpackbits(z["bits"], axis=1)[:, : int(z["n_bits"])]
        return bits, z["metal_vec"]

    def read_choices(self) -> dict:
        return json.loads(self.choices.read_text()) if self.choices.exists() else {}

    def write_choice(self, key: str, value) -> None:
        ch = self.read_choices()
        ch[key] = value
        self.choices.write_text(json.dumps(ch, indent=2))

    def centralities(self, name: str, universe_name: str = "all", nodes: np.ndarray | None = None) -> dict:
        path = self.cent(name, universe_name)
        if path.exists():
            z = np.load(path)
            return {k: z[k] for k in z.files}
        top = Topology.load(self.graph(name))
        if nodes is not None:
            top = top.subgraph(nodes)
        c3 = self.cfg["phase3"]
        res = centrality.all_centralities(top, c3["betweenness_exact_max_n"], c3["betweenness_k"], c3["seed"])
        res.pop("_betweenness_method")
        np.savez_compressed(path, **res)
        return res


@dataclass
class Problem:
    """A search problem restricted to an objective's universe (all MOFs, or the HSE subset)."""

    objective: objectives.Objective
    X: np.ndarray                  # embedding rows of the universe
    topology: Topology             # search topology on the universe
    cent: dict[str, np.ndarray]    # max-normalised centralities on the universe
    communities: np.ndarray        # reference Leiden communities restricted to the universe
    mof_ids: np.ndarray            # qmof_id of each universe node
    topology_name: str

    @property
    def n(self) -> int:
        return len(self.X)


@functools.lru_cache(maxsize=64)
def _cached_problem(cfg_key: str, objective: str, topology_name: str, comm_phi: float,
                    embedding: str = "default") -> Problem:
    cfg = json.loads(cfg_key)
    st = Store(cfg)
    df = _cached_df(str(st.clean))
    obj = objectives.make(objective, df, cfg["objectives"]["hit_fraction"])
    emb_path = st.embedding if embedding == "default" else st.ensure_geometric_embedding()
    X = _cached_embedding(str(emb_path))[obj.universe]
    top = Topology.load(st.graph(topology_name))
    full = len(obj.universe) == len(df)
    uname = "all" if full else objective
    if not full:
        top = top.subgraph(obj.universe)
    cent = st.centralities(topology_name, uname, None if full else obj.universe)
    comm = np.load(st.communities(comm_phi))[obj.universe]
    return Problem(obj, X, top, cent, comm, df["qmof_id"].to_numpy()[obj.universe], topology_name)


@functools.lru_cache(maxsize=4)
def _cached_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@functools.lru_cache(maxsize=4)
def _cached_embedding(path: str) -> np.ndarray:
    return np.load(path)


def problem(cfg: dict, objective: str, topology_name: str, comm_phi: float, embedding: str = "default") -> Problem:
    """``embedding``: "default" (linker + metal PCA-32) or "geometric" (decoupled, no linker/metal info)."""
    key = json.dumps({k: cfg[k] for k in ("paths", "phase2", "phase3", "objectives")}, sort_keys=True)
    return _cached_problem(key, objective, topology_name, float(comm_phi), embedding)
