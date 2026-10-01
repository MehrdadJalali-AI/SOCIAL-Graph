"""Run plans for Phases 6 and 7 (shared with Phase 8, which reloads the same specs).

Variant labels are canonical: ``topo=<graph name>`` optionally followed by ``|<ablation>``. Together
with the method, objective, budget, P, community phi and seed they fully determine a run.
"""

from __future__ import annotations

import pandas as pd

from .experiment import RunSpec
from .store import Store, graph_name


def seeds(cfg: dict, which: str) -> list[int]:
    lo, hi = cfg[which]
    return list(range(lo, hi + 1))


def phi_star(cfg: dict) -> float:
    return float(Store(cfg).read_choices().get("phi_star", cfg["phase3"]["phis"][-1]))


def best_topology(cfg: dict) -> str:
    return Store(cfg).read_choices().get("best_topology", graph_name(phi_star(cfg)))


def gs_spec(group: str, topo: str, obj: str, frac: float, seed: int, comm_phi: float, P: int = 10,
            params: dict | None = None, tag: str = "", embedding: str = "default") -> RunSpec:
    variant = f"topo={topo}" + (f"|{tag}" if tag else "")
    return RunSpec(group, "graph_social", variant, obj, frac, seed, topo, comm_phi, P, params or {}, embedding)


# --- Phase 6 ---------------------------------------------------------------------------------------
def pilot_variants(cfg: dict, phi: float) -> dict[str, str]:
    """label -> topology graph name."""
    return {
        "MOFGalaxyNet": graph_name(phi),
        f"MOFGalaxyNet+ρ={cfg['phase6']['rho']:.2f}": graph_name(phi, "rho", cfg["phase6"]["rho"]),
        "degree-preserving random": graph_name(phi, "degrand"),
        "Watts–Strogatz": graph_name(phi, "ws"),
    }


def phase6_specs(cfg: dict) -> list[tuple[str, RunSpec]]:
    phi = phi_star(cfg)
    p6 = cfg["phase6"]
    out = []
    for s in seeds(cfg, "seeds_pilot"):
        for label, topo in pilot_variants(cfg, phi).items():
            out.append((label, gs_spec("phase6", topo, p6["objective"], p6["budget_frac"], s, phi)))
        out.append(("random search", RunSpec("phase6", "random", f"topo={graph_name(phi)}", p6["objective"],
                                              p6["budget_frac"], s, graph_name(phi), phi)))
    return out


def sensitivity_specs(cfg: dict, objective: str, frac: float, seed_list: list[int], group: str) -> list[RunSpec]:
    """phi x rho grid (topology (a)/(b)) for Graph-SOCIAL; communities fixed at phi*."""
    ps = phi_star(cfg)
    out = []
    for phi in cfg["phase3"]["phis"]:
        for rho in [0.0, *cfg["phase3"]["rhos"]]:
            topo = graph_name(phi, "rho", rho)
            out += [gs_spec(group, topo, objective, frac, s, ps) for s in seed_list]
    return out


# --- Phase 7 ---------------------------------------------------------------------------------------
def objectives_to_run(cfg: dict) -> list[str]:
    df = Store(cfg).load_clean()
    objs = list(cfg["phase7"]["objectives"])
    if "O4" in objs and df["hse_gap"].notna().sum() < cfg["objectives"]["o4_min_mofs"]:
        objs.remove("O4")
    return objs


def main_specs(cfg: dict) -> list[RunSpec]:
    ps, best, base = phi_star(cfg), best_topology(cfg), graph_name(phi_star(cfg))
    P = cfg["experiments"]["default_P"]
    out = []
    for obj in objectives_to_run(cfg):
        for frac in cfg["phase7"]["budgets"]:
            for s in seeds(cfg, "seeds_full"):
                for m in cfg["phase7"]["methods"]:
                    if m == "graph_social":
                        out.append(gs_spec("phase7_main", best, obj, frac, s, ps, P))
                    else:
                        out.append(RunSpec("phase7_main", m, f"topo={base}", obj, frac, s, base, ps, P))
    return out


def ablation_table(cfg: dict) -> list[dict]:
    """Every ablation configuration: label, topology, params, P, tag, embedding."""
    ps, best = phi_star(cfg), best_topology(cfg)
    P = cfg["experiments"]["default_P"]

    def row(label, topo=best, params=None, p=P, tag="", embedding="default"):
        return {"label": label, "topo": topo, "params": params or {}, "P": p, "tag": tag, "embedding": embedding}

    rows = [
        row("default"),
        row("uniform neighbour weights", params={"neighbor_weights": "uniform"}, tag="uniform_weights"),
        row("no neighbour term (α=β=0)", params={"alpha": 0.0, "beta": 0.0}, tag="no_neighbor"),
        row("centrality = degree", params={"centrality": "degree"}, tag="cent=degree"),
        row("centrality = PageRank", params={"centrality": "pagerank"}, tag="cent=pagerank"),
        row("mutation off", params={"mutation": "off"}, tag="mutation=off"),
        row("uniform mutation", params={"mutation": "uniform"}, tag="mutation=uniform"),
        row("sync off", params={"sync": False}, tag="sync=off"),
        row("elite off", params={"elite": False}, tag="elite=off"),
        row("P = 5", p=5),
        row("P = 20", p=20),
        row("h = 1", params={"h": 1}, tag="h=1"),
    ]
    topo_rows = [("(a) MOFGalaxyNet", graph_name(ps)),
                 *[(f"(b) +ρ={r:.2f}", graph_name(ps, "rho", r)) for r in cfg["phase3"]["rhos"]],
                 ("(c) degree-preserving random", graph_name(ps, "degrand")),
                 ("(d) Watts–Strogatz", graph_name(ps, "ws"))]
    rows += [row(f"topology {lab}", topo=t) for lab, t in topo_rows]
    rows += [row(f"φ = {phi}", topo=graph_name(phi)) for phi in cfg["phase3"]["phis"]]
    # Decoupled search space: geometric descriptors only, neighbourhoods from (a) or (c).
    rows += [row("decoupled embedding, topology (a) MOFGalaxyNet", topo=graph_name(ps), embedding="geometric"),
             row("decoupled embedding, topology (c) degree-preserving random", topo=graph_name(ps, "degrand"),
                 embedding="geometric")]
    return rows


def ablation_specs(cfg: dict) -> list[tuple[str, RunSpec]]:
    ps = phi_star(cfg)
    out = []
    for obj in cfg["phase7"]["ablation_objectives"]:
        for r in ablation_table(cfg):
            for s in seeds(cfg, "seeds_full"):
                out.append((r["label"], gs_spec("phase7_ablation", r["topo"], obj, cfg["phase7"]["ablation_budget"],
                                                s, ps, r["P"], r["params"], r["tag"], r["embedding"])))
    return out


def phase7_sensitivity_specs(cfg: dict) -> list[RunSpec]:
    out = []
    for obj in cfg["phase7"]["ablation_objectives"]:
        out += sensitivity_specs(cfg, obj, cfg["phase7"]["ablation_budget"], seeds(cfg, "seeds_full"),
                                 "phase7_sensitivity")
    return out


def labelled_frame(pairs: list[tuple[str, RunSpec]], results: pd.DataFrame) -> pd.DataFrame:
    """Attach labels to results; a run shared by several labels (e.g. 'default' and 'topology (a)') appears
    once per label."""
    lab = pd.DataFrame({"key": [s.key for _, s in pairs], "label": [lb for lb, _ in pairs]}).drop_duplicates()
    return results.drop_duplicates("key").merge(lab, on="key")
