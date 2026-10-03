"""Revision v0.4, phase 1: verification checks (no manuscript edits).

1.1 Centroid-only policy with and without the shared 10-MOF random initial design (seeds 0-29).
1.2 Pilot random baseline (O2, 2%, seeds 0-9): recomputed with the benchmark code, compared with the stored runs,
    exact hypergeometric test of the pilot value.
1.4 Iteration accounting of the population methods for every objective and budget.

    python scripts/revision_phase1.py
Outputs in results/revision/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphsocial import config as C  # noqa: E402
from graphsocial import methods, objectives, plan, store  # noqa: E402
from graphsocial import stats as S  # noqa: E402
from graphsocial.experiment import RunSpec, budget_for  # noqa: E402

OUT = ROOT / "results" / "revision"


def centroid_policies(cfg) -> pd.DataFrame:
    st = store.Store(cfg)
    df = st.load_clean()
    X = np.load(st.embedding)
    P = cfg["experiments"]["default_P"]
    rows = []
    for o in plan.objectives_to_run(cfg):
        ob = objectives.make(o, df, cfg["objectives"]["hit_fraction"])
        Xu = X[ob.universe]
        order = np.argsort(np.linalg.norm(Xu - Xu.mean(axis=0), axis=1), kind="stable")
        for b in cfg["phase7"]["budgets"]:
            B = budget_for(b, len(order), cfg["experiments"]["min_budget"])
            no_init = ob.hits[order[:B]].sum() / ob.hits.sum()
            with_init = []
            for seed in plan.seeds(cfg, "seeds_full"):
                init = methods.initial_design(len(order), P, seed)
                taken = np.zeros(len(order), dtype=bool)
                taken[init] = True
                rest = [i for i in order if not taken[i]][: B - P]
                ev = np.concatenate([init, np.asarray(rest, dtype=int)])
                assert len(ev) == B and len(set(ev.tolist())) == B
                with_init.append(ob.hits[ev].sum() / ob.hits.sum())
            rows.append({"objective": o, "budget_frac": b, "budget": B,
                         "centroid_no_init_recall": float(no_init),
                         "centroid_with_init_recall_mean": float(np.mean(with_init)),
                         "centroid_with_init_recall_std": float(np.std(with_init, ddof=1))})
    return pd.DataFrame(rows)


def pilot_random(cfg) -> tuple[pd.DataFrame, dict]:
    phi = plan.phi_star(cfg)
    runs = C.PROJECT_ROOT / cfg["paths"]["runs"]
    rows = []
    prob = store.problem(cfg, "O2", f"mgn_phi{phi:.2f}", phi)
    B = budget_for(0.02, prob.n, cfg["experiments"]["min_budget"])
    for seed in range(10):
        pilot = RunSpec("phase6", "random", f"topo=mgn_phi{phi:.2f}", "O2", 0.02, seed, f"mgn_phi{phi:.2f}", phi)
        bench = [s for s in plan.main_specs(cfg) if s.method == "random" and s.objective == "O2"
                 and s.budget_frac == 0.02 and s.seed == seed][0]
        stored = json.loads(pilot.paths(runs)[1].read_text())
        o = methods.make("random").run(prob, B, seed, methods.initial_design(prob.n, 10, seed))
        recomputed = float(prob.objective.hits[o.order].sum() / prob.objective.hits.sum())
        rows.append({"seed": seed, "pilot_key": pilot.key, "benchmark_key": bench.key,
                     "same_key": pilot.key == bench.key, "stored_recall": stored["final_recall"],
                     "recomputed_recall": recomputed, "hits_found": stored["hits_found"]})
    d = pd.DataFrame(rows)
    K, N = int(prob.objective.hits.sum()), prob.n
    exp_total, p = S.hypergeom_sum_test(int(d.hits_found.sum()), len(d), N, K, B)
    full = [json.loads(s.paths(runs)[1].read_text())["final_recall"] for s in plan.main_specs(cfg)
            if s.method == "random" and s.objective == "O2" and s.budget_frac == 0.02]
    summary = {"budget": B, "N": N, "K": K, "pilot_mean": float(d.stored_recall.mean()),
               "pilot_std": float(d.stored_recall.std(ddof=1)), "pilot_hits_total": int(d.hits_found.sum()),
               "expected_hits_total": exp_total, "exact_p_two_sided": p, "expected_recall": B / N,
               "benchmark_30_mean": float(np.mean(full)), "benchmark_30_std": float(np.std(full, ddof=1)),
               "benchmark_seeds_10_29_mean": float(np.mean(full[10:])),
               "all_keys_identical": bool(d.same_key.all()),
               "recomputed_equals_stored": bool(np.allclose(d.stored_recall, d.recomputed_recall))}
    return d, summary


def iteration_accounting(cfg) -> pd.DataFrame:
    P = cfg["experiments"]["default_P"]
    df = store.Store(cfg).load_clean()
    rows = []
    for o in plan.objectives_to_run(cfg):
        n = len(objectives.make(o, df).universe)
        for b in cfg["phase7"]["budgets"]:
            B = budget_for(b, n, cfg["experiments"]["min_budget"])
            T = B // P
            rows.append({"objective": o, "budget_frac": b, "pool_N": n, "budget_B": B, "P": P, "T": T,
                         "initial_design_evals": P, "update_iterations": T - 1,
                         "update_evals": (T - 1) * P, "remainder_evals_near_elite": B - T * P})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = C.load_config(None)
    OUT.mkdir(parents=True, exist_ok=True)
    cp = centroid_policies(cfg)
    cp.to_csv(OUT / "centroid_policy_variants.csv", index=False)
    d, summ = pilot_random(cfg)
    d.to_csv(OUT / "pilot_random_check.csv", index=False)
    (OUT / "pilot_random_summary.json").write_text(json.dumps(summ, indent=1))
    it = iteration_accounting(cfg)
    it.to_csv(OUT / "iteration_accounting.csv", index=False)
    pd.set_option("display.width", 200)
    print(cp.round(4).to_string(index=False))
    print(json.dumps(summ, indent=1))
    print(it.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
