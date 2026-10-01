"""Command-line entry point.

    python -m graphsocial.run --stage 1
    python -m graphsocial.run --stage all --config configs/smoke.yaml --smoke
    python -m graphsocial.run --status

Stages are gated: a FAIL writes the report, prints a one-line summary and exits non-zero. Later
stages refuse to start while an earlier gate is FAIL unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys

from . import config as C

STAGES: dict[int, str] = {
    1: "phase1",  # MOFGalaxyNet reproduction (GATE 1)
    2: "phase2",  # QMOF data preparation (GATE 2, soft)
    3: "phase3",  # graphs and comparison topologies
    4: "phase4",  # landscape pre-checks (GATE 4)
    5: "phase5",  # method checks
    6: "phase6",  # H1 pilot (GATE 6)
    7: "phase7",  # full benchmark, ablations, sensitivity
    8: "phase8",  # analysis, report, release
}
PLANNED = {
    1: "MOFGalaxyNet 2k reproduction (GATE 1)",
    2: "QMOF data preparation (GATE 2, soft)",
    3: "QMOF-MOFGalaxyNet + comparison topologies",
    4: "landscape pre-checks (GATE 4)",
    5: "Graph-SOCIAL and baselines (tests)",
    6: "H1 pilot (GATE 6)",
    7: "full benchmark, ablations, sensitivity",
    8: "analysis, report, release",
}
HARD_GATES = {1, 4, 5, 6}

log = logging.getLogger("graphsocial")


def _status(cfg: dict) -> None:
    st = C.State(C.path(cfg, "state"))
    print(f"state file: {st.file}")
    for s in sorted(set(STAGES) | set(PLANNED)):
        rec = st.get(s)
        name = PLANNED[s]
        if rec is None:
            impl = "" if s in STAGES else " (not implemented yet)"
            print(f"  stage {s}: pending{impl} — {name}")
        else:
            print(f"  stage {s}: {rec['status']} | gate {rec['gate']} | {rec['time']} | {rec['summary']}")


def _blocking_gate(st: C.State, stage: int, overrides: set[int]) -> int | None:
    for s in sorted(HARD_GATES - overrides):
        if s < stage and (rec := st.get(s)) and rec["gate"] == "FAIL":
            return s
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m graphsocial.run")
    ap.add_argument("--stage", default=None, help="1..8 or 'all'")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny subsets; finishes in minutes")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="continue past a failed gate")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--provisional", action="store_true",
                    help="stage 8 only: analyse incomplete runs as a clearly marked draft (manuscript checks fail)")
    ap.add_argument("--only-methods", default=None, help="comma list: run only these methods (stage 6/7)")
    ap.add_argument("--skip-methods", default=None, help="comma list: skip these methods (stage 6/7)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = C.load_config(args.config, smoke=args.smoke)
    cfg["n_jobs"] = args.n_jobs
    cfg["provisional"] = args.provisional
    cfg["method_filter"] = {"only": args.only_methods.split(",") if args.only_methods else None,
                            "skip": args.skip_methods.split(",") if args.skip_methods else []}
    if args.status:
        _status(cfg)
        return 0
    if args.stage is None:
        ap.error("--stage or --status is required")

    stages = sorted(STAGES) if args.stage == "all" else [int(args.stage)]
    st = C.State(C.path(cfg, "state"))
    overrides = set(cfg.get("gate_overrides", []))
    for s in stages:
        if s not in STAGES:
            print(f"[stage {s}] not implemented yet: {PLANNED.get(s, '?')}")
            return 3
        if (blk := _blocking_gate(st, s, overrides)) is not None and not args.force:
            print(f"[stage {s}] blocked: gate {blk} is FAIL (re-run with --force to override)")
            return 2
        log.info("stage %d (%s) starting", s, STAGES[s])
        st.record(s, "running")
        res = importlib.import_module(f".stages.{STAGES[s]}", __package__).run(cfg, force=args.force)
        st.record(s, "done", res.gate, res.summary)
        print(res.line(), flush=True)
        if not res.passed and s in HARD_GATES and not args.force:
            if s in overrides:
                print(f"[stage {s}] gate FAIL overridden by config (gate_overrides) — continuing")
                continue
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
