"""One module per pipeline stage. Each exposes ``run(cfg, force=False) -> GateResult``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    stage: int
    passed: bool
    gate: str  # "PASS" | "FAIL" | "SOFT-FAIL" | "SKIPPED (smoke)" | "n/a"
    summary: str

    def line(self) -> str:
        return f"[stage {self.stage}] {self.gate}: {self.summary}"
