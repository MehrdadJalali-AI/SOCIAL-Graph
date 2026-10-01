"""Configuration loading, project paths and persistent run state (gates, stage status)."""

from __future__ import annotations

import copy
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None, smoke: bool = False) -> dict[str, Any]:
    """Load ``default.yaml`` and merge ``path`` (and ``smoke.yaml`` if ``smoke``) on top."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        cfg = _deep_merge(cfg, yaml.safe_load(Path(path).read_text()) or {})
    if smoke and not cfg.get("smoke"):
        cfg = _deep_merge(cfg, yaml.safe_load((PROJECT_ROOT / "configs" / "smoke.yaml").read_text()))
    cfg["smoke"] = bool(cfg.get("smoke") or smoke)
    return cfg


def path(cfg: dict[str, Any], key: str) -> Path:
    """Resolve a configured path relative to the project root and create its parent."""
    p = PROJECT_ROOT / cfg["paths"][key]
    (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def resolve(rel: str | Path) -> Path:
    return PROJECT_ROOT / rel


class State:
    """JSON-backed record of stage completion and gate outcomes (used by ``--status``)."""

    def __init__(self, file: Path):
        self.file = file
        self.data: dict[str, Any] = json.loads(file.read_text()) if file.exists() else {"stages": {}}

    def save(self) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.file)

    def record(self, stage: int, status: str, gate: str | None = None, summary: str = "") -> None:
        self.data["stages"][str(stage)] = {
            "status": status,
            "gate": gate,
            "summary": summary,
            "time": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def get(self, stage: int) -> dict[str, Any] | None:
        return self.data["stages"].get(str(stage))
