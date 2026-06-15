from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(resolve_project_root(config_path, cfg))
    return cfg


def resolve_project_root(config_path: Path, cfg: dict[str, Any]) -> Path:
    root = cfg.get("project_root", ".")
    if root == ".":
        return config_path.parent.parent.resolve()
    root_path = Path(root)
    if root_path.is_absolute():
        return root_path.resolve()
    return (config_path.parent.parent / root_path).resolve()


def project_path(cfg: dict[str, Any], relative: str | Path) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return Path(cfg["_project_root"]) / path

