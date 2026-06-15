from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import ItemNet


def _checkpoint_state_with_quality_names(state: dict[str, Any]) -> dict[str, Any]:
    if not any(key.startswith("rank_head.") for key in state):
        return state
    return {
        ("quality_head." + key[len("rank_head.") :]) if key.startswith("rank_head.") else key: value
        for key, value in state.items()
    }


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[ItemNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    mappings = checkpoint["mappings"]
    model_cfg = cfg.get("model", {})
    class_mapping = mappings["item_class_to_idx"]
    quality_mapping = mappings.get("quality_to_idx") or mappings.get("rank_to_idx", {})
    quality_head_enabled = bool(model_cfg.get("quality_head", model_cfg.get("rank_head", True)))
    model = ItemNet(
        num_classes=len(class_mapping),
        num_qualities=len(quality_mapping) if quality_head_enabled else 0,
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        base_channels=int(model_cfg.get("base_channels", 80)),
        dropout=float(model_cfg.get("dropout", 0.02)),
        metric_head=model_cfg.get("metric_head"),
    )
    model.load_state_dict(_checkpoint_state_with_quality_names(checkpoint["model_state"]))
    model.to(device)
    model.eval()
    return model, checkpoint
