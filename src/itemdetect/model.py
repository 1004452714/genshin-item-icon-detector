from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ItemNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_qualities: int,
        embedding_dim: int = 256,
        base_channels: int = 80,
        dropout: float = 0.02,
        metric_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        metric_cfg = metric_head or {}
        self.metric_head_enabled = bool(metric_cfg.get("enabled", False))
        self.metric_head_type = str(metric_cfg.get("type", "cosface")).lower()
        self.metric_scale = float(metric_cfg.get("scale", 48.0))
        self.metric_margin = float(metric_cfg.get("margin", 0.35))
        if self.metric_head_enabled and self.metric_head_type != "cosface":
            raise ValueError("metric_head.type 目前只支持 cosface")

        c = int(base_channels)
        self.features = nn.Sequential(
            ConvBlock(3, c, stride=2),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(c * 4, int(embedding_dim)),
        )
        self.classifier = nn.Linear(int(embedding_dim), int(num_classes), bias=not self.metric_head_enabled)
        self.quality_head = nn.Linear(int(embedding_dim), int(num_qualities)) if int(num_qualities) > 0 else None

    def classify(self, embedding: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        if not self.metric_head_enabled:
            return self.classifier(embedding)

        logits = F.linear(embedding, F.normalize(self.classifier.weight, dim=1))
        if targets is not None:
            one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(dtype=logits.dtype, device=logits.device)
            logits = logits - one_hot * self.metric_margin
        return logits * self.metric_scale

    @property
    def metric_output_scale(self) -> float:
        return self.metric_scale if self.metric_head_enabled else 1.0

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(x)
        embedding = F.normalize(self.embedding(features), dim=1)
        class_logits = self.classify(embedding, targets)
        if self.quality_head is None:
            quality_logits = embedding.new_zeros((embedding.shape[0], 0))
        else:
            quality_logits = self.quality_head(embedding)
        return embedding, class_logits, quality_logits


class OnnxItemWrapper(nn.Module):
    def __init__(self, model: ItemNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding, _, _ = self.model(x)
        return embedding
