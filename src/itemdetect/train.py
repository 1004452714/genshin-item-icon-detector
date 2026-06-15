from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

from .config import load_config, project_path
from .dataset import ItemDataset, load_labels, make_mappings, split_labels
from .model import ItemNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker_info = get_worker_info()
    if worker_info is not None and hasattr(worker_info.dataset, "rng"):
        worker_info.dataset.rng = np.random.default_rng(worker_seed)


def build_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, dict[str, int]]]:
    root = Path(cfg["_project_root"])
    df = load_labels(project_path(cfg, cfg["data"]["labels_csv"]))
    train_df, val_df = split_labels(df)
    mappings = make_mappings(df)
    seed = int(cfg["train"].get("seed", 42))
    views_per_sample = int(cfg["train"].get("views_per_sample", 1))
    train_set = ItemDataset(train_df, root, cfg, mappings, train=True, seed=seed, views_per_sample=views_per_sample)
    val_set = ItemDataset(val_df, root, cfg, mappings, train=False, seed=seed + 1)
    batch_size = int(cfg["train"].get("batch_size", 64))
    num_workers = int(cfg["data"].get("num_workers", 0))
    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(cfg["data"].get("pin_memory", torch.cuda.is_available())),
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["data"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 4))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, mappings


def top1_margins(score_logits: torch.Tensor, score_scale: float) -> list[float]:
    if score_logits.shape[1] < 2:
        return []
    scores = score_logits / max(score_scale, 1e-12)
    top2 = torch.topk(scores, k=2, dim=1).values
    return (top2[:, 0] - top2[:, 1]).detach().cpu().tolist()


def supervised_contrastive_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())
    temperature = max(float(temperature), 1e-6)
    logits = embedding @ embedding.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count.gt(0)
    if not valid.any():
        return embedding.new_zeros(())

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    loss = -(log_prob * positive_mask).sum(dim=1)[valid] / positive_count[valid]
    return loss.mean()


def batch_hard_cosine_loss(
    embedding: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if embedding.shape[0] < 2:
        return embedding.new_zeros(())
    similarities = embedding @ embedding.T
    self_mask = torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    negative_mask = labels[:, None].ne(labels[None, :])
    valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
    if not valid.any():
        return embedding.new_zeros(())

    hardest_positive = similarities.masked_fill(~positive_mask, 1.0).min(dim=1).values
    hardest_negative = similarities.masked_fill(~negative_mask, -1.0).max(dim=1).values
    return torch.relu(float(margin) + hardest_negative[valid] - hardest_positive[valid]).mean()


def target_logit_gap_loss(
    score_logits: torch.Tensor,
    labels: torch.Tensor,
    score_scale: float,
    margin: float,
) -> torch.Tensor:
    if score_logits.shape[1] < 2:
        return score_logits.new_zeros(())
    scores = score_logits / max(score_scale, 1e-12)
    target_scores = scores.gather(1, labels[:, None]).squeeze(1)
    target_mask = F.one_hot(labels, num_classes=scores.shape[1]).to(dtype=torch.bool, device=scores.device)
    hardest_negative = scores.masked_fill(target_mask, -1e4).max(dim=1).values
    return torch.relu(float(margin) - (target_scores - hardest_negative)).mean()


def run_epoch(
    model: ItemNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    cfg: dict[str, Any],
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_quality_correct = 0
    total_count = 0
    quality_count = 0
    margins: list[float] = []
    quality_weight = float(cfg["train"].get("quality_loss_weight", 0.0))
    contrastive_weight = float(cfg["train"].get("contrastive_loss_weight", 0.0))
    contrastive_temperature = float(cfg["train"].get("contrastive_temperature", 0.07))
    hard_negative_weight = float(cfg["train"].get("hard_negative_loss_weight", 0.0))
    hard_negative_margin = float(cfg["train"].get("hard_negative_margin", 0.12))
    logit_gap_weight = float(cfg["train"].get("logit_gap_loss_weight", 0.0))
    logit_gap_margin = float(cfg["train"].get("logit_gap_margin", 0.10))
    use_amp = bool(cfg["train"].get("mixed_precision", True)) and device.type == "cuda"

    for batch in tqdm(loader, leave=False, ascii=True):
        images = batch["image"].to(device, non_blocking=True)
        class_idx = batch["item_class_idx"].to(device, non_blocking=True)
        quality_idx = batch["quality_idx"].to(device, non_blocking=True)
        if images.dim() == 5:
            batch_size, views, channels, height, width = images.shape
            images = images.reshape(batch_size * views, channels, height, width)
            class_idx = class_idx.repeat_interleave(views)
            quality_idx = quality_idx.repeat_interleave(views)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with autocast(device_type=device.type, enabled=use_amp):
                embedding, class_logits, quality_logits = model(images, class_idx if is_train else None)
                loss = ce(class_logits, class_idx)
                if quality_weight > 0 and quality_logits.shape[1] > 0:
                    valid_quality = quality_idx.ge(0)
                    if valid_quality.any():
                        loss = loss + quality_weight * ce(quality_logits[valid_quality], quality_idx[valid_quality])
                if is_train and contrastive_weight > 0:
                    loss = loss + contrastive_weight * supervised_contrastive_loss(
                        embedding,
                        class_idx,
                        contrastive_temperature,
                    )
                if is_train and hard_negative_weight > 0:
                    loss = loss + hard_negative_weight * batch_hard_cosine_loss(
                        embedding,
                        class_idx,
                        hard_negative_margin,
                    )
                if is_train and logit_gap_weight > 0:
                    score_logits_for_loss = model.classify(embedding, None)
                    loss = loss + logit_gap_weight * target_logit_gap_loss(
                        score_logits_for_loss,
                        class_idx,
                        model.metric_output_scale,
                        logit_gap_margin,
                    )

            with torch.no_grad():
                with autocast(device_type=device.type, enabled=use_amp):
                    score_logits = model.classify(embedding.detach(), None)

            if is_train:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += float(loss.detach().cpu()) * images.shape[0]
        total_correct += int(score_logits.argmax(dim=1).eq(class_idx).sum().detach().cpu())
        if quality_logits.shape[1] > 0:
            valid_quality = quality_idx.ge(0)
            if valid_quality.any():
                total_quality_correct += int(
                    quality_logits[valid_quality].argmax(dim=1).eq(quality_idx[valid_quality]).sum().detach().cpu(),
                )
                quality_count += int(valid_quality.sum().detach().cpu())
        total_count += int(images.shape[0])
        margins.extend(top1_margins(score_logits, model.metric_output_scale))

    return {
        "loss": total_loss / max(1, total_count),
        "top1": total_correct / max(1, total_count),
        "quality_top1": total_quality_correct / max(1, quality_count),
        "margin_mean": float(np.mean(margins)) if margins else 0.0,
        "margin_p10": float(np.percentile(margins, 10)) if margins else 0.0,
    }


def save_checkpoint(
    path: Path,
    model: ItemNet,
    mappings: dict[str, dict[str, int]],
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "mappings": mappings,
        "config": cfg,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def train_main(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["train"].get("seed", 42)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader, mappings = build_loaders(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg.get("model", {})
    model = ItemNet(
        num_classes=len(mappings["item_class_to_idx"]),
        num_qualities=len(mappings["quality_to_idx"])
        if model_cfg.get("quality_head", True)
        else 0,
        embedding_dim=int(model_cfg.get("embedding_dim", 256)),
        base_channels=int(model_cfg.get("base_channels", 80)),
        dropout=float(model_cfg.get("dropout", 0.02)),
        metric_head=model_cfg.get("metric_head"),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("learning_rate", 5e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda" and bool(cfg["train"].get("mixed_precision", True)))
    save_dir = project_path(cfg, cfg["train"].get("save_dir", "outputs/checkpoints"))
    best_top1 = -1.0
    best_margin_p10 = float("-inf")
    quality_enabled = bool(model_cfg.get("quality_head", True))

    print(f"设备={device}")
    if quality_enabled:
        print(f"类别数={len(mappings['item_class_to_idx'])} 品质类别数={len(mappings['quality_to_idx'])}")
    else:
        print(f"类别数={len(mappings['item_class_to_idx'])} 品质头=关闭")
    for epoch in range(1, int(cfg["train"].get("epochs", 160)) + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer, scaler, cfg)
        val_metrics = run_epoch(model, val_loader, device, None, None, cfg)
        if quality_enabled:
            print(
                f"轮次={epoch} "
                f"训练损失={train_metrics['loss']:.4f} 训练top1={train_metrics['top1']:.4f} "
                f"训练品质top1={train_metrics['quality_top1']:.4f} 训练间隔P10={train_metrics['margin_p10']:.4f} "
                f"验证损失={val_metrics['loss']:.4f} 验证top1={val_metrics['top1']:.4f} "
                f"验证品质top1={val_metrics['quality_top1']:.4f} "
                f"验证间隔均值={val_metrics['margin_mean']:.4f} 验证间隔P10={val_metrics['margin_p10']:.4f}"
            )
        else:
            print(
                f"轮次={epoch} "
                f"训练损失={train_metrics['loss']:.4f} 训练top1={train_metrics['top1']:.4f} "
                f"训练间隔P10={train_metrics['margin_p10']:.4f} "
                f"验证损失={val_metrics['loss']:.4f} 验证top1={val_metrics['top1']:.4f} "
                f"验证间隔均值={val_metrics['margin_mean']:.4f} 验证间隔P10={val_metrics['margin_p10']:.4f}"
            )
        save_checkpoint(save_dir / "last.pt", model, mappings, cfg, epoch, val_metrics)
        is_better = val_metrics["top1"] > best_top1 or (
            val_metrics["top1"] == best_top1 and val_metrics["margin_p10"] >= best_margin_p10
        )
        if is_better:
            best_top1 = val_metrics["top1"]
            best_margin_p10 = val_metrics["margin_p10"]
            save_checkpoint(save_dir / "best.pt", model, mappings, cfg, epoch, val_metrics)
            (save_dir / "mappings.json").write_text(json.dumps(mappings, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_main(args.config)
