from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .image_ops import (
    adjust_saturation_rgba,
    alpha_blend,
    overlay_rgba,
    random_background,
    random_color_drift,
    random_degrade,
    random_region_occlusion,
    read_rgb,
    read_rgba,
    resize_cover,
    resize_rgba,
    resize_contain_rgba,
    solid_background,
)


REQUIRED_COLUMNS = {
    "variant_id",
    "item_class_id",
    "item_id",
    "item_name",
    "category",
    "quality_level",
    "image_path",
    "background_path",
    "food_quality",
    "food_base_name",
    "weapon_state",
    "allowed_quality_levels",
    "source_json",
    "split",
}

ALLOWED_CATEGORIES = {"material", "food", "relic", "weapon"}
ALLOWED_FOOD_QUALITIES = {"", "normal", "weird", "delicious"}
ALLOWED_WEAPON_STATES = {"", "normal", "awaken"}
SPECIAL_DELICIOUS_OVERLAY_FOODS = {"鎏金殿堂", "一捧绿野", "雾凇秋分", "白浪拂沙"}


def load_labels(path: str | Path) -> pd.DataFrame:
    labels_path = Path(path)
    if not labels_path.exists():
        raise FileNotFoundError(f"找不到 labels csv: {labels_path}")
    df = pd.read_csv(labels_path, dtype=str).fillna("")
    if "quality_level" not in df.columns and "rank_level" in df.columns:
        df["quality_level"] = df["rank_level"]
    if "allowed_quality_levels" not in df.columns and "allowed_rank_levels" in df.columns:
        df["allowed_quality_levels"] = df["allowed_rank_levels"]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"labels csv 缺少字段: {sorted(missing)}")
    if df.empty:
        raise ValueError("labels csv 是空文件")
    return df


def _sort_key_number_text(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 10_000, value


def make_mappings(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    classes = sorted(df["item_class_id"].astype(str).unique())
    qualities = sorted((str(x) for x in df["quality_level"].astype(str).unique() if str(x)), key=_sort_key_number_text)
    return {
        "item_class_to_idx": {name: i for i, name in enumerate(classes)},
        "idx_to_item_class": {str(i): name for i, name in enumerate(classes)},
        "quality_to_idx": {name: i for i, name in enumerate(qualities)},
        "idx_to_quality": {str(i): name for i, name in enumerate(qualities)},
    }


def split_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and df["split"].astype(str).str.len().gt(0).any():
        split = df["split"].astype(str).str.lower()
        train_df = df[split.eq("train")].copy()
        val_df = df[split.isin(["val", "valid", "validation"])].copy()
        if train_df.empty:
            raise ValueError("labels csv 有 split 字段，但没有 train 行")
        if val_df.empty:
            val_df = train_df.copy()
        return train_df, val_df
    return df.copy(), df.copy()


def resolve_data_path(root: Path, value: Any) -> Path | None:
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return root / path


class ItemDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        cfg: dict[str, Any],
        mappings: dict[str, dict[str, int]],
        train: bool,
        seed: int = 42,
        views_per_sample: int = 1,
        prototype: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.train = train
        self.prototype = prototype
        self.mappings = mappings
        self.rng = np.random.default_rng(seed)
        self.views_per_sample = max(1, int(views_per_sample))
        self.cache_images = bool(cfg.get("data", {}).get("cache_images", True))
        self._rgba_cache: dict[str, np.ndarray] = {}
        self._rgb_cache: dict[str, np.ndarray] = {}
        image_size = cfg["data"].get("image_size", [125, 125])
        self.size = (int(image_size[0]), int(image_size[1]))
        norm = cfg.get("normalization", {})
        self.mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        if self.train and self.views_per_sample > 1:
            tensors = [self.image_to_tensor(self.render_row(row)) for _ in range(self.views_per_sample)]
            tensor = np.stack(tensors, axis=0)
        else:
            tensor = self.image_to_tensor(self.render_row(row))

        item_class_id = str(row["item_class_id"])
        quality_level = str(row["quality_level"])
        category = str(row.get("category", ""))
        class_idx = self.mappings["item_class_to_idx"][item_class_id]
        quality_to_idx = self.mappings.get("quality_to_idx") or self.mappings.get("rank_to_idx", {})
        quality_idx = quality_to_idx.get(quality_level, -1)
        quality_loss_categories = {str(value) for value in self.cfg.get("train", {}).get("quality_loss_categories", [])}
        if quality_loss_categories and category not in quality_loss_categories:
            quality_idx = -1
        return {
            "image": torch.from_numpy(tensor).float(),
            "item_class_idx": torch.tensor(class_idx, dtype=torch.long),
            "quality_idx": torch.tensor(quality_idx, dtype=torch.long),
            "variant_id": str(row["variant_id"]),
            "item_class_id": item_class_id,
            "item_id": str(row["item_id"]),
        }

    def image_to_tensor(self, image: np.ndarray) -> np.ndarray:
        tensor = image.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)
        return (tensor - self.mean) / self.std

    def read_rgba_cached(self, path: Path) -> np.ndarray:
        if not self.cache_images:
            return read_rgba(path)
        key = str(path)
        image = self._rgba_cache.get(key)
        if image is None:
            image = read_rgba(path)
            self._rgba_cache[key] = image
        return image

    def read_rgb_cached(self, path: Path) -> np.ndarray:
        if not self.cache_images:
            return read_rgb(path)
        key = str(path)
        image = self._rgb_cache.get(key)
        if image is None:
            image = read_rgb(path)
            self._rgb_cache[key] = image
        return image

    def load_icon(self, row: pd.Series) -> np.ndarray:
        image_path = resolve_data_path(self.root, row["image_path"])
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"找不到物品图标: {image_path}")
        icon = self.read_rgba_cached(image_path)
        food_quality = str(row.get("food_quality", ""))
        item_name = str(row.get("item_name", ""))
        compose_cfg = self.cfg.get("compose", {})
        if food_quality == "weird":
            factor = float(compose_cfg.get("weird_food_saturation_factor", 0.8))
            icon = adjust_saturation_rgba(icon, factor)
        elif food_quality == "delicious" or item_name in SPECIAL_DELICIOUS_OVERLAY_FOODS:
            overlay_path = resolve_data_path(self.root, compose_cfg.get("delicious_overlay", ""))
            if overlay_path is None or not overlay_path.exists():
                raise FileNotFoundError(f"找不到美味食物叠层: {overlay_path}")
            overlay = self.read_rgba_cached(overlay_path)
            if overlay.shape[:2] != icon.shape[:2]:
                overlay = resize_rgba(overlay, (icon.shape[1], icon.shape[0]))
            icon = overlay_rgba(icon, overlay, 0, 0)
        return icon

    def render_row(self, row: pd.Series) -> np.ndarray:
        augment_cfg = self.cfg.get("augment", {})
        random_bg_cfg = augment_cfg.get("random_background", {})
        category = str(row.get("category", ""))
        random_bg_exclude = {str(value) for value in random_bg_cfg.get("exclude_categories", [])}
        use_random_background = (
            self.train
            and not self.prototype
            and bool(random_bg_cfg.get("enabled", False))
            and category not in random_bg_exclude
            and self.rng.random() < float(random_bg_cfg.get("probability", 0.0))
        )
        if use_random_background:
            background = random_background(self.size, random_bg_cfg, self.rng)
        else:
            bg_path = resolve_data_path(self.root, row["background_path"])
            if bg_path and bg_path.exists():
                background = resize_cover(self.read_rgb_cached(bg_path), self.size)
            else:
                background = solid_background(self.size, row.get("quality_level", row.get("rank_level", "")))
        if self.train and augment_cfg.get("background_drift"):
            background = random_color_drift(background, augment_cfg["background_drift"], self.rng)

        icon = self.load_icon(row)
        compose_cfg = self.cfg.get("compose", {})
        icon_base_scale = float(compose_cfg.get("icon_scale", 1.0))
        icon_offset = compose_cfg.get("icon_offset", [0, 0])
        base_shift_x = int(icon_offset[0]) if len(icon_offset) > 0 else 0
        base_shift_y = int(icon_offset[1]) if len(icon_offset) > 1 else 0
        if self.train:
            scale_min, scale_max = augment_cfg.get("scale_range", [1.0, 1.0])
            scale = icon_base_scale * float(self.rng.uniform(float(scale_min), float(scale_max)))
            translate = int(augment_cfg.get("translate_px", 0))
            shift_x = base_shift_x + (int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0)
            shift_y = base_shift_y + (int(self.rng.integers(-translate, translate + 1)) if translate > 0 else 0)
        else:
            scale = icon_base_scale
            shift_x = base_shift_x
            shift_y = base_shift_y

        foreground = resize_contain_rgba(icon, self.size, scale, shift_x, shift_y)
        image = alpha_blend(foreground, background)
        if self.train:
            if not self.prototype:
                image = random_region_occlusion(image, augment_cfg.get("occlusion", {}), self.rng)
            image = random_color_drift(image, augment_cfg, self.rng)
            image = random_degrade(image, augment_cfg, self.rng)
        return image
