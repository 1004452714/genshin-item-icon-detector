from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from .config import load_config, project_path
from .dataset import (
    ALLOWED_CATEGORIES,
    ALLOWED_FOOD_QUALITIES,
    ALLOWED_WEAPON_STATES,
    load_labels,
    resolve_data_path,
)


def validate_image(path: Path, row_index: int, label: str, errors: list[str], warnings: list[str]) -> None:
    if not path.exists():
        errors.append(f"第 {row_index} 行找不到{label}: {path}")
        return
    try:
        with Image.open(path) as img:
            if label == "物品图标" and "A" not in img.getbands():
                warnings.append(f"第 {row_index} 行物品图标没有 alpha 通道: {path}")
    except Exception as exc:
        errors.append(f"第 {row_index} 行无法读取{label} {path}: {exc}")


def validate_data(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])
    labels_path = project_path(cfg, cfg["data"]["labels_csv"])
    df = load_labels(labels_path)
    errors: list[str] = []
    warnings: list[str] = []

    duplicated_variants = df[df["variant_id"].astype(str).duplicated()]["variant_id"].astype(str).tolist()
    if duplicated_variants:
        errors.append(f"variant_id 重复: {sorted(set(duplicated_variants))[:20]}")

    class_count = df["item_class_id"].nunique()
    quality_values = sorted(df["quality_level"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else 99)
    for i, row in df.iterrows():
        category = str(row["category"])
        food_quality = str(row["food_quality"])
        weapon_state = str(row["weapon_state"])
        quality_level = str(row["quality_level"])
        if category not in ALLOWED_CATEGORIES:
            errors.append(f"第 {i} 行未知类别: {category}")
        if food_quality not in ALLOWED_FOOD_QUALITIES:
            errors.append(f"第 {i} 行未知食物品级: {food_quality}")
        if weapon_state not in ALLOWED_WEAPON_STATES:
            errors.append(f"第 {i} 行未知武器状态: {weapon_state}")
        if quality_level not in {"0", "1", "2", "3", "4", "5"}:
            errors.append(f"第 {i} 行未知品质: {quality_level}")
        if category == "food" and not food_quality:
            errors.append(f"第 {i} 行食物缺少 food_quality")
        if category == "weapon" and weapon_state not in {"normal", "awaken"}:
            errors.append(f"第 {i} 行武器缺少 normal/awaken 状态")
        if category != "weapon" and weapon_state:
            errors.append(f"第 {i} 行非武器不应有 weapon_state")
        if category != "food" and food_quality:
            errors.append(f"第 {i} 行非食物不应有 food_quality")
        if category == "relic":
            allowed = {x for x in str(row["allowed_quality_levels"]).split("|") if x}
            if not allowed:
                errors.append(f"第 {i} 行圣遗物缺少 allowed_quality_levels")
            if quality_level not in allowed:
                errors.append(f"第 {i} 行圣遗物 quality_level 不在 allowed_quality_levels 中")

        icon_path = resolve_data_path(root, row["image_path"])
        bg_path = resolve_data_path(root, row["background_path"])
        if icon_path is None:
            errors.append(f"第 {i} 行 image_path 为空")
        else:
            validate_image(icon_path, i, "物品图标", errors, warnings)
        if bg_path is None:
            errors.append(f"第 {i} 行 background_path 为空")
        else:
            validate_image(bg_path, i, "背景图", errors, warnings)

    print(f"行数={len(df)}")
    print(f"主类别数={class_count}")
    print(f"品质值={','.join(quality_values)}")
    for warning in warnings:
        print(f"警告: {warning}")
    if errors:
        for error in errors:
            print(f"错误: {error}")
        raise SystemExit(1)
    print("数据校验通过")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_data(args.config)
