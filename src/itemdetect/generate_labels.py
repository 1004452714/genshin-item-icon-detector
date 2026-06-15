from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FIELDNAMES = [
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
]

MATERIAL_ITEM1_TYPES = {0, 8}
MATERIAL_ITEM2_TYPES = {0, 1, 4, 5, 6, 10, 11, 13, 14, 15, 21, 24, 25, 31, 32, 33}
FOOD_MATERIAL_TYPES = {1, 15}
SKIPPED_FOOD_ITEM_IDS = {"121374"}
SKIPPED_FOOD_ITEM_ID_REASON = "食物 item_id 黑名单"

QUALITY_BACKGROUND = {
    "0": "UI_QUALITY_WHITE.png",
    "1": "UI_QUALITY_WHITE.png",
    "2": "UI_QUALITY_GREEN.png",
    "3": "UI_QUALITY_BLUE.png",
    "4": "UI_QUALITY_PURPLE.png",
    "5": "UI_QUALITY_ORANGE.png",
}

FOOD_PREFIXES = {
    "奇怪的": "weird",
    "美味的": "delicious",
}

SKIPPED_FOOD_NAME_REASON = "食物生成黑名单"
SKIPPED_FOOD_NAMES = {
    "摇滚团子牛奶",
    "海灯节特色扣三丝",
    "海灯节特色来来菜",
    "海灯节特色炸萝卜丸子",
    "海灯节特色烤吃虎鱼",
    "海灯节特色热卤面",
    "海灯节特色白玉汤",
    "海灯节特色禽蛋羹",
}

KNOWN_MISSING_MATERIAL_IMAGE_REASON = "已知缺图材料黑名单"
KNOWN_MISSING_RELIC_IMAGE_REASON = "已知缺图圣遗物黑名单"
SKIPPED_DUPLICATE_MATERIAL_REASON_PREFIX = "同图多义且按规则忽略："
KNOWN_MISSING_MATERIAL_IMAGE_NAMES = {
    "火元素晶片",
    "水元素晶片",
    "草元素晶片",
    "电元素晶片",
    "风元素晶片",
    "冰元素晶片",
    "岩元素晶片",
    "原初元素晶片",
    "火炽奇卡卡",
    "水润奇卡卡",
    "草萌奇卡卡",
    "电击奇卡卡",
    "风纹奇卡卡",
    "冰结奇卡卡",
    "岩皮奇卡卡",
    "原初奇卡卡",
    "测试用体力成长道具",
    "测试用体力临时增长道具",
    "礼券秘盒",
    "料理",
    "合成产物",
}

KNOWN_MISSING_RELIC_IMAGE_NAMES = {
    "北风之盏",
    "冰河之冠",
    "高天的风之主杯",
    "高天的风之主冠",
    "高天的风之主花",
    "高天的风之主沙",
    "高天的风之主羽",
    "祭风礼冠",
    "凛冬霜心",
    "凝冰成砂",
    "雪藏之羽",
}

MERGED_MATERIAL_ICONS = {
    "UI_ItemIcon_109000": "食谱",
    "UI_ItemIcon_221003": "图谱",
    "UI_ItemIcon_221001": "说明/配方",
    "UI_ItemIcon_221035": "鱼饵配方",
    "UI_ItemIcon_241": "奇丽凝晶",
    "UI_ItemIcon_100061": "兽肉",
    "UI_ItemIcon_100064": "禽肉",
    "UI_ItemIcon_100950": "奇特的羽毛",
    "UI_ItemIcon_101005": "深赤之石",
    "UI_ItemIcon_107010": "绯红玉髓",
    "UI_ItemIcon_220038": "迷你仙灵·紫苑",
    "UI_ItemIcon_117002": "深秘圣物匣·二等",
    "UI_ItemIcon_117003": "深秘圣物匣·一等",
}

SKIPPED_DUPLICATE_MATERIAL_ICONS = {
    "UI_ItemIcon_120815": "同图多义且按规则忽略：培育区域扩展",
    "UI_ItemIcon_120821": "同图多义且按规则忽略：改进炼金釜",
    "UI_ItemIcon_220013": "同图多义且按规则忽略：「寻宝仙灵」",
    "UI_ItemIcon_220108": "同图多义且按规则忽略：派蒙的留影机/寻迹留影机",
    "UI_ItemIcon_101714": "同图多义且按规则忽略：香气四溢的调料",
    "UI_ItemIcon_220042": "同图多义且按规则忽略：阿叶夏混沌勘探器/芭努的智慧",
    "UI_ItemIcon_220051": "同图多义且按规则忽略：荒泷·盛世豪鼓/绮筵之鼓",
    "UI_ItemIcon_220085": "同图多义且按规则忽略：铭随流镜",
    "UI_ItemIcon_220090": "同图多义且按规则忽略：布列松的特别「留影机」/「回忆捕手留影机」",
}


@dataclass(frozen=True)
class SkippedLabel:
    source_json: str
    item_id: str
    item_name: str
    icon_name: str
    reason: str


@dataclass(frozen=True)
class UnusedImageSummary:
    folder_name: str
    count: int
    examples: list[str]


@dataclass(frozen=True)
class SharedIconNameSummary:
    category: str
    image_path: str
    names: list[str]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = list(data.values())
    elif isinstance(data, list):
        values = data
    else:
        raise ValueError(f"{path} 不是 JSON 数组或对象")
    return [item for item in values if isinstance(item, dict)]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def relative_for_csv(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def icon_png(icon_name: str) -> str:
    icon_name = icon_name.strip()
    return icon_name if icon_name.lower().endswith(".png") else f"{icon_name}.png"


def icon_stem(icon_name: str) -> str:
    return Path(icon_png(icon_name)).stem


def strip_icon_prefix(icon_name: str, prefix: str) -> str:
    icon_name = icon_stem(icon_name)
    return icon_name[len(prefix) :] if icon_name.startswith(prefix) else icon_name


def format_inline_names(names: Iterable[str]) -> str:
    return "、".join(sorted({name for name in names if name}))


def print_skipped_names_warning(reason: str, names: Iterable[str]) -> None:
    inline_names = format_inline_names(names)
    if inline_names:
        print(f"警告: {reason}，已跳过名称: {inline_names}")


def background_for_quality(quality_level: str, background_dir: Path) -> Path:
    name = QUALITY_BACKGROUND.get(quality_level)
    if not name:
        raise ValueError(f"未配置 RankLevel={quality_level} 的背景映射")
    path = background_dir / name
    if not path.exists():
        raise FileNotFoundError(f"找不到 RankLevel={quality_level} 对应背景图: {path}")
    return path


def normalize_food_name(name: str) -> tuple[str, str]:
    for prefix, quality in FOOD_PREFIXES.items():
        if name.startswith(prefix):
            return name[len(prefix) :], quality
    return name, "normal"


def should_use_material(item: dict[str, Any]) -> bool:
    item_type = item.get("ItemType")
    material_type = item.get("MaterialType")
    return (item_type == 1 and material_type in MATERIAL_ITEM1_TYPES) or (
        item_type == 2 and material_type in MATERIAL_ITEM2_TYPES
    )


def material_category(item: dict[str, Any]) -> str:
    if item.get("ItemType") == 2 and item.get("MaterialType") in FOOD_MATERIAL_TYPES:
        return "food"
    return "material"


def is_skipped_food_name(name: str) -> bool:
    return name in SKIPPED_FOOD_NAMES


def build_row(
    *,
    root: Path,
    source_json: str,
    item_id: str,
    item_name: str,
    category: str,
    quality_level: str,
    image_path: Path,
    background_path: Path,
    item_class_id: str,
    variant_id: str,
    split: str,
    food_quality: str = "",
    food_base_name: str = "",
    weapon_state: str = "",
    allowed_quality_levels: str = "",
) -> dict[str, str]:
    return {
        "variant_id": variant_id,
        "item_class_id": item_class_id,
        "item_id": item_id,
        "item_name": item_name,
        "category": category,
        "quality_level": quality_level,
        "image_path": relative_for_csv(image_path, root),
        "background_path": relative_for_csv(background_path, root),
        "food_quality": food_quality,
        "food_base_name": food_base_name,
        "weapon_state": weapon_state,
        "allowed_quality_levels": allowed_quality_levels,
        "source_json": source_json,
        "split": split,
    }


def rows_from_materials(
    *,
    root: Path,
    json_path: Path,
    item_dir: Path,
    background_dir: Path,
    split: str,
) -> tuple[list[dict[str, str]], list[SkippedLabel], set[str]]:
    rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    expected_icons: set[str] = set()
    emitted_merged_variants: set[tuple[str, str]] = set()
    for item in load_json_list(json_path):
        if not should_use_material(item):
            continue

        item_id = as_text(item.get("Id"))
        item_name = as_text(item.get("Name"))
        icon_name = as_text(item.get("Icon"))
        quality_level = as_text(item.get("RankLevel"))
        if icon_name:
            expected_icons.add(icon_name)
        if not item_id or not item_name or not icon_name or quality_level == "":
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, "缺少 Id/Name/Icon/RankLevel"))
            continue

        category = material_category(item)
        if category == "food" and item_id in SKIPPED_FOOD_ITEM_IDS:
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, SKIPPED_FOOD_ITEM_ID_REASON))
            continue
        if category == "food" and is_skipped_food_name(item_name):
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, SKIPPED_FOOD_NAME_REASON))
            continue

        image_path = item_dir / icon_png(icon_name)
        if not image_path.exists():
            reason = (
                KNOWN_MISSING_MATERIAL_IMAGE_REASON
                if item_name in KNOWN_MISSING_MATERIAL_IMAGE_NAMES
                else f"找不到图片 {image_path}"
            )
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, reason))
            continue

        background_path = background_for_quality(quality_level, background_dir)
        if category == "food":
            food_base_name, food_quality = normalize_food_name(item_name)
            icon_id = strip_icon_prefix(icon_name, "UI_ItemIcon_")
            item_class_id = f"food:{icon_id}:{food_quality}"
            variant_id = item_class_id
        else:
            food_base_name = ""
            food_quality = ""
            icon_key = icon_stem(icon_name)
            icon_id = strip_icon_prefix(icon_name, "UI_ItemIcon_")
            if icon_key in SKIPPED_DUPLICATE_MATERIAL_ICONS:
                skipped.append(
                    SkippedLabel(
                        json_path.name,
                        item_id,
                        item_name,
                        icon_name,
                        SKIPPED_DUPLICATE_MATERIAL_ICONS[icon_key],
                    )
                )
                continue
            if icon_key in MERGED_MATERIAL_ICONS:
                item_class_id = f"material:{icon_id}"
                variant_id = f"{item_class_id}:quality:{quality_level}"
                dedupe_key = (item_class_id, quality_level)
                if dedupe_key in emitted_merged_variants:
                    continue
                emitted_merged_variants.add(dedupe_key)
                item_id = f"merged:{icon_id}:{quality_level}"
                item_name = MERGED_MATERIAL_ICONS[icon_key]
            else:
                item_class_id = f"material:{item_id}"
                variant_id = item_class_id

        rows.append(
            build_row(
                root=root,
                source_json=json_path.name,
                item_id=item_id,
                item_name=item_name,
                category=category,
                quality_level=quality_level,
                image_path=image_path,
                background_path=background_path,
                item_class_id=item_class_id,
                variant_id=variant_id,
                split=split,
                food_quality=food_quality,
                food_base_name=food_base_name,
                allowed_quality_levels=quality_level,
            )
        )
    return rows, skipped, expected_icons


def rows_from_relics(
    *,
    root: Path,
    json_path: Path,
    relic_dir: Path,
    background_dir: Path,
    split: str,
) -> tuple[list[dict[str, str]], list[SkippedLabel], set[str]]:
    raw_rows = load_json_list(json_path)
    allowed_qualities_by_class: dict[str, set[str]] = {}
    for item in raw_rows:
        icon_name = as_text(item.get("Icon"))
        quality_level = as_text(item.get("RankLevel"))
        if icon_name and quality_level:
            item_class_id = f"relic:{strip_icon_prefix(icon_name, 'UI_RelicIcon_')}"
            allowed_qualities_by_class.setdefault(item_class_id, set()).add(quality_level)

    rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    expected_icons: set[str] = set()
    for row_index, item in enumerate(raw_rows):
        item_name = as_text(item.get("Name"))
        icon_name = as_text(item.get("Icon"))
        quality_level = as_text(item.get("RankLevel"))
        item_id = f"relic_{row_index}"
        if icon_name:
            expected_icons.add(icon_name)
        if not item_name or not icon_name or quality_level == "":
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, "缺少 Name/Icon/RankLevel"))
            continue
        image_path = relic_dir / icon_png(icon_name)
        if not image_path.exists():
            reason = (
                KNOWN_MISSING_RELIC_IMAGE_REASON
                if item_name in KNOWN_MISSING_RELIC_IMAGE_NAMES
                else f"找不到图片 {image_path}"
            )
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, reason))
            continue

        background_path = background_for_quality(quality_level, background_dir)
        relic_icon_id = strip_icon_prefix(icon_name, "UI_RelicIcon_")
        item_class_id = f"relic:{relic_icon_id}"
        allowed_quality_levels = "|".join(sorted(allowed_qualities_by_class[item_class_id], key=lambda x: int(x)))
        rows.append(
            build_row(
                root=root,
                source_json=json_path.name,
                item_id=item_id,
                item_name=item_name,
                category="relic",
                quality_level=quality_level,
                image_path=image_path,
                background_path=background_path,
                item_class_id=item_class_id,
                variant_id=f"{item_class_id}:{quality_level}:{row_index}",
                split=split,
                allowed_quality_levels=allowed_quality_levels,
            )
        )
    return rows, skipped, expected_icons


def rows_from_weapons(
    *,
    root: Path,
    json_path: Path,
    weapon_dir: Path,
    background_dir: Path,
    split: str,
) -> tuple[list[dict[str, str]], list[SkippedLabel], set[str]]:
    rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    expected_icons: set[str] = set()
    for item in load_json_list(json_path):
        item_id = as_text(item.get("Id"))
        item_name = as_text(item.get("Name"))
        quality_level = as_text(item.get("RankLevel"))
        icon_pairs = [("normal", as_text(item.get("Icon"))), ("awaken", as_text(item.get("AwakenIcon")))]
        for _, icon_name in icon_pairs:
            if icon_name:
                expected_icons.add(icon_name)
        if not item_id or not item_name or quality_level == "":
            skipped.append(SkippedLabel(json_path.name, item_id, item_name, "", "缺少 Id/Name/RankLevel"))
            continue

        background_path = background_for_quality(quality_level, background_dir)
        item_class_id = f"weapon:{item_id}"
        for weapon_state, icon_name in icon_pairs:
            if not icon_name:
                skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, f"缺少 {weapon_state} 图标字段"))
                continue
            image_path = weapon_dir / icon_png(icon_name)
            if not image_path.exists():
                skipped.append(SkippedLabel(json_path.name, item_id, item_name, icon_name, f"找不到图片 {image_path}"))
                continue
            rows.append(
                build_row(
                    root=root,
                    source_json=json_path.name,
                    item_id=item_id,
                    item_name=item_name,
                    category="weapon",
                    quality_level=quality_level,
                    image_path=image_path,
                    background_path=background_path,
                    item_class_id=item_class_id,
                    variant_id=f"{item_class_id}:{weapon_state}",
                    split=split,
                    weapon_state=weapon_state,
                    allowed_quality_levels=quality_level,
                )
            )
    return rows, skipped, expected_icons


def summarize_unused_images(folder: Path, expected_icon_names: Iterable[str], folder_name: str) -> UnusedImageSummary:
    expected = {name.removesuffix(".png") for name in expected_icon_names}
    unused = [path.stem for path in sorted(folder.glob("*.png"), key=lambda p: p.name.lower()) if path.stem not in expected]
    return UnusedImageSummary(folder_name=folder_name, count=len(unused), examples=unused[:50])


def summarize_shared_icon_names(rows: Iterable[dict[str, str]]) -> list[SharedIconNameSummary]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        category = row["category"]
        if category in {"food", "material"}:
            groups[(category, row["image_path"])].append(row)

    summaries: list[SharedIconNameSummary] = []
    for (category, image_path), group_rows in sorted(groups.items()):
        names = sorted({row["item_name"] for row in group_rows if row["item_name"]})
        if len(names) <= 1:
            continue
        if category == "food":
            base_names = sorted(
                {
                    row["food_base_name"] or normalize_food_name(row["item_name"])[0]
                    for row in group_rows
                    if row["item_name"]
                },
            )
            if len(base_names) <= 1:
                continue
        summaries.append(SharedIconNameSummary(category=category, image_path=image_path, names=names))
    return summaries


def generate_labels(
    *,
    root: Path,
    json_dir: Path,
    item_dir: Path,
    relic_dir: Path,
    weapon_dir: Path,
    background_dir: Path,
    out_path: Path,
    split: str,
    categories: set[str] | None = None,
) -> tuple[int, list[SkippedLabel], list[UnusedImageSummary], list[SharedIconNameSummary]]:
    selected_categories = categories or {"material", "food", "relic", "weapon"}
    unknown_categories = selected_categories - {"material", "food", "relic", "weapon"}
    if unknown_categories:
        raise ValueError(f"未知类别: {sorted(unknown_categories)}")

    material_json = json_dir / "Material.json"
    relic_json = json_dir / "Reliquary.json"
    weapon_json = json_dir / "Weapon.json"
    for path in [material_json, relic_json, weapon_json]:
        if not path.exists():
            raise FileNotFoundError(f"找不到 JSON 文件: {path}")
    for path, name in [(item_dir, "材料图标目录"), (relic_dir, "圣遗物图标目录"), (weapon_dir, "武器图标目录")]:
        if not path.exists():
            raise FileNotFoundError(f"找不到{name}: {path}")
    if not background_dir.exists():
        raise FileNotFoundError(f"找不到背景图目录: {background_dir}")

    all_rows: list[dict[str, str]] = []
    skipped: list[SkippedLabel] = []
    unused: list[UnusedImageSummary] = []
    if selected_categories & {"material", "food"}:
        material_rows, material_skipped, material_expected = rows_from_materials(
            root=root,
            json_path=material_json,
            item_dir=item_dir,
            background_dir=background_dir,
            split=split,
        )
        all_rows.extend(row for row in material_rows if row["category"] in selected_categories)
        skipped.extend(material_skipped)
        unused.append(summarize_unused_images(item_dir, material_expected, "assets/icons/items"))
    if "relic" in selected_categories:
        relic_rows, relic_skipped, relic_expected = rows_from_relics(
            root=root,
            json_path=relic_json,
            relic_dir=relic_dir,
            background_dir=background_dir,
            split=split,
        )
        all_rows.extend(relic_rows)
        skipped.extend(relic_skipped)
        unused.append(summarize_unused_images(relic_dir, relic_expected, "assets/icons/relics"))
    if "weapon" in selected_categories:
        weapon_rows, weapon_skipped, weapon_expected = rows_from_weapons(
            root=root,
            json_path=weapon_json,
            weapon_dir=weapon_dir,
            background_dir=background_dir,
            split=split,
        )
        all_rows.extend(weapon_rows)
        skipped.extend(weapon_skipped)
        unused.append(summarize_unused_images(weapon_dir, weapon_expected, "assets/icons/weapons"))

    duplicate_variants = sorted({row["variant_id"] for row in all_rows if sum(r["variant_id"] == row["variant_id"] for r in all_rows) > 1})
    if duplicate_variants:
        raise ValueError(f"生成了重复 variant_id: {duplicate_variants[:20]}")

    shared_icon_names = summarize_shared_icon_names(all_rows)
    all_rows.sort(key=lambda row: (row["category"], row["item_class_id"], row["quality_level"], row["variant_id"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows), skipped, unused, shared_icon_names


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="从 item JSON 生成 data/generated/labels.csv")
    parser.add_argument("--json-dir", default=str(root / "data" / "metadata"))
    parser.add_argument("--item-dir", default=str(root / "assets" / "icons" / "items"))
    parser.add_argument("--relic-dir", default=str(root / "assets" / "icons" / "relics"))
    parser.add_argument("--weapon-dir", default=str(root / "assets" / "icons" / "weapons"))
    parser.add_argument("--background-dir", default=str(root / "assets" / "backgrounds"))
    parser.add_argument("--out", default=str(root / "data" / "generated" / "labels.csv"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--category",
        action="append",
        choices=["material", "food", "relic", "weapon"],
        help="只生成指定类别；可重复传入。默认生成全部类别。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    count, skipped, unused, shared_icon_names = generate_labels(
        root=root,
        json_dir=Path(args.json_dir),
        item_dir=Path(args.item_dir),
        relic_dir=Path(args.relic_dir),
        weapon_dir=Path(args.weapon_dir),
        background_dir=Path(args.background_dir),
        out_path=Path(args.out),
        split=args.split,
        categories=set(args.category) if args.category else None,
    )
    known_missing_names = sorted(
        {item.item_name for item in skipped if item.reason == KNOWN_MISSING_MATERIAL_IMAGE_REASON and item.item_name}
    )
    print_skipped_names_warning(KNOWN_MISSING_MATERIAL_IMAGE_REASON, known_missing_names)

    known_missing_relic_names = sorted(
        {item.item_name for item in skipped if item.reason == KNOWN_MISSING_RELIC_IMAGE_REASON and item.item_name}
    )
    print_skipped_names_warning(KNOWN_MISSING_RELIC_IMAGE_REASON, known_missing_relic_names)

    skipped_food_names = sorted(
        {item.item_name for item in skipped if item.reason == SKIPPED_FOOD_NAME_REASON and item.item_name}
    )
    print_skipped_names_warning(SKIPPED_FOOD_NAME_REASON, skipped_food_names)

    skipped_food_item_ids = sorted(
        {f"{item.item_id}:{item.item_name}" for item in skipped if item.reason == SKIPPED_FOOD_ITEM_ID_REASON and item.item_id}
    )
    print_skipped_names_warning(SKIPPED_FOOD_ITEM_ID_REASON, skipped_food_item_ids)

    duplicate_reason_names: dict[str, set[str]] = defaultdict(set)
    for item in skipped:
        if item.reason.startswith(SKIPPED_DUPLICATE_MATERIAL_REASON_PREFIX) and item.item_name:
            duplicate_reason_names[item.reason].add(item.item_name)
    for reason, names in sorted(duplicate_reason_names.items()):
        print_skipped_names_warning(reason, names)

    for summary in shared_icon_names:
        category_name = "食物" if summary.category == "food" else "材料"
        suffix = "，已排除同一食物的普通/奇怪/美味共用" if summary.category == "food" else ""
        print(
            f"警告: {category_name}同图不同名{suffix}: {summary.image_path}；"
            f"名称: {format_inline_names(summary.names)}"
        )

    for item in skipped:
        if item.reason in {
            KNOWN_MISSING_MATERIAL_IMAGE_REASON,
            KNOWN_MISSING_RELIC_IMAGE_REASON,
            SKIPPED_FOOD_NAME_REASON,
            SKIPPED_FOOD_ITEM_ID_REASON,
        } or item.reason.startswith(
            SKIPPED_DUPLICATE_MATERIAL_REASON_PREFIX,
        ):
            continue
        print(
            "警告: 跳过 "
            f"source={item.source_json} item_id={item.item_id} 名称={item.item_name} "
            f"图片={item.icon_name}，原因: {item.reason}"
        )
    for summary in unused:
        if summary.count:
            examples = ", ".join(summary.examples)
            suffix = " ..." if summary.count > len(summary.examples) else ""
            print(f"警告: {summary.folder_name} 未被 JSON 引用的 PNG 数量={summary.count}，示例: {examples}{suffix}")
    print(f"生成行数={count}")
    print(f"跳过行数={len(skipped)}")
    print(f"输出文件={Path(args.out)}")


if __name__ == "__main__":
    main()
