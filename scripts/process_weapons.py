from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COPY_SUFFIX_RE = re.compile(r" \((\d+)\)$")
AWAKEN_SUFFIX = "_Awaken"
STATE_ORDER = ("normal", "awaken")
NORMALIZED_SIZE = (256, 256)
POSITION_EPSILON = 0.25


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    state_stem: str
    weapon_key: str
    state: str
    copy_index: int


@dataclass(frozen=True)
class ImageMetrics:
    alpha_area: float
    opaque_count: int
    center_x: float
    center_y: float


@dataclass(frozen=True)
class StateDecision:
    status: str
    reason: str
    icon_record: ImageRecord | None
    side_record: ImageRecord | None
    review_records: tuple[ImageRecord, ...]
    alpha_ratio: float | None
    opaque_ratio: float | None
    position_delta: float | None


@dataclass
class WeaponGroup:
    weapon_key: str
    records: dict[str, dict[int, ImageRecord]] = field(
        default_factory=lambda: {"normal": {}, "awaken": {}}
    )
    raw_records: list[ImageRecord] = field(default_factory=list)


@dataclass(frozen=True)
class ClassifiedGroup:
    weapon_key: str
    group: WeaponGroup
    status: str
    reason: str
    state_decisions: dict[str, StateDecision]


def strip_copy_suffix(stem: str) -> tuple[str, int]:
    match = COPY_SUFFIX_RE.search(stem)
    if not match:
        return stem, 0
    return stem[: match.start()], int(match.group(1))


def parse_record(path: Path) -> ImageRecord:
    state_stem, copy_index = strip_copy_suffix(path.stem)
    if state_stem.endswith(AWAKEN_SUFFIX):
        weapon_key = state_stem[: -len(AWAKEN_SUFFIX)]
        state = "awaken"
    else:
        weapon_key = state_stem
        state = "normal"
    return ImageRecord(path, state_stem, weapon_key, state, copy_index)


def discover_images(raw_dir: Path) -> list[ImageRecord]:
    return [parse_record(path) for path in sorted(raw_dir.glob("*.png"))]


def build_groups(records: Iterable[ImageRecord]) -> dict[str, WeaponGroup]:
    groups: dict[str, WeaponGroup] = {}
    for record in records:
        group = groups.setdefault(record.weapon_key, WeaponGroup(record.weapon_key))
        group.raw_records.append(record)
        group.records[record.state][record.copy_index] = record
    return groups


def sorted_records(records: Iterable[ImageRecord]) -> list[ImageRecord]:
    return sorted(records, key=lambda record: (record.state, record.copy_index, record.path.name))


def is_processable_group(group: WeaponGroup) -> bool:
    return all(len(group.records[state]) >= 2 for state in STATE_ORDER)


def skip_reason(group: WeaponGroup) -> str:
    details = []
    for state in STATE_ORDER:
        indexes = sorted(group.records[state])
        if len(indexes) < 2:
            details.append(f"{state}={indexes or 'missing'}")
    details.append(f"file_count={len(group.raw_records)}")
    return "not_processable_group: " + ", ".join(details)


def resize_for_metrics(image: Image.Image) -> Image.Image:
    if image.size == NORMALIZED_SIZE:
        return image
    return image.resize(NORMALIZED_SIZE, Image.Resampling.LANCZOS)


def read_metrics(path: Path) -> ImageMetrics:
    with Image.open(path) as source:
        image = resize_for_metrics(source.convert("RGBA"))
        alpha = image.getchannel("A")
        histogram = alpha.histogram()
        width, _height = alpha.size
        total_alpha = sum(value * count for value, count in enumerate(histogram))
        weighted_x = 0.0
        weighted_y = 0.0
        if total_alpha:
            for offset, value in enumerate(alpha.getdata()):
                if value:
                    weighted_x += (offset % width) * value
                    weighted_y += (offset // width) * value
    alpha_area = sum(value * count for value, count in enumerate(histogram)) / 255.0
    opaque_count = sum(histogram[32:])
    center_x = weighted_x / total_alpha if total_alpha else 0.0
    center_y = weighted_y / total_alpha if total_alpha else 0.0
    return ImageMetrics(alpha_area, opaque_count, center_x, center_y)


def ratio_and_larger(left: float, right: float) -> tuple[float, int | None]:
    if left == right:
        return 1.0, None
    if left > right:
        return left / max(right, 1e-9), 0
    return right / max(left, 1e-9), 1


def compare_metrics(
    left: ImageMetrics,
    right: ImageMetrics,
) -> tuple[float, int | None, float, int | None]:
    alpha_ratio, alpha_larger = ratio_and_larger(left.alpha_area, right.alpha_area)
    opaque_ratio, opaque_larger = ratio_and_larger(float(left.opaque_count), float(right.opaque_count))
    return alpha_ratio, alpha_larger, opaque_ratio, opaque_larger


def position_icon_index(left: ImageMetrics, right: ImageMetrics) -> tuple[int, float, str]:
    delta = (right.center_x + right.center_y) - (left.center_x + left.center_y)
    if delta > POSITION_EPSILON:
        return 1, delta, "position_right_down"
    if delta < -POSITION_EPSILON:
        return 0, delta, "position_right_down"
    return 0, delta, "position_tie_name_rule"


def review_files_decision(
    records: Iterable[ImageRecord],
    reason: str,
    alpha_ratio: float | None = None,
    opaque_ratio: float | None = None,
    position_delta: float | None = None,
) -> StateDecision:
    return StateDecision(
        status="review_files",
        reason=reason,
        icon_record=None,
        side_record=None,
        review_records=tuple(sorted_records(records)),
        alpha_ratio=alpha_ratio,
        opaque_ratio=opaque_ratio,
        position_delta=position_delta,
    )


def selected_state_decision(
    status: str,
    reason: str,
    icon_record: ImageRecord,
    side_record: ImageRecord,
    alpha_ratio: float | None,
    opaque_ratio: float | None,
    position_delta: float | None = None,
) -> StateDecision:
    return StateDecision(
        status=status,
        reason=reason,
        icon_record=icon_record,
        side_record=side_record,
        review_records=(),
        alpha_ratio=alpha_ratio,
        opaque_ratio=opaque_ratio,
        position_delta=position_delta,
    )


def decide_two_records(
    records: list[ImageRecord],
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> StateDecision:
    left_record, right_record = records[:2]
    left = metrics[left_record.path]
    right = metrics[right_record.path]
    alpha_ratio, alpha_larger, opaque_ratio, opaque_larger = compare_metrics(left, right)

    if alpha_larger is None or opaque_larger is None:
        if alpha_larger is None and opaque_larger is None:
            return selected_state_decision(
                "review",
                "ratio_tie_name_rule",
                left_record,
                right_record,
                alpha_ratio,
                opaque_ratio,
                0.0,
            )
        return review_files_decision(records, "metric_tie", alpha_ratio, opaque_ratio)

    if alpha_larger != opaque_larger:
        return review_files_decision(records, "metric_conflict", alpha_ratio, opaque_ratio)

    if min(alpha_ratio, opaque_ratio) < min_ratio:
        icon_index, position_delta, position_reason = position_icon_index(left, right)
        icon_record = records[icon_index]
        side_record = records[1 - icon_index]
        return selected_state_decision(
            "review",
            f"low_ratio<{min_ratio:g}:{position_reason}",
            icon_record,
            side_record,
            alpha_ratio,
            opaque_ratio,
            position_delta,
        )

    icon_record = records[alpha_larger]
    side_record = records[1 - alpha_larger]
    return selected_state_decision("auto", "ok", icon_record, side_record, alpha_ratio, opaque_ratio)


def metric_spread(records: list[ImageRecord], metrics: dict[Path, ImageMetrics], field: str) -> float:
    values = [float(getattr(metrics[record.path], field)) for record in records]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return 1.0
    return maximum / max(minimum, 1e-9)


def same_scale(
    records: list[ImageRecord],
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> tuple[bool, float, float]:
    alpha_spread = metric_spread(records, metrics, "alpha_area")
    opaque_spread = metric_spread(records, metrics, "opaque_count")
    return max(alpha_spread, opaque_spread) < min_ratio, alpha_spread, opaque_spread


def cluster_records_by_scale(
    records: list[ImageRecord],
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> list[list[ImageRecord]]:
    clusters: list[list[ImageRecord]] = []
    for record in sorted(records, key=lambda item: metrics[item.path].alpha_area):
        record_metrics = metrics[record.path]
        for cluster in clusters:
            cluster_metrics = metrics[cluster[0].path]
            alpha_ratio, _alpha_larger, opaque_ratio, _opaque_larger = compare_metrics(cluster_metrics, record_metrics)
            if max(alpha_ratio, opaque_ratio) < min_ratio:
                cluster.append(record)
                break
        else:
            clusters.append([record])
    return [sorted_records(cluster) for cluster in clusters]


def average_cluster_metrics(records: list[ImageRecord], metrics: dict[Path, ImageMetrics]) -> ImageMetrics:
    count = len(records)
    return ImageMetrics(
        alpha_area=sum(metrics[record.path].alpha_area for record in records) / count,
        opaque_count=round(sum(metrics[record.path].opaque_count for record in records) / count),
        center_x=sum(metrics[record.path].center_x for record in records) / count,
        center_y=sum(metrics[record.path].center_y for record in records) / count,
    )


def decide_many_records(
    records: list[ImageRecord],
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> StateDecision:
    is_same_scale, alpha_spread, opaque_spread = same_scale(records, metrics, min_ratio)
    if is_same_scale:
        return selected_state_decision(
            "review",
            "same_scale_name_rule",
            records[0],
            records[1],
            alpha_spread,
            opaque_spread,
            0.0,
        )

    clusters = cluster_records_by_scale(records, metrics, min_ratio)
    if len(clusters) != 2:
        return review_files_decision(records, f"scale_cluster_count={len(clusters)}", alpha_spread, opaque_spread)

    left_cluster, right_cluster = clusters
    left_metrics = average_cluster_metrics(left_cluster, metrics)
    right_metrics = average_cluster_metrics(right_cluster, metrics)
    alpha_ratio, alpha_larger, opaque_ratio, opaque_larger = compare_metrics(left_metrics, right_metrics)

    if alpha_larger is None or opaque_larger is None:
        return review_files_decision(records, "metric_tie", alpha_ratio, opaque_ratio)
    if alpha_larger != opaque_larger:
        return review_files_decision(records, "metric_conflict", alpha_ratio, opaque_ratio)

    icon_cluster = clusters[alpha_larger]
    side_cluster = clusters[1 - alpha_larger]
    return selected_state_decision("auto", "ok", icon_cluster[0], side_cluster[0], alpha_ratio, opaque_ratio)


def decide_state(
    records: Iterable[ImageRecord],
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> StateDecision:
    sorted_state_records = sorted_records(records)
    if len(sorted_state_records) < 2:
        return review_files_decision(sorted_state_records, "not_enough_state_files")
    if len(sorted_state_records) == 2:
        return decide_two_records(sorted_state_records, metrics, min_ratio)
    return decide_many_records(sorted_state_records, metrics, min_ratio)


def classify_group(
    group: WeaponGroup,
    metrics: dict[Path, ImageMetrics],
    min_ratio: float,
) -> ClassifiedGroup:
    state_decisions = {
        state: decide_state(group.records[state].values(), metrics, min_ratio)
        for state in STATE_ORDER
    }
    invalid_reasons = [
        f"{state}:{decision.reason}"
        for state, decision in state_decisions.items()
        if decision.status == "review_files"
    ]
    if invalid_reasons:
        return ClassifiedGroup(group.weapon_key, group, "invalid", "; ".join(invalid_reasons), state_decisions)

    review_reasons = [
        f"{state}:{decision.reason}"
        for state, decision in state_decisions.items()
        if decision.status == "review"
    ]
    if review_reasons:
        return ClassifiedGroup(group.weapon_key, group, "review", "; ".join(review_reasons), state_decisions)

    return ClassifiedGroup(group.weapon_key, group, "auto", "ok", state_decisions)


def configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def ensure_workspace(processed_dir: Path, dry_run: bool) -> dict[str, Path]:
    subdirs = {
        "icon": processed_dir / "icon",
        "side": processed_dir / "side",
        "review": processed_dir / "review",
        "invalid": processed_dir / "invalid",
    }
    for directory in subdirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        if dry_run:
            continue
        for path in directory.iterdir():
            if path.name == ".gitkeep":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    return subdirs


def add_suffix_name(record: ImageRecord, suffix: str) -> str:
    return f"{record.state_stem}_{suffix}{record.path.suffix}"


def copy_record(record: ImageRecord, target_path: Path, dry_run: bool, status: str, reason: str) -> None:
    logging.info("%s%s -> %s [%s] %s", "[dry-run] " if dry_run else "", record.path, target_path, status, reason)
    if dry_run:
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(record.path, target_path)


def copy_selected_group(
    classified: ClassifiedGroup,
    subdirs: dict[str, Path],
    dry_run: bool,
) -> list[str]:
    outputs: list[str] = []
    for state in STATE_ORDER:
        decision = classified.state_decisions[state]
        if decision.icon_record is not None:
            target_path = subdirs["icon"] / add_suffix_name(decision.icon_record, "ICON")
            copy_record(decision.icon_record, target_path, dry_run, classified.status, classified.reason)
            outputs.append(target_path.name)
        if decision.side_record is not None:
            target_path = subdirs["side"] / add_suffix_name(decision.side_record, "SIDE")
            copy_record(decision.side_record, target_path, dry_run, classified.status, classified.reason)
    return outputs


def copy_review_group(
    classified: ClassifiedGroup,
    subdirs: dict[str, Path],
    dry_run: bool,
) -> list[str]:
    outputs: list[str] = []
    for record in sorted_records(classified.group.raw_records):
        target_path = subdirs["review"] / add_suffix_name(record, "review")
        copy_record(record, target_path, dry_run, classified.status, classified.reason)
        outputs.append(target_path.name)
    return outputs


def copy_invalid_group(
    group: WeaponGroup,
    reason: str,
    subdirs: dict[str, Path],
    dry_run: bool,
) -> list[str]:
    outputs: list[str] = []
    for record in sorted_records(group.raw_records):
        target_path = subdirs["invalid"] / record.path.name
        copy_record(record, target_path, dry_run, "invalid", reason)
        outputs.append(target_path.name)
    return outputs


def log_name_list(title: str, names: list[str]) -> None:
    logging.info("%s 总数=%d", title, len(names))
    for name in names:
        logging.info("  - %s", name)


def strip_icon_suffix(path: Path) -> str:
    stem = path.stem
    suffix = "_ICON"
    if not stem.endswith(suffix):
        raise ValueError(f"文件名不是 _ICON 后缀: {path.name}")
    return stem[: -len(suffix)] + path.suffix


def apply_icon_files(processed_dir: Path, train_dir: Path, dry_run: bool) -> int:
    source_dirs = [processed_dir / "icon", processed_dir / "review"]
    icon_files: list[Path] = []
    for source_dir in source_dirs:
        if source_dir.exists():
            icon_files.extend(sorted(source_dir.glob("*_ICON.png")))

    train_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for icon_file in icon_files:
        target_path = train_dir / strip_icon_suffix(icon_file)
        logging.info("%sapply %s -> %s", "[dry-run] " if dry_run else "", icon_file, target_path)
        if not dry_run:
            shutil.copy2(icon_file, target_path)
        copied += 1
    logging.info("复制到训练目录数量=%d", copied)
    return copied


def process(args: argparse.Namespace) -> int:
    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    train_dir = args.train_dir.resolve()
    log_path = args.log_file.resolve() if args.log_file else processed_dir / "process.log"

    raw_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path)
    subdirs = ensure_workspace(processed_dir, args.dry_run)

    records = discover_images(raw_dir)
    groups = build_groups(records)
    metrics: dict[Path, ImageMetrics] = {}

    logging.info("原始目录: %s", raw_dir)
    logging.info("处理目录: %s", processed_dir)
    logging.info("训练目录: %s", train_dir)
    logging.info("原始 PNG 数量=%d", len(records))

    if not records:
        logging.info("没有找到 PNG 文件。")
        return 0

    invalid_names: list[str] = []
    review_names: list[str] = []
    auto_icon_names: list[str] = []
    processable_groups: list[WeaponGroup] = []

    for group in sorted(groups.values(), key=lambda item: item.weapon_key):
        if is_processable_group(group):
            processable_groups.append(group)
            continue
        reason = skip_reason(group)
        logging.info("不符合条件: %s %s", group.weapon_key, reason)
        invalid_names.extend(copy_invalid_group(group, reason, subdirs, args.dry_run))

    for group in processable_groups:
        for state in STATE_ORDER:
            for record in group.records[state].values():
                metrics[record.path] = read_metrics(record.path)

    classified_groups = [
        classify_group(group, metrics, args.min_ratio)
        for group in processable_groups
    ]

    auto_count = 0
    review_count = 0
    invalid_count = len(invalid_names)
    for classified in classified_groups:
        logging.info("分类: %s status=%s reason=%s", classified.weapon_key, classified.status, classified.reason)
        if classified.status == "auto":
            auto_count += 1
            auto_icon_names.extend(copy_selected_group(classified, subdirs, args.dry_run))
        elif classified.status == "review":
            review_count += 1
            review_names.extend(copy_review_group(classified, subdirs, args.dry_run))
        else:
            invalid_count += len(classified.group.raw_records)
            invalid_names.extend(copy_invalid_group(classified.group, classified.reason, subdirs, args.dry_run))

    log_name_list("不符合条件文件", invalid_names)
    log_name_list("需要 review 文件", review_names)
    logging.info("自动 ICON 文件数量=%d", len(auto_icon_names))
    for name in auto_icon_names:
        logging.info("  - %s", name)
    logging.info(
        "汇总: images=%d groups=%d processable=%d auto_groups=%d review_groups=%d invalid_files=%d review_files=%d",
        len(records),
        len(groups),
        len(processable_groups),
        auto_count,
        review_count,
        invalid_count,
        len(review_names),
    )
    logging.info("日志文件: %s", log_path)

    print("")
    print(f"不符合条件文件总数={len(invalid_names)}")
    for name in invalid_names:
        print(f"  - {name}")
    print(f"需要 review 文件总数={len(review_names)}")
    for name in review_names:
        print(f"  - {name}")
    print(f"日志文件: {log_path}")

    if args.dry_run:
        print("dry-run 模式，不会复制到训练素材目录。")
        return 0

    print("")
    print("请手动复查并将正确的图标重命名为 _ICON 后缀格式，完成后输入 APPLY 继续复制到训练素材目录。")
    confirmation = input("输入 APPLY 继续，其他输入将跳过复制: ").strip()
    if confirmation != "APPLY":
        logging.info("用户未输入 APPLY，跳过复制到训练目录。")
        print("已跳过复制到训练素材目录。")
        return 0

    copied = apply_icon_files(processed_dir, train_dir, args.dry_run)
    print(f"已复制到训练素材目录: {copied} 个文件")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="处理原始武器图标，区分 ICON/SIDE/review/invalid。")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "assets" / "raw" / "weapons")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "assets" / "processed" / "weapons")
    parser.add_argument("--train-dir", type=Path, default=ROOT / "assets" / "icons" / "weapons")
    parser.add_argument("--min-ratio", type=float, default=1.04)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", type=Path, help="默认写入 processed-dir/process.log")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return process(args)
    except Exception:
        logging.exception("处理武器图标失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
