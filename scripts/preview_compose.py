from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.config import load_config, project_path
from itemdetect.dataset import ItemDataset, load_labels, make_mappings


def safe_name(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text)[:120]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成物品图标与背景合成预览")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--out-dir", default="outputs/previews")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--train-augment", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    df = load_labels(project_path(cfg, cfg["data"]["labels_csv"]))
    mappings = make_mappings(df)
    dataset = ItemDataset(
        df.head(args.count),
        cfg["_project_root"],
        cfg,
        mappings,
        train=args.train_augment,
        seed=int(cfg["train"].get("seed", 42)),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(dataset)):
        row = df.iloc[idx]
        image = dataset.render_row(row)
        out_path = out_dir / f"{idx:03d}_{safe_name(row['item_name'])}_{safe_name(row['variant_id'])}.png"
        Image.fromarray(image).save(out_path)
        print(f"已写入: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from None
