from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.infer import InferenceEngine, print_inference_rows  # noqa: E402


def collect_images(image_args: list[str] | None, image_dir: str) -> list[Path]:
    if image_args:
        images = [Path(value) for value in image_args]
    else:
        root = Path(image_dir)
        images = sorted(root.glob("*.png"), key=lambda path: path.name)

    missing = [str(path) for path in images if not path.exists()]
    if missing:
        raise FileNotFoundError("找不到图片: " + ", ".join(missing))
    if not images:
        raise ValueError("没有找到 PNG 图片")
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量测试物品图标识别结果")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--image", action="append", help="单张图片路径，可重复传入")
    parser.add_argument("--image-dir", default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = collect_images(args.image, args.image_dir)
    engine = InferenceEngine(
        config_path=args.config,
        model_path=args.model,
        prototypes_path=args.prototypes,
        provider=args.provider,
    )

    total_ms = 0.0
    for image_path in images:
        print("")
        print(f"==== {image_path.name} ====")
        start = time.perf_counter()
        rows = engine.run(image_path, args.top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        total_ms += elapsed_ms
        print_inference_rows(rows, elapsed_ms=elapsed_ms)

    average_ms = total_ms / max(len(images), 1)
    print("")
    print("==== 汇总 ====")
    print(f"图片总数={len(images)}")
    print(f"总推理耗时={total_ms:.2f}ms")
    print(f"平均耗时={average_ms:.2f}ms/张")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from None
