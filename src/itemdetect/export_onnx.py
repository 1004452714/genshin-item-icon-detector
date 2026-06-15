from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_model_from_checkpoint
from .config import load_config, project_path
from .model import OnnxItemWrapper


def export_onnx(config_path: str | Path, checkpoint_path: str | Path, out_path: str | Path | None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cpu")
    model, _ = load_model_from_checkpoint(checkpoint_path, device)
    wrapper = OnnxItemWrapper(model).eval()
    image_size = cfg["data"].get("image_size", [125, 125])
    width, height = int(image_size[0]), int(image_size[1])
    dummy = torch.randn(1, 3, height, width, dtype=torch.float32)
    output = Path(out_path) if out_path else project_path(cfg, cfg["export"]["onnx_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        output,
        input_names=["input_image"],
        output_names=["embedding"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=int(cfg.get("export", {}).get("opset", 17)),
    )
    print(f"已写入 {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.config, args.checkpoint, args.out)
