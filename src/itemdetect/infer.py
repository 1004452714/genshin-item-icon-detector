from __future__ import annotations

import argparse
import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from .config import load_config


def decode_vector(text: str) -> np.ndarray:
    vec = np.frombuffer(base64.b64decode(text), dtype="<f4").astype(np.float32)
    return vec / max(np.linalg.norm(vec), 1e-12)


def preprocess_image(path: str | Path, cfg: dict[str, Any]) -> np.ndarray:
    image_size = cfg["data"].get("image_size", [125, 125])
    width, height = int(image_size[0]), int(image_size[1])
    image = np.asarray(Image.open(path).convert("RGB"))
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    tensor = image.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)
    norm = cfg.get("normalization", {})
    mean = np.asarray(norm.get("mean", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(norm.get("std", [0.5, 0.5, 0.5]), dtype=np.float32).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor[None, ...].astype(np.float32)


def aggregate_topk_by_class(
    df: pd.DataFrame,
    scores: np.ndarray,
    top_k: int,
) -> list[tuple[int, float]]:
    order = np.argsort(-scores)
    seen: set[str] = set()
    results: list[tuple[int, float]] = []
    for idx in order:
        row = df.iloc[int(idx)]
        class_id = str(row["item_class_id"])
        if class_id in seen:
            continue
        seen.add(class_id)
        results.append((int(idx), float(scores[int(idx)])))
        if len(results) >= top_k:
            break
    return results


class InferenceEngine:
    def __init__(
        self,
        *,
        config_path: str | Path,
        model_path: str | Path,
        prototypes_path: str | Path,
        provider: str,
    ) -> None:
        self.cfg = load_config(config_path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.df = pd.read_csv(prototypes_path, dtype=str).fillna("")
        if "quality_level" not in self.df.columns and "rank_level" in self.df.columns:
            self.df["quality_level"] = self.df["rank_level"]
        if "allowed_quality_levels" not in self.df.columns and "allowed_rank_levels" in self.df.columns:
            self.df["allowed_quality_levels"] = self.df["allowed_rank_levels"]
        for column in ["variant_id", "item_class_id", "item_name", "food_base_name", "quality_level", "weapon_state"]:
            if column not in self.df.columns:
                self.df[column] = ""
        self.matrix = np.stack([decode_vector(x) for x in self.df["embedding"]])

    def run(self, image_path: str | Path, top_k: int) -> list[dict[str, str | float]]:
        outputs = self.session.run(None, {self.input_name: preprocess_image(image_path, self.cfg)})
        embedding = outputs[0][0].astype(np.float32)
        embedding = embedding / max(np.linalg.norm(embedding), 1e-12)
        scores = self.matrix @ embedding

        top = aggregate_topk_by_class(self.df, scores, max(1, int(top_k)))
        rows: list[dict[str, str | float]] = []
        for idx, score in top:
            row = self.df.iloc[idx]
            rows.append(
                {
                    "score": score,
                    "variant_id": str(row.get("variant_id", "")),
                    "item_class_id": str(row.get("item_class_id", "")),
                    "item_name": str(row.get("item_name", "")),
                    "quality_level": str(row.get("quality_level", "")),
                    "food_base_name": str(row.get("food_base_name", "")),
                    "weapon_state": str(row.get("weapon_state", "")),
                }
            )
        return rows


def top1_top2_gap(rows: list[dict[str, str | float]]) -> float:
    if len(rows) < 2:
        return 0.0
    return float(rows[0]["score"]) - float(rows[1]["score"])


def print_inference_rows(rows: list[dict[str, str | float]], elapsed_ms: float | None = None) -> None:
    gap = top1_top2_gap(rows)
    if elapsed_ms is None:
        print(f"top1_top2差距={gap:.4f}")
    else:
        print(f"推理耗时={elapsed_ms:.2f}ms top1_top2差距={gap:.4f}")
    for position, row in enumerate(rows, start=1):
        print(
            f"Top{position} "
            f"分数={float(row['score']):.4f} "
            f"名称={row['item_name']} "
            f"食物基础名={row['food_base_name']} "
            f"品质={row['quality_level']} "
            f"武器状态={row['weapon_state']} "
            f"item_class_id={row['item_class_id']} "
            f"variant_id={row['variant_id']}"
        )


def run_inference(
    *,
    config_path: str | Path,
    model_path: str | Path,
    prototypes_path: str | Path,
    image_path: str | Path,
    top_k: int,
    provider: str,
) -> list[dict[str, str | float]]:
    engine = InferenceEngine(
        config_path=config_path,
        model_path=model_path,
        prototypes_path=prototypes_path,
        provider=provider,
    )
    return engine.run(image_path, top_k)


def infer(
    config_path: str | Path,
    model_path: str | Path,
    prototypes_path: str | Path,
    image_path: str | Path,
    top_k: int,
    provider: str,
) -> None:
    engine = InferenceEngine(
        config_path=config_path,
        model_path=model_path,
        prototypes_path=prototypes_path,
        provider=provider,
    )
    start = time.perf_counter()
    rows = engine.run(image_path, top_k)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    print_inference_rows(rows, elapsed_ms=elapsed_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infer(args.config, args.model, args.prototypes, args.image, args.top_k, args.provider)
