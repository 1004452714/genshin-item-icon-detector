from __future__ import annotations

import argparse
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .checkpoint import load_model_from_checkpoint
from .config import load_config, project_path
from .dataset import ItemDataset, load_labels


def encode_vector(vec: np.ndarray) -> str:
    return base64.b64encode(vec.astype("<f4", copy=False).tobytes()).decode("ascii")


def build_prototypes(config_path: str | Path, checkpoint_path: str | Path, out_path: str | Path | None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    mappings = checkpoint["mappings"]
    df = load_labels(project_path(cfg, cfg["data"]["labels_csv"]))
    samples = int(cfg.get("prototype", {}).get("samples_per_label", 24))
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), ascii=True):
        one = pd.DataFrame([row])
        dataset = ItemDataset(one, cfg["_project_root"], cfg, mappings, train=samples > 1, prototype=True)
        vectors = []
        with torch.no_grad():
            for _ in range(max(1, samples)):
                item = dataset[0]
                image = item["image"].unsqueeze(0).to(device)
                embedding, _, _ = model(image)
                vectors.append(embedding.squeeze(0).detach().cpu().numpy())
        proto = np.mean(np.stack(vectors), axis=0)
        proto = proto / max(np.linalg.norm(proto), 1e-12)
        quality_level = str(row.get("quality_level", row.get("rank_level", "")))
        rows.append(
            {
                "variant_id": row["variant_id"],
                "item_class_id": row["item_class_id"],
                "item_name": row["item_name"],
                "food_base_name": row["food_base_name"],
                "quality_level": quality_level,
                "weapon_state": row["weapon_state"],
                "embedding": encode_vector(proto),
            }
        )

    output = Path(out_path) if out_path else project_path(cfg, cfg["prototype"]["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8")
    print(f"已写入 {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_prototypes(args.config, args.checkpoint, args.out)
