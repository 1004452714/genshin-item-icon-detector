from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.infer import infer, parse_args


if __name__ == "__main__":
    args = parse_args()
    try:
        infer(args.config, args.model, args.prototypes, args.image, args.top_k, args.provider)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from None
