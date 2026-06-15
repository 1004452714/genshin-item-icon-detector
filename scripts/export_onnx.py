from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.export_onnx import export_onnx, parse_args


if __name__ == "__main__":
    args = parse_args()
    try:
        export_onnx(args.config, args.checkpoint, args.out)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"错误: {exc}") from None
