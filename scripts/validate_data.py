from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itemdetect.validate_data import parse_args, validate_data


if __name__ == "__main__":
    args = parse_args()
    validate_data(args.config)
