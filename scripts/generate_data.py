#!/usr/bin/env python
"""Regenerate the reproducible synthetic dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paybridge.data.generator import generate_dataset, write_dataset  # noqa: E402


def main() -> int:
    dataset = generate_dataset()
    manifest = write_dataset(dataset)
    print(json.dumps({"summary": dataset.summary(), "row_counts": manifest["row_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
