#!/usr/bin/env python
"""Train and persist the FraudLens model bundle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paybridge.data.loader import load_transactions  # noqa: E402
from paybridge.fraud.model import save_bundle, train_models  # noqa: E402


def main() -> int:
    df = load_transactions()
    bundle, report = train_models(df)
    path = save_bundle(bundle)
    print(f"Selected model : {report.selected_model}")
    print(f"Threshold      : {report.threshold:.4f} ({report.threshold_objective})")
    print(f"Saved bundle   : {path}")
    print(json.dumps(report.split_summary, indent=2))
    for result in report.results:
        t = result["test"]
        print(
            f"  {result['name']:<24} selected={result['selected']!s:<5} "
            f"test P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
            f"PR-AUC={t['pr_auc']:.3f} FPR={t['false_positive_rate']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
