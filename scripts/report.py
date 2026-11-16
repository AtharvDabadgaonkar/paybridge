#!/usr/bin/env python
"""Print every headline figure PayBridge claims, recomputed from the data.

Run this to verify (or refresh) any number quoted in the README or docs:

    .venv/bin/python scripts/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from paybridge.analytics import kpis  # noqa: E402
from paybridge.compliance.onboarding import apply_overrides, compliance_kpis  # noqa: E402
from paybridge.data.loader import (  # noqa: E402
    load_compliance_cases, load_manifest, load_product_events, load_transactions,
)
from paybridge.fraud import evaluation as E  # noqa: E402
from paybridge.fraud.model import ensure_model  # noqa: E402
from paybridge.payments.pricing import savings_vs_swift  # noqa: E402
from paybridge.product import metrics as PM  # noqa: E402
from paybridge.program.program import qa_summary  # noqa: E402
from paybridge.program.shadow_mode import run_shadow_mode  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> int:
    df = load_transactions()
    bundle = ensure_model()
    manifest = load_manifest()

    section("DATASET (synthetic, seeded)")
    print(json.dumps(manifest.get("row_counts", {}), indent=2))
    print(f"seed                : {manifest.get('seed')}")
    print(f"window              : {manifest.get('window', {}).get('days')} days ending "
          f"{manifest.get('window', {}).get('end', '')[:10]}")
    print(f"realised fraud rate : {df['fraud_label'].mean() * 100:.3f}%")

    section("PORTFOLIO KPIs (whole dataset)")
    k = kpis.executive_kpis(df)
    for key in ("total_volume_usd", "transaction_count", "success_rate_pct",
                "avg_transaction_fee_usd", "avg_cost_pct_of_amount", "avg_settlement_hours",
                "modelled_savings_usd", "modelled_savings_pct", "fraud_alert_rate_pct",
                "contribution_margin_usd", "contribution_margin_pct"):
        print(f"{key:<28}: {k[key]:,.4f}")

    section("MODELLED PAYOUT COST SAVING vs SWIFT (documented assumptions)")
    for amount in (500, 5_000, 50_000):
        row = savings_vs_swift("MX", float(amount))
        parts = " | ".join(f"{rail} {v['savings_pct']:.1f}%" for rail, v in row.items())
        print(f"US-MX  ${amount:>7,}: {parts}")

    section("FRAUDLENS MODEL SELECTION (validation PR-AUC)")
    report = bundle.report
    print(f"selected model : {report.selected_model} ({report.selection_criterion})")
    print(f"threshold      : {report.threshold:.4f} tuned by {report.threshold_objective}")
    print(json.dumps(report.split_summary, indent=2))
    for result in report.results:
        t, v = result["test"], result["validation"]
        print(f"  {result['name']:<24} selected={str(result['selected']):<5} "
              f"val PR-AUC={v['pr_auc']:.3f} | test P={t['precision']:.3f} R={t['recall']:.3f} "
              f"F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f} FPR={t['false_positive_rate']:.4f}")

    section("APPROACH COMPARISON (held-out chronological test split)")
    test = E.test_split(df)
    print(f"test rows={len(test):,}  labelled fraud={int(test['fraud_label'].sum())}")
    frame = E.comparison_frame(E.compare_approaches(test, bundle))
    print(frame[["approach", "threshold", "precision", "recall", "f1", "pr_auc",
                 "false_positive_rate", "alerts", "review_volume_pct", "total_cost_usd"]]
          .round(4).to_string(index=False))

    section("FALSE POSITIVES AT MATCHED RECALL")
    matched = E.matched_point_comparison(test, bundle)
    base = matched["baseline"]
    print(f"baseline (rules only): recall={base['recall']:.4f} FP={base['false_positives']} "
          f"TP={base['true_positives']}")
    for comp in matched["comparisons"]:
        print(f"  {comp['approach']:<22} threshold={comp['matched_threshold']:.4f} "
              f"recall={comp['recall']:.4f} FP={comp['false_positives']} "
              f"reduction={comp['false_positive_reduction_pct']:+.2f}%")

    section("SHADOW-MODE VALIDATION (measured)")
    shadow = run_shadow_mode(df, bundle, sample_size=4000, latency_sample=150)
    print(f"replayed={shadow.transactions_replayed:,}  criteria met={shadow.criteria_met}/"
          f"{len(shadow.criteria)}  overall pass={shadow.passed}")
    print(shadow.criteria_frame()[["label", "direction", "target", "measured", "met"]]
          .to_string(index=False))

    section("COMPLIANCE OPERATIONS (simulated workflow)")
    cases = apply_overrides(load_compliance_cases())
    for key, value in compliance_kpis(cases).items():
        print(f"{key:<34}: {value:,.3f}" if isinstance(value, float) else f"{key:<34}: {value}")

    section("PRODUCT ANALYTICS")
    events = load_product_events()
    print(json.dumps(PM.activation_metrics(events), indent=2, default=str))
    print(PM.feature_adoption(events).round(2).to_string(index=False))
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in PM.payment_product_kpis(df).items()}, indent=2))

    section("QUALITY ASSURANCE (last recorded pytest run)")
    print(json.dumps(qa_summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
