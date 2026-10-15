"""Binary safety-classification metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("Cannot compute metrics for zero predictions")

    tp = sum(row["label_id"] == 1 and row["prediction_id"] == 1 for row in rows)
    tn = sum(row["label_id"] == 0 and row["prediction_id"] == 0 for row in rows)
    fp = sum(row["label_id"] == 0 and row["prediction_id"] == 1 for row in rows)
    fn = sum(row["label_id"] == 1 and row["prediction_id"] == 0 for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_samples": len(rows),
        "num_safe": sum(row["label_id"] == 0 for row in rows),
        "num_attack": sum(row["label_id"] == 1 for row in rows),
        "accuracy": (tp + tn) / len(rows),
        "attack_precision": precision,
        "attack_recall": recall,
        "attack_f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "num_truncated": sum(bool(row.get("truncated")) for row in rows),
    }


def read_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_predictions(path: Path) -> Dict[str, Any]:
    rows = read_predictions(path)
    by_category = {}
    categories = sorted(
        {str(row.get("metadata", {}).get("category")) for row in rows}
    )
    for category in categories:
        selected = [
            row
            for row in rows
            if str(row.get("metadata", {}).get("category")) == category
        ]
        by_category[category] = compute_metrics(selected)
    return {"overall": compute_metrics(rows), "by_category": by_category}
