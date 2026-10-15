import json
from pathlib import Path

from safe_quant.metrics import compute_metrics, summarize_predictions


def test_binary_metrics() -> None:
    rows = [
        {"label_id": 0, "prediction_id": 0, "truncated": False},
        {"label_id": 0, "prediction_id": 1, "truncated": True},
        {"label_id": 1, "prediction_id": 0, "truncated": False},
        {"label_id": 1, "prediction_id": 1, "truncated": False},
    ]

    metrics = compute_metrics(rows)

    assert metrics["accuracy"] == 0.5
    assert metrics["attack_precision"] == 0.5
    assert metrics["attack_recall"] == 0.5
    assert metrics["attack_f1"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["num_truncated"] == 1


def test_summary_groups_categories(tmp_path: Path) -> None:
    rows = [
        {
            "label_id": 0,
            "prediction_id": 0,
            "truncated": False,
            "metadata": {"category": "benign"},
        },
        {
            "label_id": 1,
            "prediction_id": 1,
            "truncated": False,
            "metadata": {"category": "injection"},
        },
    ]
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    summary = summarize_predictions(path)

    assert summary["overall"]["accuracy"] == 1.0
    assert set(summary["by_category"]) == {"benign", "injection"}
