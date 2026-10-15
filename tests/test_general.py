import math
from pathlib import Path

import pytest

from scripts.build_summary_table import build_rows
from safe_quant.config import load_benchmark_config
from safe_quant.evaluate_general import (
    _extract_summary,
    _validate_protocol as validate_general_protocol,
)
from safe_quant.evaluate_wikitext2 import (
    _perplexity,
    _validate_protocol as validate_wikitext_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_general_config_and_summary() -> None:
    config = load_benchmark_config(ROOT / "configs/benchmarks/general_4task.yaml")
    validate_general_protocol(config)
    raw = {
        "results": {
            task: {"acc,none": value, "acc_norm,none": value + 0.01}
            for task, value in zip(
                ("arc_easy", "arc_challenge", "piqa", "winogrande"),
                (0.8, 0.6, 0.7, 0.5),
            )
        },
        "n-samples": {
            task: {"original": 100, "effective": 100}
            for task in ("arc_easy", "arc_challenge", "piqa", "winogrande")
        },
    }

    summary = _extract_summary(raw, limit=None)

    assert summary["general_4task_avg"] == pytest.approx(0.65)
    assert summary["tasks"]["arc_easy"]["accuracy"] == 0.8
    assert summary["tasks"]["arc_easy"]["accuracy_normalized_diagnostic"] == 0.81
    assert summary["is_full_evaluation"] is True


def test_general_protocol_rejects_scientific_changes() -> None:
    config = load_benchmark_config(ROOT / "configs/benchmarks/general_4task.yaml")
    config["protocol"]["num_fewshot"] = 5

    with pytest.raises(ValueError, match="protocol changed"):
        validate_general_protocol(config)


def test_frozen_wikitext_config_and_token_weighted_perplexity() -> None:
    config = load_benchmark_config(ROOT / "configs/benchmarks/wikitext2.yaml")
    validate_wikitext_protocol(config)

    total_nll = 2 * math.log(2) + math.log(8)
    assert _perplexity(total_nll, 3) == pytest.approx(32 ** (1 / 3))


def test_wikitext_perplexity_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _perplexity(0.0, 0)


def test_final_table_uses_primary_accuracies() -> None:
    general = {
        "wikitext2": {"perplexity": {"bf16": 6.0, "spinquant_w4a4kv4": 8.0}},
        "general_4task": {
            "tasks": {
                task: {"bf16": 0.8, "spinquant_w4a4kv4": 0.7}
                for task in ("arc_easy", "arc_challenge", "piqa", "winogrande")
            },
            "general_4task_avg": {"bf16": 0.8, "spinquant_w4a4kv4": 0.7},
        },
    }
    finguard_bf16 = {"overall": {"accuracy": 0.9}}
    finguard_spinquant = {"overall": {"accuracy": 0.85}}
    wildguard = {
        "metrics": {
            "accuracy": {"bf16": 0.82, "spinquant_w4a4kv4": 0.70}
        }
    }

    rows = build_rows(
        general, finguard_bf16, finguard_spinquant, wildguard
    )

    assert rows[0]["wikitext2_perplexity"] == 6.0
    assert rows[1]["general_4task_avg"] == 0.7
    assert rows[1]["finguard_accuracy"] == 0.85
    assert rows[1]["wildguard_accuracy"] == 0.70
