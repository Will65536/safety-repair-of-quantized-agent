import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_quant.benchmarks.wildguard import (
    _select_indices,
    load_wildguard_prompts,
    summarize_wildguard_predictions,
)
from safe_quant.config import load_benchmark_config
from safe_quant.evaluate_wildguard import (
    _implementation_validation,
    _validate_benchmark_config,
)
from safe_quant.wildguard_adapter import (
    HARMFUL_CANDIDATE,
    HARMFUL_TOKEN_ID,
    SYSTEM_PROMPT,
    UNHARMFUL_CANDIDATE,
    UNHARMFUL_TOKEN_ID,
    USER_PROMPT_TEMPLATE,
    LlamaWildGuardLikelihoodAdapter,
    prediction_from_margin,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        if text == HARMFUL_CANDIDATE:
            return [HARMFUL_TOKEN_ID]
        if text == UNHARMFUL_CANDIDATE:
            return [UNHARMFUL_TOKEN_ID]
        if text.endswith(HARMFUL_CANDIDATE):
            return [11, 12, HARMFUL_TOKEN_ID]
        if text.endswith(UNHARMFUL_CANDIDATE):
            return [11, 12, UNHARMFUL_TOKEN_ID]
        return [11, 12]

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
        return "rendered-assistant-boundary"


def _score_row(label, prediction, margin, is_score_tie=False):
    yes_logit = 3.0 + margin
    no_logit = 3.0
    normalizer = 5.0
    return {
        "gt_prompt_harmfulness": label,
        "label_id": 1 if label == "harmful" else 0,
        "prediction": prediction,
        "prediction_id": 1 if prediction == "harmful" else 0,
        "prompt_type": "adversarial",
        "prompt_harm_category": "category",
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_log_probability": yes_logit - normalizer,
        "no_log_probability": no_logit - normalizer,
        "log_normalizer": normalizer,
        "harmfulness_margin": margin,
        "is_score_tie": is_score_tie,
        "candidate_boundary": {
            "harmful": {"valid": True},
            "unharmful": {"valid": True},
        },
    }


def test_frozen_config_and_prompt_are_exact() -> None:
    config = load_benchmark_config(
        ROOT / "configs/benchmarks/wildguard_train_validation.yaml"
    )
    _validate_benchmark_config(config)

    assert config["protocol"]["system_prompt"] == SYSTEM_PROMPT
    assert config["protocol"]["user_prompt_template"] == USER_PROMPT_TEMPLATE
    assert config["protocol"]["candidates"] == {
        "harmful": {"text": " Yes", "expected_token_id": 7566},
        "unharmful": {"text": " No", "expected_token_id": 2360},
    }


def test_candidate_tokens_are_single_tokens_at_assistant_boundary() -> None:
    adapter = LlamaWildGuardLikelihoodAdapter(
        model=object(),
        tokenizer=FakeTokenizer(),
        protocol=load_benchmark_config(
            ROOT / "configs/benchmarks/wildguard_train_validation.yaml"
        )["protocol"],
    )

    rendered = adapter.render_prompt("example")
    checks = adapter._validate_assistant_boundary(rendered)

    assert checks["harmful"]["appended_token_ids"] == [HARMFUL_TOKEN_ID]
    assert checks["unharmful"]["appended_token_ids"] == [UNHARMFUL_TOKEN_ID]
    assert all(check["valid"] for check in checks.values())


def test_metrics_use_harmful_as_positive_and_report_margins() -> None:
    rows = [
        _score_row("harmful", "harmful", 2.0),
        _score_row("harmful", "unharmful", -1.0),
        _score_row("unharmful", "harmful", 0.5),
        _score_row("unharmful", "unharmful", -2.0),
    ]

    metrics = summarize_wildguard_predictions(rows)

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["harmful_recall"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["prediction_class_balance"]["harmful"]["count"] == 2
    assert metrics["harmfulness_margin_by_ground_truth"]["harmful"]["mean"] == 0.5


def test_exact_tie_is_primary_unharmful_and_sensitivity_harmful() -> None:
    rows = [
        _score_row("harmful", "unharmful", 0.0, is_score_tie=True),
        _score_row("unharmful", "unharmful", -1.0),
    ]

    metrics = summarize_wildguard_predictions(rows)

    assert metrics["score_tie_count"] == 1
    assert metrics["score_tie_rate"] == 0.5
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 0, "fn": 1, "tp": 0}
    sensitivity = metrics["sensitivity_exact_ties_mapped_to_harmful"]
    assert sensitivity["status"] == "sensitivity analysis; not primary metrics"
    assert sensitivity["metrics"]["confusion_matrix"] == {
        "tn": 1,
        "fp": 0,
        "fn": 0,
        "tp": 1,
    }


def test_v2_primary_threshold_maps_exact_tie_to_unharmful() -> None:
    assert prediction_from_margin(0.25) == "harmful"
    assert prediction_from_margin(0.0) == "unharmful"
    assert prediction_from_margin(-0.25) == "unharmful"


def test_implementation_validation_uses_numerical_tolerance() -> None:
    records = [
        _score_row("harmful", "harmful", 2.0),
        _score_row("unharmful", "unharmful", -2.0),
    ]
    repeated = [dict(row) for row in records]
    approximately_repeated = [dict(row) for row in records]
    approximately_repeated[0]["yes_logit"] += 5e-6
    approximately_repeated[0]["yes_log_probability"] += 5e-6
    approximately_repeated[0]["harmfulness_margin"] += 5e-6

    report = _implementation_validation(
        records,
        repeated,
        approximately_repeated,
        {"reconstruction_atol": 1e-6, "repeat_atol": 1e-5, "repeat_rtol": 1e-5},
    )

    assert report["overall_pass"] is True
    assert report["hard_requirements"]["repeat_predictions_identical"]["pass"]
    assert report["hard_requirements"]["repeat_scores_numerically_close"]["pass"]


def test_validation_selection_is_deterministic_and_balanced() -> None:
    rows = [
        {"prompt_harm_label": label, "adversarial": adversarial}
        for label in ("harmful", "unharmful")
        for adversarial in (False, True)
        for _ in range(70)
    ]
    config = {
        "selection": "stratified_random_without_replacement",
        "selection_seed": 0,
        "strata": ["prompt_harm_label", "adversarial"],
        "samples_per_stratum": 64,
    }

    first, metadata = _select_indices(rows, config)
    second, _ = _select_indices(rows, config)

    assert first == second
    assert len(first) == 256
    assert set(metadata["selected_stratum_counts"].values()) == {64}


def test_loader_uses_cached_hf_token_and_preserves_metadata(monkeypatch) -> None:
    seen = {}

    class FakeDataset:
        def __init__(self, rows, fingerprint="source"):
            self.rows = rows
            self._fingerprint = fingerprint

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

        def filter(self, function):
            return FakeDataset(
                [row for row in self.rows if function(row)], "filtered"
            )

        def select(self, indices):
            return FakeDataset([self.rows[index] for index in indices], "selected")

    source = FakeDataset(
        [
            {
                "prompt": "request",
                "prompt_harm_label": "unharmful",
                "adversarial": False,
                "subcategory": "benign",
                "prompt_harm_agreement": "three-way",
            },
            {
                "prompt": "unlabeled",
                "prompt_harm_label": None,
                "adversarial": True,
                "subcategory": None,
                "prompt_harm_agreement": None,
            },
        ]
    )

    def fake_load_dataset(*args, **kwargs):
        seen.update(kwargs)
        return source

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    rows, metadata = load_wildguard_prompts(
        {
            "dataset": {
                "path": "allenai/wildguardmix",
                "name": "wildguardtest",
                "split": "test",
                "revision": "pinned",
                "filtering": "drop_null_prompt_harm_label",
                "selection": "all",
            }
        }
    )

    assert seen["token"] is True
    assert len(rows) == 1
    assert rows[0]["prompt_type"] == "vanilla"
    assert rows[0]["prompt_harm_category"] == "benign"
    assert metadata["source_num_rows"] == 2
    assert metadata["num_rows_after_prompt_label_filter"] == 1


def test_frozen_protocol_rejects_changes() -> None:
    config = load_benchmark_config(
        ROOT / "configs/benchmarks/wildguard_prompt.yaml"
    )
    config["protocol"]["system_prompt"] = "changed"

    with pytest.raises(ValueError, match="protocol changed"):
        _validate_benchmark_config(config)
