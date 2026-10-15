"""Evaluate prompt harmfulness with the frozen WildGuardTest Llama adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from safe_quant.benchmarks.wildguard import (
    SAFETY_EVAL_REVISION,
    SAFETY_EVAL_TASK_URL,
    load_wildguard_prompts,
    summarize_wildguard_predictions,
)
from safe_quant.config import load_benchmark_config, load_model_config
from safe_quant.model_loaders import load_model
from safe_quant.wildguard_adapter import (
    HARMFUL_CANDIDATE,
    HARMFUL_TOKEN_ID,
    SYSTEM_PROMPT,
    UNHARMFUL_CANDIDATE,
    UNHARMFUL_TOKEN_ID,
    USER_PROMPT_TEMPLATE,
    LlamaWildGuardLikelihoodAdapter,
    validate_frozen_protocol,
)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _package_versions(names: Iterable[str]) -> Dict[str, Any]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "WildGuard model evaluation is GPU work and must run inside a "
            "Slurm job on Delta"
        )


def _batches(rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _validate_benchmark_config(config: Dict[str, Any]) -> None:
    if config["loader"] != "wildguard_llama_likelihood":
        raise ValueError(
            "WildGuard evaluator requires loader=wildguard_llama_likelihood"
        )
    validate_frozen_protocol(config["protocol"])
    dataset = config["dataset"]
    if dataset.get("path") != "allenai/wildguardmix":
        raise ValueError("Frozen WildGuard dataset path changed")
    if dataset.get("revision") != "d29c47f41c8b51348b5c8e8c81c039b3132b66d1":
        raise ValueError("Frozen WildGuard dataset revision changed")
    reference = config.get("reference", {})
    if reference.get("revision") != SAFETY_EVAL_REVISION:
        raise ValueError("Frozen safety-eval reference revision changed")
    if "validation" in config:
        if config["protocol"]["use_cache"]:
            raise ValueError("WildGuardTrain validation retains the original uncached protocol")
        expected_dataset = {
            "name": "wildguardtrain",
            "split": "train",
            "filtering": "drop_null_prompt_harm_label",
            "selection": "stratified_random_without_replacement",
            "selection_seed": 0,
            "strata": ["prompt_harm_label", "adversarial"],
            "samples_per_stratum": 64,
        }
        expected_validation = {
            "repeat_examples": 8,
            "repeat_atol": 1e-5,
            "repeat_rtol": 1e-5,
            "reconstruction_atol": 1e-6,
        }
        if any(dataset.get(key) != value for key, value in expected_dataset.items()):
            raise ValueError("Frozen WildGuardTrain validation selection changed")
        if config["validation"] != expected_validation:
            raise ValueError("Frozen WildGuardTrain validation checks changed")
    else:
        expected_test = {
            "name": "wildguardtest",
            "split": "test",
            "filtering": "drop_null_prompt_harm_label",
            "selection": "all",
        }
        if any(dataset.get(key) != value for key, value in expected_test.items()):
            raise ValueError("Frozen WildGuardTest evaluation selection changed")


def _require_bf16_validation_model(
    model_config: Dict[str, Any], benchmark_config: Dict[str, Any]
) -> None:
    if "validation" not in benchmark_config:
        return
    precision = model_config.get("precision", {})
    if model_config.get("loader") != "transformers" or precision.get("weights") != "bf16":
        raise ValueError("WildGuardTrain validation preflight is frozen to BF16")


def _all_close(
    first: Sequence[Dict[str, Any]],
    second: Sequence[Dict[str, Any]],
    fields: Sequence[str],
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    field_results = {}
    overall = True
    for field in fields:
        differences = [
            abs(float(left[field]) - float(right[field]))
            for left, right in zip(first, second)
        ]
        close = all(
            math.isclose(
                float(left[field]), float(right[field]), rel_tol=rtol, abs_tol=atol
            )
            for left, right in zip(first, second)
        )
        field_results[field] = {
            "pass": close,
            "max_absolute_difference": max(differences, default=0.0),
        }
        overall = overall and close
    return {"pass": overall, "fields": field_results, "atol": atol, "rtol": rtol}


def _implementation_validation(
    records: Sequence[Dict[str, Any]],
    repeated_first: Sequence[Dict[str, Any]],
    repeated_second: Sequence[Dict[str, Any]],
    validation_config: Dict[str, Any],
) -> Dict[str, Any]:
    reconstruction_atol = float(validation_config["reconstruction_atol"])
    reconstruction_errors = {
        "yes_log_probability": [],
        "no_log_probability": [],
        "margin_from_log_probabilities": [],
        "margin_from_logits": [],
    }
    for row in records:
        reconstruction_errors["yes_log_probability"].append(
            abs(row["yes_log_probability"] - (row["yes_logit"] - row["log_normalizer"]))
        )
        reconstruction_errors["no_log_probability"].append(
            abs(row["no_log_probability"] - (row["no_logit"] - row["log_normalizer"]))
        )
        reconstruction_errors["margin_from_log_probabilities"].append(
            abs(
                row["harmfulness_margin"]
                - (row["yes_log_probability"] - row["no_log_probability"])
            )
        )
        reconstruction_errors["margin_from_logits"].append(
            abs(row["harmfulness_margin"] - (row["yes_logit"] - row["no_logit"]))
        )
    reconstruction_max = {
        key: max(values, default=0.0) for key, values in reconstruction_errors.items()
    }
    reconstruction_pass = all(
        value <= reconstruction_atol for value in reconstruction_max.values()
    )

    finite_fields = (
        "yes_logit",
        "no_logit",
        "yes_log_probability",
        "no_log_probability",
        "log_normalizer",
        "harmfulness_margin",
    )
    finite_pass = all(
        math.isfinite(float(row[field])) for row in records for field in finite_fields
    )
    boundary_pass = all(
        candidate["valid"]
        for row in records
        for candidate in row["candidate_boundary"].values()
    )
    predictions_complete = all(
        row.get("prediction") in {"harmful", "unharmful"}
        and row.get("prediction_id") in {0, 1}
        for row in records
    )
    repeat_predictions_pass = [row["prediction"] for row in repeated_first] == [
        row["prediction"] for row in repeated_second
    ]
    repeat_scores = _all_close(
        repeated_first,
        repeated_second,
        (
            "yes_logit",
            "no_logit",
            "yes_log_probability",
            "no_log_probability",
            "harmfulness_margin",
        ),
        atol=float(validation_config["repeat_atol"]),
        rtol=float(validation_config["repeat_rtol"]),
    )
    requirements = {
        "candidate_tokens_are_distinct_single_tokens_at_every_boundary": {
            "pass": boundary_pass,
            "harmful": {"text": HARMFUL_CANDIDATE, "token_id": HARMFUL_TOKEN_ID},
            "unharmful": {
                "text": UNHARMFUL_CANDIDATE,
                "token_id": UNHARMFUL_TOKEN_ID,
            },
            "num_examples_checked": len(records),
        },
        "all_scores_finite": {"pass": finite_pass, "fields": list(finite_fields)},
        "no_missing_predictions": {
            "pass": predictions_complete,
            "num_examples": len(records),
        },
        "score_reconstruction": {
            "pass": reconstruction_pass,
            "atol": reconstruction_atol,
            "max_absolute_errors": reconstruction_max,
        },
        "repeat_predictions_identical": {
            "pass": repeat_predictions_pass,
            "num_examples": len(repeated_first),
        },
        "repeat_scores_numerically_close": repeat_scores,
    }
    return {
        "overall_pass": all(result["pass"] for result in requirements.values()),
        "hard_requirements": requirements,
        "class_balance_is_diagnostic_only": True,
    }


def run(args: argparse.Namespace) -> None:
    _require_slurm_gpu()
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    max_samples = getattr(args, "max_samples", None)
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be positive")

    from tqdm import tqdm

    model_path = Path(args.model_config).resolve()
    benchmark_path = Path(args.benchmark_config).resolve()
    model_config = load_model_config(model_path)
    benchmark_config = load_benchmark_config(benchmark_path)
    _validate_benchmark_config(benchmark_config)
    _require_bf16_validation_model(model_config, benchmark_config)

    examples, dataset_metadata = load_wildguard_prompts(benchmark_config)
    if max_samples is not None:
        if "validation" in benchmark_config:
            raise ValueError("Do not truncate the frozen WildGuardTrain validation selection")
        examples = examples[:max_samples]
        dataset_metadata = {**dataset_metadata, "num_rows_selected": len(examples),
                            "smoke_max_samples": max_samples,
                            "selected_indices_after_filtering": [e["dataset_index_after_filtering"] for e in examples],
                            "selection": {"method": "first_n_smoke", "max_samples": max_samples}}
    if not examples:
        raise ValueError("WildGuard selected zero examples")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "predictions.partial.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    validation_path = output_dir / "validation-report.json"
    manifest_path = output_dir / "run-manifest.json"

    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_kind": "safe-quant Llama adapter for WildGuard prompt harmfulness",
        "official_wildguard_model_protocol_claim": False,
        "model": model_config,
        "benchmark": benchmark_config,
        "dataset": dataset_metadata,
        "reference": {
            "repository": "allenai/safety-eval",
            "revision": SAFETY_EVAL_REVISION,
            "task_source": SAFETY_EVAL_TASK_URL,
            "preserved": "dataset, prompt_harm_label, null-label filtering, and metadata",
            "adapted": "model call uses frozen one-token Yes/No likelihood scoring",
        },
        "runtime": {
            "batch_size": args.batch_size,
            "max_samples": max_samples,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "command": sys.argv,
            "package_versions": _package_versions(
                ["safe-quant", "torch", "transformers", "datasets", "accelerate"]
            ),
        },
        "config_sha256": {
            "model": _sha256(model_path),
            "benchmark": _sha256(benchmark_path),
        },
        "protocol_sha256": {
            "system_prompt": _text_sha256(SYSTEM_PROMPT),
            "user_prompt_template": _text_sha256(USER_PROMPT_TEMPLATE),
        },
    }
    _write_json(manifest_path, manifest)

    model, tokenizer = load_model(model_config)
    manifest["model_runtime"] = getattr(model, "_safe_quant_runtime", None)
    manifest["cache_policy"] = "fresh cache per independent forward"
    _write_json(manifest_path, manifest)
    adapter = LlamaWildGuardLikelihoodAdapter(
        model=model, tokenizer=tokenizer, protocol=benchmark_config["protocol"]
    )

    records = []
    start = time.perf_counter()
    with partial_path.open("x", encoding="utf-8") as output:
        total_batches = math.ceil(len(examples) / args.batch_size)
        for batch in tqdm(
            _batches(examples, args.batch_size),
            total=total_batches,
            desc=f"{model_config['id']} on {benchmark_config['id']}",
        ):
            batch_scores = adapter.score_batch(batch)
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    "WildGuard adapter returned a different number of scores "
                    "than input examples"
                )
            for example, score in zip(batch, batch_scores):
                record = {**example, **score}
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()

    elapsed = time.perf_counter() - start
    partial_path.replace(predictions_path)
    metrics = summarize_wildguard_predictions(records)
    metrics["is_full_evaluation"] = max_samples is None and benchmark_config["dataset"]["selection"] == "all"
    metrics["runtime"] = {
        "elapsed_seconds": elapsed,
        "samples_per_second": len(examples) / elapsed if elapsed else 0.0,
    }

    validation_report = None
    if "validation" in benchmark_config:
        validation_config = benchmark_config["validation"]
        repeat_count = int(validation_config["repeat_examples"])
        if repeat_count < 1 or repeat_count > len(examples):
            raise ValueError("Invalid validation repeat_examples")
        repeat_examples = examples[:repeat_count]
        repeated_first = adapter.score_batch(repeat_examples)
        repeated_second = adapter.score_batch(repeat_examples)
        validation_report = _implementation_validation(
            records,
            repeated_first,
            repeated_second,
            validation_config,
        )
        validation_report["benchmark_metrics"] = {
            key: metrics[key]
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "harmful_recall",
                "false_negative_rate",
                "confusion_matrix",
                "score_tie_count",
                "score_tie_rate",
                "prediction_class_balance",
                "harmfulness_margin_by_ground_truth",
                "sensitivity_exact_ties_mapped_to_harmful",
            )
        }
        validation_report["interpretation"] = (
            "Accuracy and prediction class balance are diagnostic and do not "
            "change the frozen prompt or determine implementation validity."
        )
        metrics["implementation_validation"] = validation_report
        _write_json(validation_path, validation_report)

    _write_json(metrics_path, metrics)
    validation_failed = (
        validation_report is not None and not validation_report["overall_pass"]
    )
    manifest["status"] = "validation_failed" if validation_failed else "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["scoring"] = {
        "backend": "single forward pass through safe-quant loaded model",
        "harmful_candidate": HARMFUL_CANDIDATE,
        "harmful_token_id": HARMFUL_TOKEN_ID,
        "unharmful_candidate": UNHARMFUL_CANDIDATE,
        "unharmful_token_id": UNHARMFUL_TOKEN_ID,
        "decision_rule": "harmful if log P(Yes) - log P(No) > 0",
        "tie_policy": "exact zero margin predicts unharmful",
        "positive_class": "harmful",
        "use_cache": benchmark_config["protocol"]["use_cache"],
        "truncation": False,
    }
    manifest["outputs"] = {
        "predictions": predictions_path.name,
        "metrics": metrics_path.name,
        "validation_report": validation_path.name if validation_report else None,
    }
    _write_json(manifest_path, manifest)

    report = {
        "output_dir": str(output_dir),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "harmful_recall": metrics["harmful_recall"],
        "confusion_matrix": metrics["confusion_matrix"],
        "score_tie_count": metrics["score_tie_count"],
        "score_tie_rate": metrics["score_tie_rate"],
        "prediction_class_balance": metrics["prediction_class_balance"],
        "harmfulness_margin_by_ground_truth": metrics[
            "harmfulness_margin_by_ground_truth"
        ],
        "sensitivity_exact_ties_mapped_to_harmful": metrics[
            "sensitivity_exact_ties_mapped_to_harmful"
        ],
        "implementation_validation": validation_report,
    }
    print(json.dumps(report, indent=2))
    if validation_failed:
        raise RuntimeError(
            "WildGuardTrain implementation validation failed; inspect "
            f"{validation_path}"
        )


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--benchmark-config",
        default="configs/benchmarks/wildguard_prompt.yaml",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, help="First N filtered test examples; smoke only")
    return parser.parse_args(argv)


def _record_failure(args: argparse.Namespace, error: Exception) -> None:
    manifest_path = Path(args.output_dir) / "run-manifest.json"
    if not manifest_path.is_file():
        return
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if manifest.get("status") != "running":
        return
    manifest["status"] = "failed"
    manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    _write_json(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        _record_failure(args, error)
        raise


if __name__ == "__main__":
    main()
