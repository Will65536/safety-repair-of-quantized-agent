#!/usr/bin/env python
"""Verify and compare paired BF16 and SpinQuant WildGuardTest runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


HIGHER_IS_BETTER = ("accuracy", "precision", "recall", "f1", "harmful_recall")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_runs(bf16_dir: Path, spinquant_dir: Path) -> Dict[str, Any]:
    bf16_manifest = _read_json(bf16_dir / "run-manifest.json")
    spin_manifest = _read_json(spinquant_dir / "run-manifest.json")
    if bf16_manifest.get("status") != "complete":
        raise ValueError("BF16 run is not complete")
    if spin_manifest.get("status") != "complete":
        raise ValueError("SpinQuant run is not complete")

    identical_settings = {
        "benchmark_config_sha256": (
            bf16_manifest["config_sha256"]["benchmark"]
            == spin_manifest["config_sha256"]["benchmark"]
        ),
        "resolved_benchmark_config": (
            bf16_manifest["benchmark"] == spin_manifest["benchmark"]
        ),
        "dataset_metadata": bf16_manifest["dataset"] == spin_manifest["dataset"],
        "batch_size": (
            bf16_manifest["runtime"]["batch_size"]
            == spin_manifest["runtime"]["batch_size"]
        ),
        "scoring": bf16_manifest["scoring"] == spin_manifest["scoring"],
    }
    if not all(identical_settings.values()):
        raise ValueError(
            "Paired WildGuard runs used different evaluation settings: "
            + repr(identical_settings)
        )

    bf16_rows = _read_predictions(bf16_dir / "predictions.jsonl")
    spin_rows = _read_predictions(spinquant_dir / "predictions.jsonl")
    identity_fields = (
        "example_id",
        "dataset_index_after_filtering",
        "prompt",
        "gt_prompt_harmfulness",
        "label_id",
        "adversarial",
        "prompt_harm_category",
    )
    bf16_identity: List[Tuple[Any, ...]] = [
        tuple(row.get(field) for field in identity_fields) for row in bf16_rows
    ]
    spin_identity: List[Tuple[Any, ...]] = [
        tuple(row.get(field) for field in identity_fields) for row in spin_rows
    ]
    if bf16_identity != spin_identity:
        raise ValueError("BF16 and SpinQuant prediction rows are not aligned")

    bf16_metrics = _read_json(bf16_dir / "metrics.json")
    spin_metrics = _read_json(spinquant_dir / "metrics.json")
    comparison = {}
    for metric in HIGHER_IS_BETTER:
        bf16_value = float(bf16_metrics[metric])
        spin_value = float(spin_metrics[metric])
        comparison[metric] = {
            "bf16": bf16_value,
            "spinquant_w4a4kv4": spin_value,
            "absolute_change_spinquant_minus_bf16": spin_value - bf16_value,
            "degradation_bf16_minus_spinquant": bf16_value - spin_value,
        }
    bf16_fnr = float(bf16_metrics["false_negative_rate"])
    spin_fnr = float(spin_metrics["false_negative_rate"])
    comparison["false_negative_rate"] = {
        "bf16": bf16_fnr,
        "spinquant_w4a4kv4": spin_fnr,
        "increase_spinquant_minus_bf16": spin_fnr - bf16_fnr,
    }

    bf16_sensitivity = bf16_metrics[
        "sensitivity_exact_ties_mapped_to_harmful"
    ]
    spin_sensitivity = spin_metrics[
        "sensitivity_exact_ties_mapped_to_harmful"
    ]
    sensitivity_comparison = {}
    for metric in HIGHER_IS_BETTER:
        bf16_value = float(bf16_sensitivity["metrics"][metric])
        spin_value = float(spin_sensitivity["metrics"][metric])
        sensitivity_comparison[metric] = {
            "bf16": bf16_value,
            "spinquant_w4a4kv4": spin_value,
            "absolute_change_spinquant_minus_bf16": spin_value - bf16_value,
            "degradation_bf16_minus_spinquant": bf16_value - spin_value,
        }
    bf16_sensitivity_fnr = float(
        bf16_sensitivity["metrics"]["false_negative_rate"]
    )
    spin_sensitivity_fnr = float(
        spin_sensitivity["metrics"]["false_negative_rate"]
    )
    sensitivity_comparison["false_negative_rate"] = {
        "bf16": bf16_sensitivity_fnr,
        "spinquant_w4a4kv4": spin_sensitivity_fnr,
        "increase_spinquant_minus_bf16": (
            spin_sensitivity_fnr - bf16_sensitivity_fnr
        ),
    }
    tie_sensitivity = {
        "status": "sensitivity analysis; not primary metrics",
        "bf16": bf16_sensitivity,
        "spinquant_w4a4kv4": spin_sensitivity,
        "paired_metrics": sensitivity_comparison,
    }

    return {
        "comparison": "WildGuardTest prompt harmfulness with frozen Llama adapter",
        "num_examples": len(bf16_rows),
        "paired_examples_verified": True,
        "identical_evaluation_settings": identical_settings,
        "models": {
            "bf16": bf16_manifest["model"],
            "spinquant_w4a4kv4": spin_manifest["model"],
        },
        "metrics": comparison,
        "score_ties": {
            "bf16": {
                "count": bf16_metrics["score_tie_count"],
                "rate": bf16_metrics["score_tie_rate"],
            },
            "spinquant_w4a4kv4": {
                "count": spin_metrics["score_tie_count"],
                "rate": spin_metrics["score_tie_rate"],
            },
        },
        "tie_sensitivity": tie_sensitivity,
        "confusion_matrices": {
            "bf16": bf16_metrics["confusion_matrix"],
            "spinquant_w4a4kv4": spin_metrics["confusion_matrix"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16_dir", type=Path)
    parser.add_argument("spinquant_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_runs(args.bf16_dir, args.spinquant_dir)
    if args.output.exists():
        raise FileExistsError(f"Comparison output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
