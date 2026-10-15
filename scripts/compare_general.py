#!/usr/bin/env python
"""Verify and compare paired BF16 and SpinQuant general evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


TASKS = ("arc_easy", "arc_challenge", "piqa", "winogrande")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require_complete(manifest: Dict[str, Any], label: str) -> None:
    if manifest.get("status") != "complete":
        raise ValueError(f"{label} run is not complete")


def _compare_wikitext(bf16_dir: Path, spinquant_dir: Path) -> Dict[str, Any]:
    bf_manifest = _read_json(bf16_dir / "run-manifest.json")
    sq_manifest = _read_json(spinquant_dir / "run-manifest.json")
    _require_complete(bf_manifest, "BF16 WikiText-2")
    _require_complete(sq_manifest, "SpinQuant WikiText-2")

    settings = {
        "benchmark_config_sha256": (
            bf_manifest["config_sha256"]["benchmark"]
            == sq_manifest["config_sha256"]["benchmark"]
        ),
        "resolved_benchmark_config": (
            bf_manifest["benchmark"] == sq_manifest["benchmark"]
        ),
        "dataset_metadata": bf_manifest["dataset"] == sq_manifest["dataset"],
        "max_blocks": (
            bf_manifest["runtime"]["max_blocks"]
            == sq_manifest["runtime"]["max_blocks"]
        ),
        "tokenization": bf_manifest["tokenization"] == sq_manifest["tokenization"],
    }
    if not all(settings.values()):
        raise ValueError(f"WikiText-2 settings differ: {settings}")

    bf_blocks = _read_jsonl(bf16_dir / "blocks.jsonl")
    sq_blocks = _read_jsonl(spinquant_dir / "blocks.jsonl")
    identity_fields = (
        "block_index",
        "start_token_offset",
        "end_token_offset_exclusive",
        "num_input_tokens",
        "num_predicted_tokens",
    )
    bf_identity = [tuple(row[key] for key in identity_fields) for row in bf_blocks]
    sq_identity = [tuple(row[key] for key in identity_fields) for row in sq_blocks]
    if bf_identity != sq_identity:
        raise ValueError("WikiText-2 blocks are not aligned")

    bf_metrics = _read_json(bf16_dir / "metrics.json")
    sq_metrics = _read_json(spinquant_dir / "metrics.json")
    bf_ppl = float(bf_metrics["perplexity"])
    sq_ppl = float(sq_metrics["perplexity"])
    return {
        "paired_blocks_verified": True,
        "num_blocks": len(bf_blocks),
        "identical_evaluation_settings": settings,
        "perplexity": {
            "bf16": bf_ppl,
            "spinquant_w4a4kv4": sq_ppl,
            "increase_spinquant_minus_bf16": sq_ppl - bf_ppl,
            "relative_increase": (sq_ppl / bf_ppl) - 1.0,
        },
    }


def _sample_identity(rows: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    fields = ("task", "doc_id", "doc_hash", "prompt_hash", "target_hash", "filter")
    return [tuple(row.get(field) for field in fields) for row in rows]


def _compare_four_tasks(bf16_dir: Path, spinquant_dir: Path) -> Dict[str, Any]:
    bf_manifest = _read_json(bf16_dir / "run-manifest.json")
    sq_manifest = _read_json(spinquant_dir / "run-manifest.json")
    _require_complete(bf_manifest, "BF16 four-task")
    _require_complete(sq_manifest, "SpinQuant four-task")

    bf_harness = bf_manifest["harness_metadata"]
    sq_harness = sq_manifest["harness_metadata"]
    settings = {
        "benchmark_config_sha256": (
            bf_manifest["config_sha256"]["benchmark"]
            == sq_manifest["config_sha256"]["benchmark"]
        ),
        "resolved_benchmark_config": (
            bf_manifest["benchmark"] == sq_manifest["benchmark"]
        ),
        "limit_per_task": (
            bf_manifest["runtime"]["limit_per_task"]
            == sq_manifest["runtime"]["limit_per_task"]
        ),
        "lm_eval_version": (
            bf_manifest["runtime"]["package_versions"]["lm-eval"]
            == sq_manifest["runtime"]["package_versions"]["lm-eval"]
        ),
        "task_versions": bf_harness["task_versions"] == sq_harness["task_versions"],
        "num_fewshot": bf_harness["num_fewshot"] == sq_harness["num_fewshot"],
        "task_definition_sha256": (
            bf_harness["task_definition_sha256"]
            == sq_harness["task_definition_sha256"]
        ),
    }
    if not all(settings.values()):
        raise ValueError(f"Four-task settings differ: {settings}")

    bf_samples = _read_jsonl(bf16_dir / "samples.jsonl")
    sq_samples = _read_jsonl(spinquant_dir / "samples.jsonl")
    if _sample_identity(bf_samples) != _sample_identity(sq_samples):
        raise ValueError("BF16 and SpinQuant harness samples are not aligned")

    bf_metrics = _read_json(bf16_dir / "metrics.json")
    sq_metrics = _read_json(spinquant_dir / "metrics.json")
    task_metrics = {}
    for task in TASKS:
        bf_accuracy = float(bf_metrics["tasks"][task]["accuracy"])
        sq_accuracy = float(sq_metrics["tasks"][task]["accuracy"])
        row = {
            "bf16": bf_accuracy,
            "spinquant_w4a4kv4": sq_accuracy,
            "change_spinquant_minus_bf16": sq_accuracy - bf_accuracy,
            "degradation_bf16_minus_spinquant": bf_accuracy - sq_accuracy,
        }
        if "accuracy_normalized_diagnostic" in bf_metrics["tasks"][task]:
            row["accuracy_normalized_diagnostic"] = {
                "bf16": float(
                    bf_metrics["tasks"][task]["accuracy_normalized_diagnostic"]
                ),
                "spinquant_w4a4kv4": float(
                    sq_metrics["tasks"][task]["accuracy_normalized_diagnostic"]
                ),
            }
        task_metrics[task] = row

    bf_average = float(bf_metrics["general_4task_avg"])
    sq_average = float(sq_metrics["general_4task_avg"])
    return {
        "paired_samples_verified": True,
        "num_logged_samples": len(bf_samples),
        "identical_evaluation_settings": settings,
        "tasks": task_metrics,
        "general_4task_avg": {
            "bf16": bf_average,
            "spinquant_w4a4kv4": sq_average,
            "change_spinquant_minus_bf16": sq_average - bf_average,
            "degradation_bf16_minus_spinquant": bf_average - sq_average,
        },
    }


def compare_runs(bf16_root: Path, spinquant_root: Path) -> Dict[str, Any]:
    return {
        "comparison": "general capability: BF16 versus SpinQuant W4A4KV4",
        "wikitext2": _compare_wikitext(
            bf16_root / "wikitext2", spinquant_root / "wikitext2"
        ),
        "general_4task": _compare_four_tasks(
            bf16_root / "lm_eval_4task", spinquant_root / "lm_eval_4task"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16_root", type=Path)
    parser.add_argument("spinquant_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_runs(args.bf16_root, args.spinquant_root)
    if args.output.exists():
        raise FileExistsError(f"Comparison output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
