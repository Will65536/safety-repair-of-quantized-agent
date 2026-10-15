"""Compare paired judgments for BF16 and a matched quantized model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

from safe_quant.benchmarks.behavioral import read_jsonl


# Native Transformers 4.44/4.45 serialize these differently. AdvBench uses
# single-model generation (never speculative/assistant decoding). Cache return
# representation is also runtime metadata, not a token-selection setting.
ADVBENCH_NATIVE_METADATA = (
    "transformers_version", "assistant_confidence_threshold", "is_assistant",
    "num_assistant_tokens", "num_assistant_tokens_schedule", "return_legacy_cache",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_runs(baseline_dir: Path, variant_dir: Path) -> Dict[str, Any]:
    base_manifest = _read_json(baseline_dir / "run-manifest.json")
    quant_manifest = _read_json(variant_dir / "run-manifest.json")
    if base_manifest["status"] != "complete" or quant_manifest["status"] != "complete":
        raise ValueError("Both judgment runs must be complete")
    if base_manifest["backend"] != quant_manifest["backend"]:
        raise ValueError("Paired runs used different evaluators")
    base_gen = base_manifest["source_generation_manifest"]
    quant_gen = quant_manifest["source_generation_manifest"]
    if base_gen["benchmark"] != quant_gen["benchmark"] or base_gen["prompts_sha256"] != quant_gen["prompts_sha256"]:
        raise ValueError("Paired runs used different prepared prompts or benchmark configs")
    base_settings = copy.deepcopy(base_gen["generation_config"])
    quant_settings = copy.deepcopy(quant_gen["generation_config"])
    if base_gen["benchmark"]["id"] == "advbench_directrequest":
        for settings in (base_settings, quant_settings):
            effective = settings.get("effective_generation_config", {})
            if effective.get("is_assistant"):
                raise ValueError("AdvBench requires single-model, non-assistant generation")
            for key in ADVBENCH_NATIVE_METADATA:
                effective.pop(key, None)
    if base_settings != quant_settings:
        raise ValueError("Paired runs used different generation settings")
    if base_manifest["evaluator"] != quant_manifest["evaluator"]:
        raise ValueError("Paired runs used different evaluator versions/settings")
    base = read_jsonl(baseline_dir / "predictions.jsonl")
    quant = read_jsonl(variant_dir / "predictions.jsonl")
    if len(base) != len(quant):
        raise ValueError("Paired run row counts differ")
    identity = ("benchmark_id", "benchmark_version", "sample_id", "source_row", "prompt", "metadata", "rendered_prompt_sha256", "prompt_token_count")
    for left, right in zip(base, quant):
        if any(left[field] != right[field] for field in identity):
            raise ValueError(f"Paired responses differ in prompt identity at {left['sample_id']}")
    benchmark_id = base_gen["benchmark"]["id"]
    base_metrics = _read_json(baseline_dir / "metrics.json")
    quant_metrics = _read_json(variant_dir / "metrics.json")
    result: Dict[str, Any] = {
        "benchmark_id": benchmark_id,
        "num_samples": len(base),
        "paired_samples_verified": True,
        "baseline_model": base_gen["model"]["id"],
        "variant_model": quant_gen["model"]["id"],
        "backend": base_manifest["backend"],
        "baseline_metrics": base_metrics,
        "variant_metrics": quant_metrics,
    }
    if benchmark_id == "advbench_directrequest" and base_manifest["backend"] == "advbench_harmbench_classifier":
        result["generation_comparison_excludes_native_metadata"] = list(ADVBENCH_NATIVE_METADATA)
        result["baseline_generation_config"] = base_gen["generation_config"]
        result["variant_generation_config"] = quant_gen["generation_config"]
        pairs = [(left["evaluator_output"]["label"], right["evaluator_output"]["label"])
                 for left, right in zip(base, quant)]
        result["paired_flips"] = {
            "baseline_negative_to_variant_positive": sum(a == 0 and b == 1 for a, b in pairs),
            "baseline_positive_to_variant_negative": sum(a == 1 and b == 0 for a, b in pairs),
        }
        result["asr_difference_variant_minus_baseline"] = quant_metrics["asr"] - base_metrics["asr"]
        result["paired_labels"] = [
            {"sample_id": left["sample_id"], "baseline_label": a, "variant_label": b}
            for left, (a, b) in zip(base, pairs)
        ]
    elif benchmark_id == "advbench_directrequest":
        result["generation_comparison_excludes_native_metadata"] = list(ADVBENCH_NATIVE_METADATA)
        result["baseline_generation_config"] = base_gen["generation_config"]
        result["variant_generation_config"] = quant_gen["generation_config"]
        result["non_refusal_rate_difference_variant_minus_baseline"] = (
            quant_metrics["non_refusal_rate"] - base_metrics["non_refusal_rate"]
        )
        pairs = [(left["evaluator_output"]["label"], right["evaluator_output"]["label"])
                 for left, right in zip(base, quant)]
        result["paired_flips"] = {
            "baseline_refusal_to_variant_non_refusal": sum(
                a == "refusal" and b == "non_refusal" for a, b in pairs),
            "baseline_non_refusal_to_variant_refusal": sum(
                a == "non_refusal" and b == "refusal" for a, b in pairs),
        }
        result["paired_labels"] = [
            {"sample_id": left["sample_id"], "baseline_label": a, "variant_label": b}
            for left, (a, b) in zip(base, pairs)
        ]
    elif benchmark_id == "harmbench_directrequest":
        pairs = [(int(left["evaluator_output"]["label"]), int(right["evaluator_output"]["label"])) for left, right in zip(base, quant)]
        result["paired_flips"] = {
            "bf16_safe_to_spinquant_harmful": sum(a == 0 and b == 1 for a, b in pairs),
            "bf16_harmful_to_spinquant_safe": sum(a == 1 and b == 0 for a, b in pairs),
        }
        result["asr_difference_variant_minus_baseline"] = quant_metrics["asr"] - base_metrics["asr"]
        result["paired_labels"] = [{"sample_id": left["sample_id"], "baseline_label": a, "variant_label": b} for left, (a, b) in zip(base, pairs)]
    elif benchmark_id in ("strongreject", "strongreject_small"):
        result["mean_difference_variant_minus_baseline"] = quant_metrics["mean_score"] - base_metrics["mean_score"]
        result["paired_scores"] = [{"sample_id": left["sample_id"], "baseline_score": left["evaluator_output"]["score"], "variant_score": right["evaluator_output"]["score"]} for left, right in zip(base, quant)]
    elif benchmark_id == "xstest":
        result["refusal_rate_change_variant_minus_baseline"] = {
            subset: quant_metrics["subsets"][subset]["refusal_rate"] - base_metrics["subsets"][subset]["refusal_rate"]
            for subset in ("safe", "unsafe_contrast")
        }
        result["paired_labels"] = [{"sample_id": left["sample_id"], "subset": left["metadata"]["subset"], "baseline_label": left["evaluator_output"]["label"], "variant_label": right["evaluator_output"]["label"]} for left, right in zip(base, quant)]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_id}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("variant_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.baseline_dir, args.variant_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("paired_")}, indent=2))


if __name__ == "__main__":
    main()
