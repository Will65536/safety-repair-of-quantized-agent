"""Map pinned AdvBench goals to HarmBench's standard behavior interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from safe_quant.benchmarks.advbench import REPOSITORY_REVISION
from safe_quant.benchmarks.behavioral import prepare_rows, sha256_file


DATASET_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
HARMBENCH_REVISION = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
EVALUATE_SHA256 = "544692cd80894afda32214d9a9d641043b3c7031f9c64ad54735d3fe3ad1e5f6"
EVAL_UTILS_SHA256 = "46f1db751866848834d45d5128a18d4cc0a820828e89796ab51bc605d10c7058"
CLASSIFIER_REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"
BEHAVIOR_FIELDS = ("BehaviorID", "Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString")


def standard_behaviors(
    rows: List[Dict[str, Any]], generation_manifest: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Validate original goals, then adapt every saved response, without filtering."""
    dataset = generation_manifest["benchmark"]["dataset"]
    if (dataset.get("repository") != "llm-attacks/llm-attacks"
            or dataset.get("revision") != REPOSITORY_REVISION
            or dataset.get("sha256") != DATASET_SHA256):
        raise ValueError("AdvBench source metadata differs from the pinned dataset")
    prepared = generation_manifest["prepared_prompts"]
    source = Path(prepared["source_file"])
    if (prepared.get("source_revision") != REPOSITORY_REVISION
            or prepared.get("source_sha256") != DATASET_SHA256
            or not source.is_file() or sha256_file(source) != DATASET_SHA256):
        raise ValueError("Original AdvBench source CSV is missing or changed")
    catalog = {r["sample_id"]: r for r in prepare_rows("advbench_directrequest", source)}
    ids = [r["sample_id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("AdvBench response IDs must be nonempty and unique")
    if generation_manifest.get("is_full_evaluation") and set(ids) != set(catalog):
        raise ValueError("Full AdvBench classifier judging requires all 520 responses")
    behaviors = []
    for row in rows:
        original = catalog.get(row["sample_id"])
        if original is None or (row["prompt"], row["source_row"]) != (original["prompt"], original["source_row"]):
            raise ValueError(f"AdvBench prompt identity changed: {row['sample_id']}")
        behaviors.append({
            "BehaviorID": row["sample_id"],
            "Behavior": original["prompt"],
            "FunctionalCategory": "standard",
            "SemanticCategory": "",
            "Tags": "",
            "ContextString": "",
        })
    return behaviors
