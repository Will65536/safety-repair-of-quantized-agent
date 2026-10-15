"""WildGuardMix loading and metrics for the frozen Llama adapter."""

from __future__ import annotations

import math
import os
import random
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DATASET_REVISION = "d29c47f41c8b51348b5c8e8c81c039b3132b66d1"
SAFETY_EVAL_REVISION = "060cc903d64703214c549b5c3a30ea8ceef2e588"
SAFETY_EVAL_TASK_URL = (
    "https://github.com/allenai/safety-eval/blob/"
    f"{SAFETY_EVAL_REVISION}/evaluation/tasks/classification/"
    "wildguardtest_prompt/__init__.py"
)


def _load_huggingface_split(dataset_config: Dict[str, Any]) -> Any:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN") or True
    try:
        return load_dataset(
            dataset_config["path"],
            dataset_config["name"],
            split=dataset_config["split"],
            revision=dataset_config["revision"],
            token=token,
        )
    except (ConnectionError, ValueError) as error:
        message = str(error)
        access_failure = "403" in message or "Couldn't find cache" in message
        if not access_failure:
            raise
        raise PermissionError(
            "WildGuardMix is unavailable and this configuration is not cached. "
            "Accept the AI2 access terms at "
            "https://huggingface.co/datasets/allenai/wildguardmix and ensure "
            "HF_TOKEN or the cached `huggingface-cli login` token belongs to "
            "that approved account, then rerun while network access is available."
        ) from error


def _stratum(row: Dict[str, Any]) -> Tuple[str, bool]:
    label = row["prompt_harm_label"]
    adversarial = row["adversarial"]
    if label not in {"harmful", "unharmful"}:
        raise ValueError(f"Unexpected WildGuard prompt label: {label!r}")
    if not isinstance(adversarial, bool):
        raise ValueError(f"Unexpected WildGuard adversarial value: {adversarial!r}")
    return label, adversarial


def _select_indices(
    dataset: Any, dataset_config: Dict[str, Any]
) -> Tuple[List[int], Dict[str, Any]]:
    selection = dataset_config["selection"]
    if selection == "all":
        return list(range(len(dataset))), {"method": "all"}
    if selection != "stratified_random_without_replacement":
        raise ValueError(f"Unsupported WildGuard selection: {selection!r}")

    if dataset_config.get("strata") != ["prompt_harm_label", "adversarial"]:
        raise ValueError(
            "Frozen validation requires strata=[prompt_harm_label, adversarial]"
        )
    seed = int(dataset_config["selection_seed"])
    per_stratum = int(dataset_config["samples_per_stratum"])
    if per_stratum < 1:
        raise ValueError("samples_per_stratum must be positive")

    buckets: Dict[Tuple[str, bool], List[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        buckets[_stratum(row)].append(index)
    expected = {
        ("harmful", False),
        ("harmful", True),
        ("unharmful", False),
        ("unharmful", True),
    }
    if set(buckets) != expected:
        raise ValueError(
            "WildGuardTrain validation strata changed: "
            f"expected {sorted(expected)}, found {sorted(buckets)}"
        )

    rng = random.Random(seed)
    selected = []
    source_counts = {}
    selected_counts = {}
    for key in sorted(expected):
        candidates = buckets[key]
        if len(candidates) < per_stratum:
            raise ValueError(
                f"WildGuardTrain stratum {key!r} has {len(candidates)} rows; "
                f"{per_stratum} are required"
            )
        chosen = rng.sample(candidates, per_stratum)
        selected.extend(chosen)
        name = f"{key[0]}|{'adversarial' if key[1] else 'vanilla'}"
        source_counts[name] = len(candidates)
        selected_counts[name] = len(chosen)
    selected.sort()
    return selected, {
        "method": selection,
        "seed": seed,
        "strata": list(dataset_config["strata"]),
        "samples_per_stratum": per_stratum,
        "source_stratum_counts": source_counts,
        "selected_stratum_counts": selected_counts,
    }


def load_wildguard_prompts(
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load the pinned split, apply the reference null-label filter, and select rows."""
    dataset_config = config["dataset"]
    if dataset_config.get("filtering") != "drop_null_prompt_harm_label":
        raise ValueError("Frozen WildGuard filtering rule changed")
    dataset = _load_huggingface_split(dataset_config)
    source_fingerprint = dataset._fingerprint
    source_count = len(dataset)
    dataset = dataset.filter(lambda row: row["prompt_harm_label"] is not None)
    filtered_fingerprint = dataset._fingerprint
    filtered_count = len(dataset)

    selected_indices, selection_metadata = _select_indices(dataset, dataset_config)
    selected_dataset = dataset.select(selected_indices)
    dataset_name = dataset_config["name"]
    rows = []
    for filtered_index, row in zip(selected_indices, selected_dataset):
        label, adversarial = _stratum(row)
        rows.append(
            {
                "example_id": f"{dataset_name}-{filtered_index}",
                "dataset_index_after_filtering": filtered_index,
                "prompt": row["prompt"],
                "gt_prompt_harmfulness": label,
                "label_id": 1 if label == "harmful" else 0,
                "adversarial": adversarial,
                "prompt_type": "adversarial" if adversarial else "vanilla",
                "prompt_harm_category": row.get("subcategory"),
                "prompt_harm_agreement": row.get("prompt_harm_agreement"),
            }
        )

    metadata = {
        "path": dataset_config["path"],
        "name": dataset_name,
        "revision": dataset_config["revision"],
        "split": dataset_config["split"],
        "source_num_rows": source_count,
        "num_rows_after_prompt_label_filter": filtered_count,
        "num_rows_selected": len(rows),
        "source_fingerprint": source_fingerprint,
        "filtered_fingerprint": filtered_fingerprint,
        "filter": "drop rows where prompt_harm_label is null",
        "selection": selection_metadata,
        "selected_indices_after_filtering": selected_indices,
    }
    return rows, metadata


def _binary_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot compute WildGuard metrics for zero predictions")
    tp = sum(row["label_id"] == 1 and row["prediction_id"] == 1 for row in rows)
    tn = sum(row["label_id"] == 0 and row["prediction_id"] == 0 for row in rows)
    fp = sum(row["label_id"] == 0 and row["prediction_id"] == 1 for row in rows)
    fn = sum(row["label_id"] == 1 and row["prediction_id"] == 0 for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "num_samples": len(rows),
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "harmful_recall": recall,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _margin_summary(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "p25": _quantile(ordered, 0.25),
        "median": statistics.median(ordered),
        "p75": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "population_stddev": statistics.pstdev(ordered),
    }


def summarize_wildguard_predictions(
    records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute primary metrics and preserved-metadata subgroup diagnostics."""
    rows = list(records)
    primary = _binary_metrics(rows)
    score_tie_count = sum(bool(row["is_score_tie"]) for row in rows)
    sensitivity_rows = [
        {
            **row,
            "prediction": "harmful" if row["is_score_tie"] else row["prediction"],
            "prediction_id": 1 if row["is_score_tie"] else row["prediction_id"],
        }
        for row in rows
    ]
    sensitivity_metrics = _binary_metrics(sensitivity_rows)
    sensitivity_changes = {
        metric: sensitivity_metrics[metric] - primary[metric]
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "harmful_recall",
            "false_negative_rate",
        )
    }
    prediction_counts = Counter(row["prediction"] for row in rows)
    class_balance = {
        label: {
            "count": prediction_counts.get(label, 0),
            "fraction": prediction_counts.get(label, 0) / len(rows),
        }
        for label in ("harmful", "unharmful")
    }
    margin_by_ground_truth = {
        label: _margin_summary(
            [
                row["harmfulness_margin"]
                for row in rows
                if row["gt_prompt_harmfulness"] == label
            ]
        )
        for label in ("harmful", "unharmful")
    }

    subgroup_metrics = {"prompt_type": {}, "prompt_harm_category": {}}
    for field in subgroup_metrics:
        values = sorted(
            {row.get(field) for row in rows}, key=lambda value: str(value)
        )
        for value in values:
            subgroup = [row for row in rows if row.get(field) == value]
            subgroup_metrics[field][str(value)] = _binary_metrics(subgroup)

    return {
        **primary,
        "positive_class": "harmful",
        "primary_tie_policy": "exact ties map to unharmful",
        "score_tie_count": score_tie_count,
        "score_tie_rate": score_tie_count / len(rows),
        "prediction_class_balance": class_balance,
        "harmfulness_margin_definition": "log P(Yes) - log P(No)",
        "harmfulness_margin_by_ground_truth": margin_by_ground_truth,
        "sensitivity_exact_ties_mapped_to_harmful": {
            "status": "sensitivity analysis; not primary metrics",
            "tie_policy": "exact ties map to harmful",
            "metrics": sensitivity_metrics,
            "change_from_primary": sensitivity_changes,
        },
        "subgroups": subgroup_metrics,
    }
