"""Prepare fixed behavioral benchmark prompts from official CSV files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_rows(benchmark_id: str, source_file: Path) -> List[Dict[str, Any]]:
    with source_file.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: List[Dict[str, Any]] = []
    for index, source in enumerate(source_rows):
        if benchmark_id == "harmbench_directrequest":
            if source["FunctionalCategory"] != "standard":
                continue
            behavior = source["Behavior"]
            context = source.get("ContextString") or ""
            prompt = f"{context}\n\n---\n\n{behavior}" if context else behavior
            sample_id = source["BehaviorID"]
            metadata = {
                "BehaviorID": sample_id,
                "FunctionalCategory": source["FunctionalCategory"],
                "SemanticCategory": source["SemanticCategory"],
                "Tags": source.get("Tags"),
                "ContextString": context,
            }
        elif benchmark_id == "advbench_directrequest":
            prompt = source["goal"]
            sample_id = f"advbench-{index:03d}"
            # Reference targets are metadata only: never append/prefill them.
            metadata = {"target": source["target"], "method": "DirectRequest"}
        elif benchmark_id in ("strongreject", "strongreject_small"):
            prompt = source["forbidden_prompt"]
            prefix = "strongreject" if benchmark_id == "strongreject" else "strongreject-small"
            sample_id = f"{prefix}-{index:03d}"
            metadata = {"category": source["category"], "source": source["source"]}
        elif benchmark_id == "xstest":
            prompt = source["prompt"]
            sample_id = str(source["id"])
            metadata = {key: source.get(key) for key in ("type", "label", "focus", "note")}
            metadata["subset"] = "unsafe_contrast" if metadata["type"].startswith("contrast_") else "safe"
        else:
            raise ValueError(f"Unsupported behavioral benchmark: {benchmark_id}")
        if not prompt or not sample_id:
            raise ValueError(f"Blank prompt or ID at source row {index}")
        rows.append({"sample_id": sample_id, "source_row": index, "prompt": prompt, "metadata": metadata})
    expected = {"strongreject": 313, "strongreject_small": 60, "xstest": 450,
                "advbench_directrequest": 520}
    if benchmark_id in expected and len(rows) != expected[benchmark_id]:
        raise ValueError(f"Expected {expected[benchmark_id]} {benchmark_id} rows; found {len(rows)}")
    if benchmark_id == "harmbench_directrequest" and not rows:
        raise ValueError("HarmBench standard selection is empty")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {benchmark_id}")
    return rows
