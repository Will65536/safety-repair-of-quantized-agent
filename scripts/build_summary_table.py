#!/usr/bin/env python
"""Build the requested result table from completed evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_rows(
    general: Dict[str, Any],
    finguard_bf16: Dict[str, Any],
    finguard_spinquant: Dict[str, Any],
    wildguard: Dict[str, Any],
) -> List[Dict[str, Any]]:
    wiki = general["wikitext2"]["perplexity"]
    four_task = general["general_4task"]
    rows = []
    for model_key, model_label, finguard in (
        ("bf16", "Llama-3-8B-Instruct BF16", finguard_bf16),
        (
            "spinquant_w4a4kv4",
            "Llama-3-8B-Instruct SpinQuant W4A4KV4",
            finguard_spinquant,
        ),
    ):
        rows.append(
            {
                "model": model_label,
                "wikitext2_perplexity": wiki[model_key],
                "arc_easy_accuracy": four_task["tasks"]["arc_easy"][model_key],
                "arc_challenge_accuracy": four_task["tasks"]["arc_challenge"][
                    model_key
                ],
                "piqa_accuracy": four_task["tasks"]["piqa"][model_key],
                "winogrande_accuracy": four_task["tasks"]["winogrande"][
                    model_key
                ],
                "general_4task_avg": four_task["general_4task_avg"][model_key],
                "finguard_accuracy": finguard["overall"]["accuracy"],
                "wildguard_accuracy": wildguard["metrics"]["accuracy"][model_key],
            }
        )
    return rows


def _markdown(rows: List[Dict[str, Any]]) -> str:
    columns = (
        ("Model", "model"),
        ("WikiText2 PPL", "wikitext2_perplexity"),
        ("ARC-E", "arc_easy_accuracy"),
        ("ARC-C", "arc_challenge_accuracy"),
        ("PIQA", "piqa_accuracy"),
        ("WinoGrande", "winogrande_accuracy"),
        ("General 4-task Avg", "general_4task_avg"),
        ("FinGuard", "finguard_accuracy"),
        ("WildGuard", "wildguard_accuracy"),
    )
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" if key == "model" else "---:" for _, key in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row[key]
            values.append(str(value) if key == "model" else f"{float(value):.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general-comparison", type=Path, required=True)
    parser.add_argument("--finguard-bf16", type=Path, required=True)
    parser.add_argument("--finguard-spinquant", type=Path, required=True)
    parser.add_argument("--wildguard-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(
        _read_json(args.general_comparison),
        _read_json(args.finguard_bf16),
        _read_json(args.finguard_spinquant),
        _read_json(args.wildguard_comparison),
    )
    if args.output.exists():
        raise FileExistsError(f"Summary output already exists: {args.output}")
    markdown_path = args.output.with_suffix(".md")
    if markdown_path.exists():
        raise FileExistsError(f"Summary output already exists: {markdown_path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    markdown = _markdown(rows)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
