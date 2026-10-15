#!/usr/bin/env python
"""Prepare one immutable prompt file to share across model variants."""

import argparse
import json
import re
from pathlib import Path

from safe_quant.benchmarks.behavioral import prepare_rows, sha256_file
from safe_quant.config import load_benchmark_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--source-revision", required=True, help="Official repository commit SHA")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_benchmark_config(args.benchmark_config)
    if config["loader"] != "behavioral_generation":
        raise ValueError("Benchmark is not behavioral")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("--source-revision must be a full 40-character repository commit SHA")
    if args.source_file.name != Path(config["dataset"]["file"]).name:
        raise ValueError("Source CSV name differs from the official benchmark configuration")
    if args.source_revision != config["dataset"]["revision"]:
        raise ValueError("Official source checkout differs from pinned benchmark revision")
    source_sha256 = sha256_file(args.source_file)
    if source_sha256 != config["dataset"]["sha256"]:
        raise ValueError("Official source CSV differs from pinned benchmark data")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    rows = prepare_rows(config["id"], args.source_file)
    expected_rows = config["dataset"].get("expected_selected_rows")
    if expected_rows is not None and len(rows) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} selected rows; found {len(rows)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = args.output_dir / "prompts.jsonl"
    with prompts.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "benchmark": config,
        "source_revision": args.source_revision,
        "source_file": str(args.source_file.resolve()),
        "source_sha256": source_sha256,
        "prompts_sha256": sha256_file(prompts),
        "num_samples": len(rows),
    }
    (args.output_dir / "prompt-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "num_samples": len(rows)}))


if __name__ == "__main__":
    main()
