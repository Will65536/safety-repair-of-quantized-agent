"""Project full behavioral runtime from a completed representative smoke run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from safe_quant.benchmarks.behavioral import read_jsonl, sha256_file


def _slurm_time(seconds: float) -> str:
    rounded = max(1800, math.ceil(seconds / 1800) * 1800)
    hours, remainder = divmod(int(rounded), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_run(smoke_dir: Path) -> Dict[str, Any]:
    manifest = json.loads((smoke_dir / "run-manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"Smoke run is not complete: {smoke_dir}")
    responses_path = smoke_dir / manifest["outputs"]["responses"]
    if sha256_file(responses_path) != manifest["outputs"]["responses_sha256"]:
        raise ValueError(f"Smoke responses changed: {responses_path}")
    rows = read_jsonl(responses_path)
    if len(rows) != manifest["selected_num_samples"]:
        raise ValueError(f"Smoke response count differs from manifest: {smoke_dir}")
    durations: List[float] = [float(row["generation_seconds"]) for row in rows]
    if not durations or any(value <= 0 for value in durations):
        raise ValueError(f"Smoke timing data is missing or invalid: {smoke_dir}")
    load_seconds = float(manifest["runtime"]["timing"]["model_load_seconds"])
    full_samples = int(manifest["prepared_prompts"]["num_samples"])
    mean_generation = statistics.fmean(durations)
    maximum_generation = max(durations)
    mean_projection = load_seconds + full_samples * mean_generation
    max_projection = load_seconds + full_samples * maximum_generation
    # Small smoke samples are intentionally representative rather than random.
    # Use the slowest observed example, then add 50% wall-time headroom and
    # round up to a scheduler-friendly 30-minute boundary.
    recommended_seconds = max_projection * 1.5
    return {
        "smoke_dir": str(smoke_dir),
        "benchmark_id": manifest["benchmark"]["id"],
        "model_id": manifest["model"]["id"],
        "smoke_samples": len(rows),
        "full_samples": full_samples,
        "model_load_seconds": load_seconds,
        "mean_generation_seconds_per_example": mean_generation,
        "maximum_generation_seconds_per_example": maximum_generation,
        "mean_projection_seconds": mean_projection,
        "max_projection_seconds": max_projection,
        "recommended_slurm_time": _slurm_time(recommended_seconds),
        "assumption": "full examples resemble the representative smoke sample; recommendation uses 1.5x the slowest-example projection",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("smoke_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([estimate_run(path) for path in args.smoke_dirs], indent=2))


if __name__ == "__main__":
    main()
