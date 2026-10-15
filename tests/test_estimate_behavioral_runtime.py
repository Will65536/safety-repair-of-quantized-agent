import json
from pathlib import Path

from safe_quant.benchmarks.behavioral import sha256_file
from safe_quant.estimate_behavioral_runtime import estimate_run


def test_estimate_run_separates_load_and_generation_time(tmp_path: Path) -> None:
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        json.dumps({"generation_seconds": 2.0}) + "\n"
        + json.dumps({"generation_seconds": 4.0}) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "complete",
        "model": {"id": "spinquant"},
        "benchmark": {"id": "example"},
        "prepared_prompts": {"num_samples": 10},
        "selected_num_samples": 2,
        "runtime": {"timing": {"model_load_seconds": 10.0}},
        "outputs": {
            "responses": "responses.jsonl",
            "responses_sha256": sha256_file(responses),
        },
    }
    (tmp_path / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    result = estimate_run(tmp_path)

    assert result["full_samples"] == 10
    assert result["mean_generation_seconds_per_example"] == 3.0
    assert result["mean_projection_seconds"] == 40.0
    assert result["max_projection_seconds"] == 50.0
    assert result["recommended_slurm_time"] == "00:30:00"
