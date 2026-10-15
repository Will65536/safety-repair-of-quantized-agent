import copy
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from safe_quant.config import load_benchmark_config
from safe_quant.evaluate import _validate_protocol as validate_finguard
from safe_quant.evaluate_general import _validate_protocol as validate_general
from safe_quant.evaluate_wikitext2 import _validate_protocol as validate_wikitext
from safe_quant.evaluate_wildguard import _validate_benchmark_config as validate_wildguard
from safe_quant.wildguard_adapter import validate_frozen_protocol


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    ("finguard", "finguard_cached", validate_finguard),
    ("wikitext2", "wikitext2_cached", validate_wikitext),
    ("wildguard_prompt", "wildguard_prompt_cached", validate_wildguard),
    ("general_4task", "general_4task_cached_bs1", validate_general),
)


@pytest.mark.parametrize("original,cached,validate", VARIANTS)
def test_cached_variants_preserve_data_prompts_and_scoring(original, cached, validate):
    old = load_benchmark_config(ROOT / f"configs/benchmarks/{original}.yaml")
    new = load_benchmark_config(ROOT / f"configs/benchmarks/{cached}.yaml")
    validate(old)
    validate(new)
    assert new["protocol"]["use_cache"] is True
    assert old["protocol"].get("use_cache", False) is False
    assert old["protocol"]["version"] != new["protocol"]["version"]
    comparable = copy.deepcopy(new)
    comparable["protocol"]["version"] = old["protocol"]["version"]
    if "use_cache" in old["protocol"]:
        comparable["protocol"]["use_cache"] = old["protocol"]["use_cache"]
    else:
        comparable["protocol"].pop("use_cache")
    if original == "general_4task":
        assert new["protocol"]["batch_size"] == 1
        comparable["protocol"]["batch_size"] = old["protocol"]["batch_size"]
    assert comparable == old


@pytest.mark.parametrize("original,cached,validate", VARIANTS)
def test_original_protocol_cannot_silently_enable_cache(original, cached, validate):
    config = load_benchmark_config(ROOT / f"configs/benchmarks/{original}.yaml")
    config["protocol"]["use_cache"] = True
    with pytest.raises(ValueError):
        validate(config)


@pytest.mark.parametrize("original,cached,validate", VARIANTS)
def test_cached_protocol_cannot_silently_disable_cache(original, cached, validate):
    config = load_benchmark_config(ROOT / f"configs/benchmarks/{cached}.yaml")
    config["protocol"]["use_cache"] = False
    with pytest.raises(ValueError):
        validate(config)


def test_cached_wildguard_keeps_candidate_ids_and_threshold():
    config = load_benchmark_config(ROOT / "configs/benchmarks/wildguard_prompt_cached.yaml")
    validate_frozen_protocol(config["protocol"])
    config["protocol"]["tie_policy"] = "harmful_on_exact_tie"
    with pytest.raises(ValueError):
        validate_frozen_protocol(config["protocol"])


def test_wildguard_train_validation_remains_uncached():
    config = load_benchmark_config(ROOT / "configs/benchmarks/wildguard_train_validation.yaml")
    config["protocol"].update(version="wildguard_llama_yes_no_cached_v1", use_cache=True)
    with pytest.raises(ValueError, match="retains the original uncached"):
        validate_wildguard(config)


def test_wildguard_smoke_subset_is_recorded_before_model_loading(tmp_path, monkeypatch):
    import safe_quant.evaluate_wildguard as module

    monkeypatch.setenv("SAFE_QUANT_MODEL_ROOT", "/models")
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-unit-test-stops-before-model-loading")
    monkeypatch.setattr(module, "_require_slurm_gpu", lambda: None)
    examples = [{"example_id": "first", "dataset_index_after_filtering": 2},
                {"example_id": "second", "dataset_index_after_filtering": 3}]
    metadata = {"num_rows_selected": 2, "selected_indices_after_filtering": [2, 3],
                "selection": {"method": "all"}}
    monkeypatch.setattr(module, "load_wildguard_prompts", lambda config: (examples, metadata))
    args = SimpleNamespace(
        batch_size=1, max_samples=1,
        model_config=str(ROOT / "configs/models/llama3_8b_instruct_flatquant_w4a4kv4.yaml"),
        benchmark_config=str(ROOT / "configs/benchmarks/wildguard_prompt_cached.yaml"),
        output_dir=str(tmp_path / "smoke"),
    )

    class StopBeforeModel(Exception):
        pass

    def stop(config):
        raise StopBeforeModel()

    monkeypatch.setattr(module, "load_model", stop)
    with pytest.raises(StopBeforeModel):
        module.run(args)
    manifest = json.loads((tmp_path / "smoke/run-manifest.json").read_text())
    assert manifest["runtime"]["max_samples"] == 1
    assert manifest["dataset"]["num_rows_selected"] == 1
    assert manifest["dataset"]["selected_indices_after_filtering"] == [2]
    assert manifest["dataset"]["selection"] == {"method": "first_n_smoke", "max_samples": 1}
    assert metadata["num_rows_selected"] == 2


def test_flatquant_benchmark_launcher_refuses_login_before_environment_setup(tmp_path):
    environment = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_flatquant_bench.sh"), "scoring", "smoke"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Slurm GPU allocation" in result.stderr
    assert not list(tmp_path.iterdir())


def test_two_smoke_groups_and_separate_full_generation_paths():
    launcher = (ROOT / "scripts/run_flatquant_bench.sh").read_text()
    assert '"$group" == generation && "$mode" == full' in launcher
    assert "--max-blocks 2" in launcher and "--limit 2" in launcher
    assert "--max-samples 4" in launcher
    assert "0,1" in launcher and "0,25" in launcher and "0,156" in launcher
    assert "strongreject_cached_256.yaml" in launcher
    assert "strongreject_small" not in launcher
    assert launcher.count("--batch-size 4") == 1
    assert launcher.count("--batch-size 2") == 1
    assert "strongreject_rubric" not in launcher
    slurm = (ROOT / "slurm/flatquant_bench.sbatch").read_text()
    assert "--gpus-per-node=1" in slurm
    assert "--time=00:30:00" in slurm


@pytest.mark.parametrize("group,steps", [
    ("general", ["general_4task"]),
    ("finguard", ["finguard"]),
    ("wildguard", ["wildguard"]),
    ("scoring", ["wikitext2", "general_4task", "finguard", "wildguard"]),
])
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_flatquant_groups_route_only_to_selected_steps(tmp_path, group, steps, mode):
    """Execute shell routing only: ALL commands are replaced with a logging stub.

    No Python evaluator, model load, dataset load or GPU operation is executed.
    The isolated copy also cannot write to the real repository's outputs.
    """
    source = (ROOT / "scripts/run_flatquant_bench.sh").read_text()
    source, count = re.subn(r"(?m)^repository_root=.*$",
                            "repository_root=" + shlex.quote(str(tmp_path)), source)
    assert count == 1
    source, count = re.subn(r"(?m)^source .*/flatquant-eval/bin/activate$", ":", source)
    assert count == 1
    start = source.index("run_step() {")
    end = source.index("\npreflight_group=", start)
    source = source[:start] + '''run_step() {
    local name=$1
    shift
    printf '%s\\n' "$name" >> "$output_root/selected-steps.txt"
    printf '%s\\n' "$*" >> "$output_root/selected-commands.txt"
}
''' + source[end:]
    script = tmp_path / "routing-only.sh"
    script.write_text(source)
    environment = {**os.environ, "SLURM_JOB_ID": "cpu-routing-test-no-inference"}
    result = subprocess.run(["bash", str(script), group, mode, "unit"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = tmp_path / f"outputs/flatquant/{group}/{mode}-unit"
    expected = ["preflight"] + (["inference"] if mode == "smoke" else []) + steps
    assert (output / "selected-steps.txt").read_text().splitlines() == expected
    commands = (output / "selected-commands.txt").read_text()
    assert f"--group {group}" in commands
    if group != "scoring":
        assert "evaluate_wikitext2.py" not in commands
    classifier_commands = [line for line in commands.splitlines()
                           if "scripts/evaluate.py " in line or "scripts/evaluate_wildguard.py " in line]
    assert len(classifier_commands) == sum(step in {"finguard", "wildguard"} for step in steps)
    for command in classifier_commands:
        expected_batch_size = 2 if "scripts/evaluate_wildguard.py " in command else 4
        assert f"--batch-size {expected_batch_size}" in command
    if group in {"finguard", "wildguard"}:
        assert ("--max-samples 4" in commands) == (mode == "smoke")
    if group == "general":
        assert "general_4task_cached_bs1.yaml" in commands
        assert ("--limit 2" in commands) == (mode == "smoke")
