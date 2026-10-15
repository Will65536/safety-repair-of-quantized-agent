"""AdvBench verification is CPU-only; GPU execution is left to the user."""

import ast
import copy
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from argparse import Namespace

import pytest

from safe_quant.benchmarks.advbench import (
    ATTACK_MANAGER_SHA256, EVALUATE_SHA256, REFUSAL_STRINGS,
    evaluator_metadata, original_strmatch,
)
from safe_quant.benchmarks.behavioral import prepare_rows, sha256_file
from safe_quant.compare_behavioral import compare_runs
from safe_quant.config import load_benchmark_config
from safe_quant.evaluate_behavioral import render_ids, validate_protocol
from safe_quant.judge_behavioral import _metrics, run as judge_run


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/behavioral/sources/advbench"
CONFIG = ROOT / "configs/benchmarks/advbench_directrequest_cached_256.yaml"


def test_pinned_data_and_prompt_identity():
    config = load_benchmark_config(CONFIG)
    validate_protocol(config)
    assert sha256_file(SOURCE / "harmful_behaviors.csv") == config["dataset"]["sha256"]
    rows = prepare_rows("advbench_directrequest", SOURCE / "harmful_behaviors.csv")
    with (SOURCE / "harmful_behaviors.csv").open(newline="") as handle:
        original = list(csv.DictReader(handle))
    assert len(rows) == 520
    assert [r["sample_id"] for r in rows] == [f"advbench-{i:03d}" for i in range(520)]
    assert [r["prompt"] for r in rows] == [r["goal"] for r in original]
    assert [r["metadata"]["target"] for r in rows] == [r["target"] for r in original]
    prepared = ROOT / "outputs/behavioral/prompts/advbench_directrequest-cached-256-v1"
    manifest = json.loads((prepared / "prompt-manifest.json").read_text())
    assert manifest["benchmark"] == config
    assert manifest["prompts_sha256"] == sha256_file(prepared / "prompts.jsonl")
    assert [json.loads(line) for line in (prepared / "prompts.jsonl").open()] == rows


def test_advbench_requires_all_520_rows(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("goal,target\nrequest,reference\n")
    with pytest.raises(ValueError, match="Expected 520"):
        prepare_rows("advbench_directrequest", path)


def test_prompt_rendering_never_prefills_target():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "request"}]
            assert kwargs == {"tokenize": True, "add_generation_prompt": True}
            return [1, 2]
    assert render_ids(Tokenizer(), "request") == [1, 2]


def test_refusal_list_matches_pinned_official_source():
    source = SOURCE / "upstream_evaluate.py"
    assert sha256_file(source) == EVALUATE_SHA256
    assert sha256_file(SOURCE / "upstream_attack_manager.py") == ATTACK_MANAGER_SHA256
    assigned = [node for node in ast.parse(source.read_text()).body
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "_test_prefixes" for t in node.targets)]
    assert tuple(ast.literal_eval(assigned[0].value)) == REFUSAL_STRINGS
    assert evaluator_metadata()["refusal_strings_sha256"] == "f7f0a2445e1954435251e8eb9bdd84049807c58c3fe663f6c45c03c57dbf8358"
    # Compare against the actual upstream expression, not a second hand-written matcher.
    tree = ast.parse((SOURCE / "upstream_attack_manager.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EvaluateAttack")
    run = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    assignment = next(n for n in ast.walk(run) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "jailbroken" for t in n.targets))
    expression = compile(ast.Expression(assignment.value), "upstream-strmatch", "eval")
    from types import SimpleNamespace
    for text in ("I cannot answer", "A preamble. I cannot answer", "i cannot answer", "", "Hello!", "Answer", "Discuss illegal activity"):
        official = eval(expression, {"self": SimpleNamespace(test_prefixes=list(REFUSAL_STRINGS)), "gen_str": text})
        assert (original_strmatch(text)["label"] == "non_refusal") == official


@pytest.mark.parametrize("text,label", [
    ("I cannot answer", "refusal"),
    ("A preamble. I cannot answer", "refusal"),
    ("i cannot answer", "non_refusal"),
    ("I am unable to answer", "non_refusal"),
    ("Hello!", "refusal"),
    ("Discuss illegal activity", "refusal"),
    ("Answer", "non_refusal"),
    ("", "non_refusal"),
])
def test_original_matching_limitations_are_explicit(text, label):
    assert original_strmatch(text)["label"] == label


def test_metrics_label_empty_outputs_without_changing_upstream_score():
    outputs = [original_strmatch(text) for text in ("I cannot answer", "Answer", "")]
    result = _metrics("advbench_directrequest", [{}, {}, {}], outputs)
    assert result["refusal_count"] == 1 and result["non_refusal_count"] == 2
    assert result["non_refusal_rate"] == 2 / 3
    assert result["empty_response_count"] == 1
    assert "asr" not in result


def test_cpu_judging_does_not_load_target_or_need_slurm(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    source = tmp_path / "generation"
    source.mkdir()
    responses = source / "responses.jsonl"
    responses.write_text("".join(json.dumps({"sample_id": str(i), "raw_response": text}) + "\n"
                                 for i, text in enumerate(("I cannot answer", "Answer"))))
    manifest = {"status": "complete", "outputs": {"responses_sha256": sha256_file(responses)},
                "selected_num_samples": 2, "benchmark": {"id": "advbench_directrequest"}}
    (source / "run-manifest.json").write_text(json.dumps(manifest))
    args = Namespace(responses=responses, backend="advbench_original_strmatch",
                     output_dir=tmp_path / "judged")
    judge_run(args)
    result = json.loads((args.output_dir / "metrics.json").read_text())
    assert result["non_refusal_rate"] == 0.5
    assert len((args.output_dir / "predictions.jsonl").read_text().splitlines()) == 2
    judged = json.loads((args.output_dir / "run-manifest.json").read_text())
    assert judged["status"] == "complete"
    assert judged["evaluator"]["repository_revision"] == evaluator_metadata()["repository_revision"]
    with pytest.raises(FileExistsError):
        judge_run(args)
    responses.write_text(responses.read_text() + "\n")
    with pytest.raises(ValueError, match="responses changed"):
        judge_run(args)


def test_advbench_paired_comparison_preserves_model_ids_and_settings(tmp_path):
    dirs = [tmp_path / "spinquant", tmp_path / "flatquant"]
    common = {"benchmark": {"id": "advbench_directrequest"}, "prompts_sha256": "same",
              "generation_config": {"max_new_tokens": 256, "batch_size": 1,
                                    "effective_generation_config": {"do_sample": False}}}
    for i, directory in enumerate(dirs):
        directory.mkdir()
        generation = copy.deepcopy(common)
        generation["model"] = {"id": directory.name}
        generation["generation_config"]["effective_generation_config"].update(
            transformers_version="4.44.2" if i == 0 else "4.45.0",
            return_legacy_cache=True if i == 0 else None,
            num_assistant_tokens=5 if i == 0 else 20)
        manifest = {"status": "complete", "backend": "advbench_original_strmatch",
                    "source_generation_manifest": generation, "evaluator": evaluator_metadata()}
        (directory / "run-manifest.json").write_text(json.dumps(manifest))
        output = original_strmatch("I cannot answer" if i == 0 else "Answer")
        (directory / "metrics.json").write_text(json.dumps(_metrics("advbench_directrequest", [{}], [output])))
        row = {"benchmark_id": "advbench_directrequest", "benchmark_version": "v1",
               "sample_id": "0", "source_row": 0, "prompt": "request", "metadata": {},
               "rendered_prompt_sha256": "same", "prompt_token_count": 5, "evaluator_output": output}
        (directory / "predictions.jsonl").write_text(json.dumps(row) + "\n")
    result = compare_runs(*dirs)
    assert result["baseline_model"] == "spinquant" and result["variant_model"] == "flatquant"
    assert result["paired_flips"]["baseline_refusal_to_variant_non_refusal"] == 1
    assert result["non_refusal_rate_difference_variant_minus_baseline"] == 1
    manifest_path = dirs[1] / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    generation = manifest["source_generation_manifest"]["generation_config"]
    generation["max_new_tokens"] = 128
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="generation settings"):
        compare_runs(*dirs)
    generation["max_new_tokens"] = 256
    generation["effective_generation_config"]["is_assistant"] = True
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-assistant"):
        compare_runs(*dirs)


def test_launcher_refuses_login_before_any_mutation(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    result = subprocess.run(["bash", str(ROOT / "scripts/run_advbench.sh"), "flatquant", "smoke"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "Slurm GPU allocation" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", ["bf16", "flatquant", "spinquant"])
@pytest.mark.parametrize("mode", ["smoke", "full"])
@pytest.mark.parametrize("batch_size", [None, 4])
def test_launcher_routing_only_no_models_or_jobs(tmp_path, model, mode, batch_size):
    source = (ROOT / "scripts/run_advbench.sh").read_text()
    source, count = re.subn(r"(?m)^repository_root=.*$", "repository_root=" + shlex.quote(str(tmp_path)), source)
    assert count == 1
    source, count = re.subn(r"(?m)^    source .*?/bin/activate$", "    :", source)
    assert count == 2
    start = source.index("run_step() {")
    end = source.index("\nrun_step preflight", start)
    source = source[:start] + '''run_step() {
    printf '%s\\n' "$*" >> "$output_root/commands.txt"
    printf '%s\\n' "$PYTHONPATH" >> "$output_root/pythonpaths.txt"
}
''' + source[end:]
    script = tmp_path / "routing.sh"
    script.write_text(source)
    arguments = ["bash", str(script), model, mode, "unit"]
    if batch_size is not None:
        arguments.append(str(batch_size))
    result = subprocess.run(arguments, cwd=tmp_path,
                            env={**os.environ, "SLURM_JOB_ID": "cpu-routing-only"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    output = tmp_path / f"outputs/behavioral/advbench_directrequest/{model}/{mode}-unit"
    commands = (output / "commands.txt").read_text()
    expected_path = str(tmp_path / "src")
    if model == "spinquant":
        expected_path += ":/work/nvme/bgxm/wliu12/repos/SpinQuant"
    assert set((output / "pythonpaths.txt").read_text().splitlines()) == {expected_path}
    expected_config = ("llama3_8b_instruct_bf16.yaml" if model == "bf16"
                       else f"llama3_8b_instruct_{model}_w4a4kv4.yaml")
    assert expected_config in commands
    if model == "bf16":
        assert "bf16_w4a4kv4.yaml" not in commands
    assert "advbench_directrequest_cached_256.yaml" in commands
    assert "--backend advbench_original_strmatch" in commands
    assert f"--batch-size {batch_size or 1}" in commands
    assert ("--sample-indices 0,74,148,222,296,370,444,519" in commands) == (mode == "smoke")
    assert "sbatch" not in commands and "srun" not in commands


@pytest.mark.parametrize("batch_size", ["0", "-1", "four", "1.5"])
def test_launcher_rejects_invalid_batch_before_environment_setup(tmp_path, batch_size):
    result = subprocess.run(["bash", str(ROOT / "scripts/run_advbench.sh"),
                             "flatquant", "smoke", "unit", batch_size], cwd=tmp_path,
                            env={**os.environ, "SLURM_JOB_ID": "cpu-validation-only"},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not list(tmp_path.iterdir())
