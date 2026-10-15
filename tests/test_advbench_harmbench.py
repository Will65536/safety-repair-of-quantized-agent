"""CPU-only adapter tests: native classifier subprocesses are always stubbed."""

from argparse import Namespace
import copy
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from safe_quant.benchmarks import advbench_harmbench as adapter
from safe_quant.benchmarks.behavioral import prepare_rows, sha256_file
from safe_quant.compare_behavioral import compare_runs
from safe_quant.judge_behavioral import _harmbench, _metrics, run as judge_run


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "outputs/behavioral/prompts/advbench_directrequest-cached-256-v1/prompt-manifest.json"


@pytest.fixture
def cached():
    prepared = json.loads(PREPARED.read_text())
    rows = prepare_rows("advbench_directrequest", Path(prepared["source_file"]))
    for row in rows:
        row["raw_response"] = "I cannot answer"
    manifest = {
        "status": "complete", "benchmark": prepared["benchmark"], "prepared_prompts": prepared,
        "is_full_evaluation": True, "selected_num_samples": 520,
        "prompts_sha256": prepared["prompts_sha256"],
        "generation_config": {"batch_size": 4, "max_new_tokens": 256},
        "model": {"id": "model"},
    }
    return rows, manifest


def test_all_original_goals_map_to_standard_behaviors_without_targets(cached):
    rows, manifest = cached
    behaviors = adapter.standard_behaviors(rows, manifest)
    assert len(behaviors) == 520
    assert [b["Behavior"] for b in behaviors] == [r["prompt"] for r in rows]
    assert [b["BehaviorID"] for b in behaviors] == [r["sample_id"] for r in rows]
    assert all(b["Tags"] == b["ContextString"] == "" for b in behaviors)
    assert all(b["FunctionalCategory"] == "standard" for b in behaviors)
    assert all(set(b) == set(adapter.BEHAVIOR_FIELDS) for b in behaviors)
    # Every saved response here contains a refusal phrase; none are prefiltered.
    assert all(r["raw_response"] == "I cannot answer" for r in rows)


@pytest.mark.parametrize("change", ["goal", "id", "source_row", "duplicate", "full_subset", "revision", "source_hash"])
def test_adapter_rejects_changed_identity_or_provenance(cached, change):
    rows, manifest = cached
    if change == "goal": rows[0]["prompt"] += " changed"
    elif change == "id": rows[0]["sample_id"] = "unknown"
    elif change == "source_row": rows[0]["source_row"] = 999
    elif change == "duplicate": rows[1]["sample_id"] = rows[0]["sample_id"]
    elif change == "full_subset": rows = rows[:2]
    elif change == "revision": manifest["benchmark"]["dataset"]["revision"] = "0" * 40
    elif change == "source_hash": manifest["prepared_prompts"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        adapter.standard_behaviors(rows, manifest)


def test_adapter_allows_matching_smoke_subset(cached):
    rows, manifest = cached
    manifest["is_full_evaluation"] = False
    assert len(adapter.standard_behaviors(rows[:8], manifest)) == 8


def test_binary_asr_is_distinct_from_string_matching_score():
    result = _metrics("advbench_directrequest", [{}, {}, {}],
                      [{"label": 0}, {"label": 1}, {"label": 0}],
                      backend="advbench_harmbench_classifier")
    assert result["asr"] == 1 / 3
    assert result["classifier_positive_count"] == 1
    assert result["classifier_negative_count"] == 2
    assert "non_refusal_rate" not in result
    with pytest.raises(ValueError, match="Invalid"):
        _metrics("advbench_directrequest", [{}], [{"label": -1}], backend="advbench_harmbench_classifier")


@pytest.fixture
def stub_native(tmp_path, monkeypatch):
    """Prepare only dummy files; no vLLM imports, GPU calls or subprocesses."""
    import safe_quant.judge_behavioral as module
    root = tmp_path / "harmbench-checkout"
    root.mkdir()
    for name in ("evaluate_completions.py", "eval_utils.py"):
        (root / name).write_text("# native evaluator stub: never executed\n")
    monkeypatch.setattr(adapter, "EVALUATE_SHA256", sha256_file(root / "evaluate_completions.py"))
    monkeypatch.setattr(adapter, "EVAL_UTILS_SHA256", sha256_file(root / "eval_utils.py"))
    classifier = tmp_path / "snapshots" / adapter.CLASSIFIER_REVISION
    classifier.mkdir(parents=True)
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-subprocess-stub-only")

    def fake_subprocess(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return SimpleNamespace(stdout=adapter.HARMBENCH_REVISION)
        assert command[1] == "evaluate_completions.py"
        assert command[command.index("--num_tokens") + 1] == "512"
        assert kwargs["cwd"] == root
        completion_path = Path(command[command.index("--completions_path") + 1])
        behavior_path = Path(command[command.index("--behaviors_path") + 1])
        result_path = Path(command[command.index("--save_path") + 1])
        assert all(path.is_absolute() for path in (completion_path, behavior_path, result_path))
        completions = json.loads(completion_path.read_text())
        with behavior_path.open(newline="") as handle:
            behaviors = list(csv.DictReader(handle))
        assert len(behaviors) == len(completions)
        assert all(b["Behavior"] == completions[b["BehaviorID"]][0]["test_case"] for b in behaviors)
        result_path.write_text(json.dumps({
            sample_id: [{"label": int(i == 0)}] for i, sample_id in enumerate(completions)
        }))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_subprocess)
    return root, classifier, calls


def test_full_cached_rescore_keeps_sources_and_proxy_results_immutable(tmp_path, cached, stub_native):
    rows, generation = cached
    root, classifier, calls = stub_native
    source = tmp_path / "generation"
    source.mkdir()
    responses = source / "responses.jsonl"
    responses.write_text("".join(json.dumps(r) + "\n" for r in rows))
    before = sha256_file(responses)
    generation["outputs"] = {"responses_sha256": before}
    (source / "run-manifest.json").write_text(json.dumps(generation))
    proxy = tmp_path / "strmatch"
    proxy.mkdir()
    (proxy / "metrics.json").write_text('{"preserved": true}')
    args = Namespace(responses=responses, output_dir=tmp_path / "harmbench-run1",
                     backend="advbench_harmbench_classifier", harmbench_root=root,
                     classifier_path=str(classifier), classifier_revision=adapter.CLASSIFIER_REVISION)
    judge_run(args)
    metrics = json.loads((args.output_dir / "metrics.json").read_text())
    manifest = json.loads((args.output_dir / "run-manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert metrics["num_samples"] == 520 and metrics["asr"] == 1 / 520
    assert len((args.output_dir / "predictions.jsonl").read_text().splitlines()) == 520
    assert manifest["evaluator"]["id"] == "advbench_harmbench_classifier"
    assert manifest["evaluator"]["repository_revision"] == adapter.HARMBENCH_REVISION
    assert manifest["evaluator"]["classifier_revision"] == adapter.CLASSIFIER_REVISION
    assert manifest["evaluator"]["classifier_decoding"]["max_tokens"] == 1
    assert sha256_file(responses) == before
    assert (proxy / "metrics.json").read_text() == '{"preserved": true}'
    assert len(calls) == 2  # git lookup plus classifier invocation, both stubbed.
    with pytest.raises(FileExistsError):
        judge_run(args)


@pytest.mark.parametrize("failure", ["invalid_label", "missing_id", "classifier_revision", "code_hash"])
def test_native_judge_fails_closed_on_inconsistent_outputs(tmp_path, cached, stub_native, monkeypatch, failure):
    import safe_quant.judge_behavioral as module
    rows, generation = cached
    root, classifier, calls = stub_native
    output = tmp_path / "judged"
    output.mkdir()
    args = Namespace(generation_manifest=generation, harmbench_root=root,
                     classifier_path=str(classifier), classifier_revision=adapter.CLASSIFIER_REVISION)
    fake_run = module.subprocess.run

    def changed_output(command, **kwargs):
        result = fake_run(command, **kwargs)
        if command[0] != "git":
            p = Path(command[command.index("--save_path") + 1])
            data = json.loads(p.read_text())
            if failure == "invalid_label": data[next(iter(data))][0]["label"] = -1
            if failure == "missing_id": data.pop(next(iter(data)))
            p.write_text(json.dumps(data))
        return result

    monkeypatch.setattr(module.subprocess, "run", changed_output)
    if failure == "classifier_revision": args.classifier_revision = "0" * 40
    if failure == "code_hash": (root / "eval_utils.py").write_text("# changed source\n")
    with pytest.raises(ValueError):
        _harmbench(rows, args, output)


def test_classifier_requires_slurm_before_any_model_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm GPU"):
        _harmbench([], Namespace(), tmp_path)


@pytest.mark.parametrize("model_ids", [("spinquant", "flatquant"), ("bf16", "flatquant"), ("bf16", "spinquant")])
def test_classifier_paired_comparison_and_backend_mismatch(tmp_path, cached, model_ids):
    rows, generation = cached
    dirs = [tmp_path / model for model in model_ids]
    for index, path in enumerate(dirs):
        path.mkdir()
        gen = copy.deepcopy(generation)
        gen["model"]["id"] = path.name
        manifest = {"status": "complete", "backend": "advbench_harmbench_classifier",
                    "source_generation_manifest": gen, "evaluator": {"id": "same-pinned-judge"}}
        (path / "run-manifest.json").write_text(json.dumps(manifest))
        labels = [0, 1] if index == 0 else [1, 0]
        judged = []
        for row, label in zip(rows, labels):
            judged.append({**row, "benchmark_id": "advbench_directrequest", "benchmark_version": "v1",
                           "rendered_prompt_sha256": "same", "prompt_token_count": 10,
                           "evaluator_output": {"label": label}})
        (path / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in judged))
        (path / "metrics.json").write_text(json.dumps({"asr": 0.5}))
    result = compare_runs(*dirs)
    assert result["paired_flips"] == {"baseline_negative_to_variant_positive": 1,
                                       "baseline_positive_to_variant_negative": 1}
    assert result["asr_difference_variant_minus_baseline"] == 0
    assert (result["baseline_model"], result["variant_model"]) == model_ids
    path = dirs[1] / "run-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["backend"] = "advbench_original_strmatch"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different evaluators"):
        compare_runs(*dirs)


def test_judge_launcher_refuses_login(tmp_path):
    environment = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
    result = subprocess.run(["bash", str(ROOT / "scripts/judge_advbench_harmbench.sh"), "flatquant"],
                            cwd=tmp_path, env=environment, text=True, capture_output=True)
    assert result.returncode == 2 and "Slurm GPU" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model", ["bf16", "flatquant", "spinquant"])
def test_judge_launcher_routes_only_to_saved_responses(tmp_path, model):
    source = (ROOT / "scripts/judge_advbench_harmbench.sh").read_text()
    source, count = re.subn(r"(?m)^repository_root=.*$", "repository_root=" + shlex.quote(str(tmp_path)), source)
    assert count == 1
    source, count = re.subn(r"(?m)^source .*/harmbench-judge/bin/activate$", ":", source)
    assert count == 1
    position = source.index("\npython -u -B scripts/judge_behavioral.py")
    source = source[:position] + '''
python() { printf '%s\\n' "$*" > "$run_root/command.txt"; }
''' + source[position:]
    script = tmp_path / "routing.sh"
    script.write_text(source)
    run = tmp_path / f"outputs/behavioral/advbench_directrequest/{model}/full-bs4-run1"
    (run / "generation").mkdir(parents=True)
    (run / "generation/responses.jsonl").write_text('{}\n')
    result = subprocess.run(["bash", str(script), model], cwd=tmp_path,
                            env={**os.environ, "SLURM_JOB_ID": "cpu-routing-only"},
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    command = (run / "command.txt").read_text()
    assert "--backend advbench_harmbench_classifier" in command
    assert str(run / "generation/responses.jsonl") in command
    assert str(run / "judgments/harmbench-run1") in command
    assert adapter.CLASSIFIER_REVISION in command
    assert "evaluate_behavioral.py" not in command and "sbatch" not in command
