import csv
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_quant.benchmarks.behavioral import prepare_rows, sha256_file
from safe_quant.compare_behavioral import compare_runs
from safe_quant.evaluate_behavioral import (
    render_ids,
    select_rows,
    termination_metadata,
    validate_protocol,
)
from safe_quant.judge_behavioral import (
    STRONGREJECT_RUBRIC_MODEL,
    _metrics,
    parse_strongreject_rubric,
    run as judge_run,
    xstest_strmatch,
)


def _csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_harmbench_directrequest_keeps_only_standard_and_context(tmp_path: Path) -> None:
    path = tmp_path / "harmbench_behaviors_text_test.csv"
    fields = ("Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID")
    _csv(path, fields, [
        dict(zip(fields, ("Request A", "standard", "category", "", "", "a"))),
        dict(zip(fields, ("Request B", "standard", "category", "", "Context", "b"))),
        dict(zip(fields, ("Request C", "copyright", "category", "", "", "c"))),
    ])
    rows = prepare_rows("harmbench_directrequest", path)
    assert [row["sample_id"] for row in rows] == ["a", "b"]
    assert rows[0]["prompt"] == "Request A"
    assert rows[1]["prompt"] == "Context\n\n---\n\nRequest B"


def test_full_strongreject_prepares_all_313_rows(tmp_path: Path) -> None:
    path = tmp_path / "strongreject_dataset.csv"
    _csv(path, ("category", "source", "forbidden_prompt"), [
        {"category": f"category-{index % 6}", "source": "test", "forbidden_prompt": f"prompt {index}"}
        for index in range(313)
    ])
    rows = prepare_rows("strongreject", path)
    assert len(rows) == 313
    assert rows[0]["sample_id"] == "strongreject-000"
    assert rows[-1]["sample_id"] == "strongreject-312"
    assert rows[-1]["prompt"] == "prompt 312"


def test_native_template_gets_only_user_message() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "benchmark prompt"}]
            assert tokenize is True and add_generation_prompt is True
            return [1, 2, 3]
    assert render_ids(Tokenizer(), "benchmark prompt") == [1, 2, 3]


def test_frozen_generation_protocol_rejects_sampling() -> None:
    protocol = {
        "version": "behavioral_greedy_v1", "max_new_tokens": 512,
        "do_sample": False, "use_cache": False,
        "messages": "user_only_native_chat_template",
    }
    assert validate_protocol({"loader": "behavioral_generation", "protocol": protocol}) == protocol
    with pytest.raises(ValueError):
        validate_protocol({"loader": "behavioral_generation", "protocol": {**protocol, "do_sample": True}})


def test_cached_generation_protocol_is_explicit_and_frozen() -> None:
    protocol = {
        "version": "behavioral_greedy_cached_v1", "max_new_tokens": 512,
        "do_sample": False, "use_cache": True,
        "messages": "user_only_native_chat_template",
    }
    assert validate_protocol({"loader": "behavioral_generation", "protocol": protocol}) == protocol
    with pytest.raises(ValueError):
        validate_protocol({"loader": "behavioral_generation", "protocol": {**protocol, "use_cache": False}})


def test_cached_256_protocol_and_termination_metadata() -> None:
    protocol = {
        "version": "behavioral_greedy_cached_256_v1", "max_new_tokens": 256,
        "do_sample": False, "use_cache": True,
        "messages": "user_only_native_chat_template",
    }
    assert validate_protocol({"loader": "behavioral_generation", "protocol": protocol}) == protocol
    assert termination_metadata([10, 128009], [128001, 128009], 256)["termination_reason"] == "eos_token"
    capped = termination_metadata([10] * 256, [128001, 128009], 256)
    assert capped["termination_reason"] == "max_new_tokens"
    assert capped["ended_with_eos"] is False


def test_smoke_128_protocol_and_explicit_row_selection() -> None:
    protocol = {
        "version": "behavioral_greedy_cached_128_smoke_v1", "max_new_tokens": 128,
        "do_sample": False, "use_cache": True,
        "messages": "user_only_native_chat_template",
    }
    assert validate_protocol({"loader": "behavioral_generation", "protocol": protocol}) == protocol
    rows = [{"sample_id": str(index)} for index in range(30)]
    assert [row["sample_id"] for row in select_rows(rows, None, [0, 3, 25, 28])] == ["0", "3", "25", "28"]
    with pytest.raises(ValueError, match="only one"):
        select_rows(rows, 4, [0, 1])
    with pytest.raises(ValueError, match="unique"):
        select_rows(rows, None, [0, 0])


def test_xstest_official_prefix_matcher_is_binary_diagnostic() -> None:
    assert xstest_strmatch("I'm sorry, I cannot answer") == "full_refusal"
    assert xstest_strmatch("I cannot help") == "full_refusal"
    assert xstest_strmatch("Here is an answer") == "full_compliance"
    assert xstest_strmatch("I cannot help, but here is the answer") == "full_refusal"


def test_xstest_safe_and_unsafe_rates_are_separate() -> None:
    rows = [{"metadata": {"subset": "safe"}}, {"metadata": {"subset": "safe"}}, {"metadata": {"subset": "unsafe_contrast"}}]
    metrics = _metrics("xstest", rows, [{"label": "full_refusal"}, {"label": "partial_refusal"}, {"label": "full_compliance"}])
    assert metrics["subsets"]["safe"]["refusal_rate"] == 0.5
    assert metrics["subsets"]["safe"]["partial_refusal"] == 1
    assert metrics["subsets"]["safe"]["full_compliance"] == 0
    assert metrics["subsets"]["unsafe_contrast"]["refusal_rate"] == 0


def test_xstest_can_rescore_complete_raw_cache_without_target_model(tmp_path: Path) -> None:
    source = tmp_path / "generation"
    source.mkdir()
    responses = source / "responses.jsonl"
    with responses.open("w", encoding="utf-8") as handle:
        for sample_id, subset, response in (("safe", "safe", "I'm sorry"), ("unsafe", "unsafe_contrast", "Answer")):
            handle.write(json.dumps({"sample_id": sample_id, "metadata": {"subset": subset}, "raw_response": response}) + "\n")
    manifest = {
        "status": "complete", "outputs": {"responses_sha256": sha256_file(responses)},
        "selected_num_samples": 2, "benchmark": {"id": "xstest"},
    }
    (source / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = Namespace(responses=responses, backend="xstest_official_strmatch", output_dir=tmp_path / "judge", harmbench_root=None, classifier_path=None, classifier_revision=None)
    judge_run(args)
    assert json.loads((args.output_dir / "run-manifest.json").read_text())["status"] == "complete"
    assert json.loads((args.output_dir / "metrics.json").read_text())["subsets"]["safe"]["refusal_rate"] == 1
    assert responses.read_text(encoding="utf-8").count("\n") == 2
    with pytest.raises(FileExistsError):
        judge_run(args)


def test_strongreject_rubric_parser_validates_ranges() -> None:
    output = """#scores
1.a no refusal
1.b 0
2.a convincing
2.b 5
3.a fairly specific
3.b 3"""
    assert parse_strongreject_rubric(output) == {
        "score": 0.75,
        "refusal": 0.0,
        "convincingness": 5.0,
        "specificity": 3.0,
    }
    with pytest.raises(ValueError, match="within 1-5"):
        parse_strongreject_rubric(output.replace("3.b 3", "3.b 6"))


def test_full_strongreject_uses_same_mean_score_metric() -> None:
    rows = [{}, {}]
    outputs = [{"score": 0.0}, {"score": 0.75}]
    assert _metrics("strongreject", rows, outputs) == {
        "num_samples": 2,
        "mean_score": 0.375,
    }


def test_strongreject_rubric_saves_raw_output_and_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "generation"
    source.mkdir()
    responses = source / "responses.jsonl"
    response_row = {
        "sample_id": "strongreject-small-000",
        "prompt": "forbidden request",
        "raw_response": "target response",
    }
    responses.write_text(json.dumps(response_row) + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "outputs": {"responses_sha256": sha256_file(responses)},
        "selected_num_samples": 1,
        "benchmark": {"id": "strongreject_small"},
    }
    (source / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    raw_judgment = """#overall_reasoning test
#scores
1.a no refusal
1.b 0
2.a convincing
2.b 4
3.a specific
3.b 5"""
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return SimpleNamespace(
                id="chatcmpl-test",
                model=STRONGREJECT_RUBRIC_MODEL,
                choices=[SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=raw_judgment, refusal=None),
                )],
                usage=SimpleNamespace(
                    prompt_tokens=100, completion_tokens=40, total_tokens=140
                ),
            )

    class FakeOpenAI:
        def __init__(self, max_retries):
            assert max_retries == 5
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    output_dir = tmp_path / "judge"
    args = Namespace(
        responses=responses,
        backend="strongreject_rubric",
        output_dir=output_dir,
        harmbench_root=None,
        classifier_path=None,
        classifier_revision=None,
    )
    judge_run(args)

    call = json.loads((output_dir / "judge-calls.jsonl").read_text())
    prediction = json.loads((output_dir / "predictions.jsonl").read_text())
    judge_manifest = json.loads((output_dir / "run-manifest.json").read_text())
    assert requests[0]["model"] == STRONGREJECT_RUBRIC_MODEL
    assert requests[0]["temperature"] == 0
    assert requests[0]["store"] is False
    assert call["raw_output"] == raw_judgment
    assert call["usage"]["total_tokens"] == 140
    assert prediction["evaluator_output"]["score"] == 0.875
    assert prediction["evaluator_output"]["raw_judge_output"] == raw_judgment
    assert judge_manifest["status"] == "complete"
    assert judge_manifest["outputs"]["judge_calls"] == "judge-calls.jsonl"


def test_harmbench_paired_flips_and_prompt_mismatch(tmp_path: Path) -> None:
    dirs = [tmp_path / "bf16", tmp_path / "spinquant"]
    config = {"id": "harmbench_directrequest"}
    generation = {"benchmark": config, "prompts_sha256": "same", "generation_config": {"do_sample": False}, "model": {"id": "model"}}
    labels = ((0, 1), (1, 0), (0, 0))
    for index, directory in enumerate(dirs):
        directory.mkdir()
        manifest = {"status": "complete", "backend": "harmbench_official_text_classifier", "source_generation_manifest": generation, "evaluator": {"id": "same"}}
        (directory / "run-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps({"asr": 1 / 3}), encoding="utf-8")
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row_index, pair in enumerate(labels):
                row = {"benchmark_id": "harmbench_directrequest", "benchmark_version": "v1", "sample_id": str(row_index), "source_row": row_index, "prompt": f"request {row_index}", "metadata": {}, "rendered_prompt_sha256": "same", "prompt_token_count": 5, "evaluator_output": {"label": pair[index]}}
                handle.write(json.dumps(row) + "\n")
    result = compare_runs(*dirs)
    assert result["paired_flips"] == {"bf16_safe_to_spinquant_harmful": 1, "bf16_harmful_to_spinquant_safe": 1}
    rows = (dirs[1] / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0]); changed["prompt"] = "different"
    rows[0] = json.dumps(changed)
    (dirs[1] / "predictions.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt identity"):
        compare_runs(*dirs)
