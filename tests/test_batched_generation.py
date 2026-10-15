"""CPU-only tests using tensor operations and a generate stub, never a model."""

from argparse import Namespace
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_quant.benchmarks.behavioral import read_jsonl, sha256_file
from safe_quant.evaluate_behavioral import (
    left_pad_generation_inputs, termination_metadata, trim_after_first_eos,
)


ROOT = Path(__file__).resolve().parents[1]


def test_mixed_length_left_padding_keeps_prompt_ids_and_masks():
    inputs, masks = left_pad_generation_inputs([[11, 12], [13, 14, 15], [16]], 9)
    assert inputs == [[9, 11, 12], [13, 14, 15], [9, 9, 16]]
    assert masks == [[0, 1, 1], [1, 1, 1], [0, 0, 1]]
    # A real EOS token inside a prompt is not padding: the mask is based on
    # lengths, not input_ids != pad_token_id (Llama uses EOT as its pad token).
    assert left_pad_generation_inputs([[9, 12], [13]], 9)[1] == [[1, 1], [0, 1]]


def test_single_example_padding_is_unchanged():
    assert left_pad_generation_inputs([[11, 12]], None) == ([[11, 12]], [[1, 1]])


@pytest.mark.parametrize("sequences", [[], [[]], [[1], []]])
def test_padding_rejects_empty_prompts(sequences):
    with pytest.raises(ValueError, match="nonempty"):
        left_pad_generation_inputs(sequences, 9)


def test_mixed_lengths_require_pad_token():
    with pytest.raises(ValueError, match="pad token"):
        left_pad_generation_inputs([[11], [12, 13]], None)


@pytest.mark.parametrize("eos", [9, [9, 10]])
def test_eos_padding_removed_and_eos_retained(eos):
    result = trim_after_first_eos([101, 9, 9, 9], eos)
    assert result == [101, 9]
    assert termination_metadata(result, eos, 256)["termination_reason"] == "eos_token"


def test_multiple_eos_and_output_cap():
    assert trim_after_first_eos([101, 10, 9, 9], [9, 10]) == [101, 10]
    result = trim_after_first_eos([101] * 256, [9, 10])
    assert len(result) == 256
    assert termination_metadata(result, [9, 10], 256)["termination_reason"] == "max_new_tokens"
    assert trim_after_first_eos([9, 9, 9], [9, 10]) == [9]


@pytest.mark.parametrize("batch_size", [None, 4])
def test_generation_loop_with_stub_handles_padding_eos_and_final_short_batch(
    tmp_path, monkeypatch, batch_size
):
    """No CUDA, model initialization, forward, real generation, or job execution."""
    import torch
    import transformers
    import safe_quant.evaluate_behavioral as module

    prompts = ROOT / "outputs/behavioral/prompts/advbench_directrequest-cached-256-v1/prompts.jsonl"
    rows = read_jsonl(prompts)[:5]
    prompt_ids = {row["prompt"]: [20 + i] * (i + 1) for i, row in enumerate(rows)}
    base = tmp_path / "base"
    base.mkdir()
    (base / "generation_config.json").write_text("{}")
    model_config = tmp_path / "model.yaml"
    model_config.write_text("id: stub\n")
    model_metadata = {"id": "stub", "model_path": str(base)}
    calls = []
    original_tensor = torch.tensor

    def cpu_tensor(data, **kwargs):
        # Redirect the evaluator's tensor allocations; never access a GPU.
        assert kwargs.pop("device", "cpu") in ("cuda:0", "cpu")
        return original_tensor(data, **kwargs)

    class Tokenizer:
        pad_token_id = 9

        def apply_chat_template(self, messages, **kwargs):
            assert len(messages) == 1 and messages[0]["role"] == "user"
            return prompt_ids[messages[0]["content"]]

        def decode(self, ids, **kwargs):
            return " ".join(str(token) for token in ids if token not in (9, 10))

    class FakeGenerationConfig(SimpleNamespace):
        def to_dict(self):
            return vars(self).copy()

    class GenerateStub:
        config = SimpleNamespace(max_position_embeddings=8192)

        def generate(self, *, input_ids, attention_mask, generation_config):
            assert input_ids.device.type == "cpu"
            assert generation_config.use_cache is True
            assert generation_config.max_new_tokens == 256
            calls.append((input_ids.tolist(), attention_mask.tolist()))
            if input_ids.shape[0] == 4:
                tails = [[101, 9] + [9] * 254,
                         [102, 103, 10] + [9] * 253,
                         [104, 105, 106, 9] + [9] * 252,
                         [107] * 256]
            else:
                tails = [[108, 9]]
            return original_tensor([ids + tail for ids, tail in zip(input_ids.tolist(), tails)])

    monkeypatch.setenv("SLURM_JOB_ID", "cpu-stub-only")
    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(module, "load_model_config", lambda path: model_metadata)
    monkeypatch.setattr(module, "load_model", lambda config: (GenerateStub(), Tokenizer()))
    monkeypatch.setattr(transformers.GenerationConfig, "from_pretrained",
                        lambda *args, **kwargs: FakeGenerationConfig(eos_token_id=[9, 10]))
    args = Namespace(model_config=model_config,
                     benchmark_config=ROOT / "configs/benchmarks/advbench_directrequest_cached_256.yaml",
                     prompts=prompts, output_dir=tmp_path / "run", max_samples=5, sample_indices=None)
    if batch_size is not None:
        args.batch_size = batch_size
    module.run(args)
    manifest = json.loads((args.output_dir / "run-manifest.json").read_text())
    predictions = read_jsonl(args.output_dir / "responses.jsonl")
    assert manifest["status"] == "complete" and manifest["is_full_evaluation"] is False
    assert manifest["generation_config"]["batch_size"] == (batch_size or 1)
    assert manifest["outputs"]["responses_sha256"] == sha256_file(args.output_dir / "responses.jsonl")
    assert [r["sample_id"] for r in predictions] == [r["sample_id"] for r in rows]
    assert len(calls) == (2 if batch_size == 4 else 5)
    assert math.isclose(sum(r["generation_seconds"] for r in predictions),
                        manifest["runtime"]["timing"]["generation_seconds"])
    assert sum(r["generated_token_count"] for r in predictions) == manifest["runtime"]["timing"]["generated_tokens"]
    for row, prediction in zip(rows, predictions):
        ids = prompt_ids[row["prompt"]]
        assert prediction["prompt_token_count"] == len(ids)
        assert prediction["rendered_prompt_sha256"] == hashlib.sha256(str(ids).encode()).hexdigest()
        assert prediction["response_sha256"] == hashlib.sha256(prediction["raw_response"].encode()).hexdigest()
    if batch_size == 4:
        assert calls[0][1] == [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
        assert calls[1][1] == [[1, 1, 1, 1, 1]]
        assert [r["actual_batch_size"] for r in predictions] == [4, 4, 4, 4, 1]
        assert [r["generated_token_count"] for r in predictions] == [2, 3, 4, 256, 2]
        assert predictions[0]["raw_response"] == "101"  # No prompt or padding leakage.
        assert predictions[1]["last_generated_token_id"] == 10
        assert predictions[3]["termination_reason"] == "max_new_tokens"
        assert "amortized" in manifest["runtime"]["batching"]["per_example_timing"]
    else:
        assert "batching" not in manifest["runtime"]
        assert all("actual_batch_size" not in r for r in predictions)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_evaluator_rejects_nonpositive_batch_before_model_load(monkeypatch, batch_size):
    from safe_quant.evaluate_behavioral import run
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-validation-only")
    with pytest.raises(ValueError, match="positive"):
        run(Namespace(batch_size=batch_size))


def test_larger_batches_remain_scoped_to_advbench(tmp_path, monkeypatch):
    from safe_quant.evaluate_behavioral import run
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-validation-only")
    monkeypatch.setenv("SAFE_QUANT_MODEL_ROOT", str(tmp_path / "unused-model-root"))
    with pytest.raises(ValueError, match="only for AdvBench"):
        run(Namespace(batch_size=4,
                      model_config=ROOT / "configs/models/llama3_8b_instruct_spinquant_w4a4kv4.yaml",
                      benchmark_config=ROOT / "configs/benchmarks/xstest_cached_256.yaml",
                      prompts=tmp_path / "unused.jsonl"))
