"""Slurm-only, inference-only smoke and fresh-process reload check (not a benchmark)."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import time

from safe_quant.benchmarks.behavioral import sha256_file
from safe_quant.config import load_model_config
from safe_quant.model_loaders import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Prior capture from a DIFFERENT process")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this check only inside a Slurm GPU allocation")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse smoke output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    report = {"status": "running", "purpose": "inference smoke, not scientific benchmark",
              "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "pid": os.getpid(), "model_config_sha256": sha256_file(args.model_config)}
    started = time.perf_counter()
    try:
        import torch
        import transformers
        from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

        if not torch.cuda.is_available():
            raise RuntimeError("The Slurm allocation has no accessible CUDA GPU")
        torch.cuda.set_device(0)
        torch.manual_seed(0)
        config = load_model_config(args.model_config)
        report["model"] = config
        report["torch"] = torch.__version__
        report["transformers"] = transformers.__version__
        print(json.dumps({"stage": "load_saved_model", "model": config["id"]}), flush=True)
        model, tokenizer = load_model(config)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["runtime"] = model._safe_quant_runtime
        print(json.dumps({"stage": "saved_model_loaded", "runtime": report["runtime"]}), flush=True)
        if hasattr(model, "_flatquant_load_audit"):
            (args.output_dir / "load-audit.json").write_text(json.dumps(model._flatquant_load_audit, indent=2) + "\n")
        assert tokenizer.encode(" Yes", add_special_tokens=False) == [7566]
        assert tokenizer.encode(" No", add_special_tokens=False) == [2360]
        generation_path = Path(config.get("generation_config_path", str(Path(config["model_path"]) / "generation_config.json")))
        defaults = GenerationConfig.from_pretrained(str(generation_path.parent), local_files_only=True)
        report["generation_config_sha256"] = sha256_file(generation_path)
        eos = defaults.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        assert 128009 in eos_ids, f"Native Instruct EOT missing from EOS list: {eos_ids}"
        eos_target = 128009
        defaults.do_sample = False
        defaults.temperature = 1.0
        defaults.top_p = 1.0
        defaults.num_beams = 1
        defaults.max_new_tokens = 16
        defaults.use_cache = True
        defaults.pad_token_id = tokenizer.pad_token_id
        logits, predictions = [], []
        with torch.inference_mode():
            for prompt in ("Say hello briefly.", "What is two plus two? Answer briefly."):
                ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
                inputs = torch.tensor([ids], device="cuda:0")
                mask = torch.ones_like(inputs)
                # Cached and uncached modes are tested only if the backend can
                # actually honor them. No benchmark's use_cache is overridden.
                use_cache = config["loader"] == "flatquant_prequantized"
                output = model(input_ids=inputs, attention_mask=mask, use_cache=use_cache)
                last = output.logits[:, -1].float().cpu()
                assert torch.isfinite(last).all(), "Nonfinite loaded-model logits"
                if use_cache:
                    inner = output.past_key_values.inner
                    assert not inner.disable_quant and inner.pages.dtype == torch.uint8
                    assert inner.pages.shape[-1] == 64, "Expected packed 128-wide INT4 cache heads"
                    report["actual_cache_storage"] = str(inner.pages.dtype)
                logits.append(last)
                del output
                generated = model.generate(input_ids=inputs, attention_mask=mask,
                                           generation_config=copy.deepcopy(defaults))
                new_ids = generated[0, len(ids):].tolist()
                assert new_ids, "Generation returned no tokens"
                predictions.append({"prompt": prompt, "prompt_ids": ids, "generated_ids": new_ids,
                                    "text": tokenizer.decode(new_ids, skip_special_tokens=False),
                                    "termination": "eos" if new_ids[-1] in eos_ids else "max_new_tokens"})

            class ForceEOT(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    scores.fill_(-float("inf"))
                    scores[:, eos_target] = 0
                    return scores

            forced = model.generate(input_ids=inputs, attention_mask=mask,
                                    generation_config=copy.deepcopy(defaults),
                                    logits_processor=LogitsProcessorList([ForceEOT()]))
            assert forced[0, inputs.shape[1]:].tolist() == [eos_target], "Native EOT stopping failed"
            report["forced_eot_stopping"] = "passed (synthetic generation-control test)"
            if config["loader"] == "flatquant_prequantized":
                try:
                    model(input_ids=inputs, attention_mask=mask, use_cache=False)
                except RuntimeError as exc:
                    if "cannot honor use_cache=False" not in str(exc):
                        raise
                    report["use_cache_false"] = "correctly rejected; unmatched frozen protocols remain blocked"
                else:
                    raise AssertionError("FlatQuant silently accepted incompatible use_cache=False")
            else:
                report["use_cache_false"] = "forward smoke passed"
                report["use_cache_true"] = "greedy cached decode passed"

        captured = torch.cat(logits)
        torch.save(captured, args.output_dir / "last-logits.pt")
        with (args.output_dir / "predictions.jsonl").open("w") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        if args.reference:
            prior = json.loads((args.reference / "report.json").read_text())
            for key in ("status", "model_config_sha256", "generation_config_sha256", "runtime"):
                expected = "passed" if key == "status" else report[key]
                if prior[key] != expected:
                    raise ValueError(f"Reference provenance/status differs: {key}")
            if prior["pid"] == report["pid"] and prior["hostname"] == report["hostname"]:
                raise ValueError("Reference must come from a separate process")
            prior_logits = torch.load(args.reference / "last-logits.pt", map_location="cpu", weights_only=True)
            error = (captured - prior_logits).abs().max().item()
            report["reload_max_abs_logit_error"] = error
            # Same saved state/runtime/inputs, not a BF16-vs-rotated parity gate.
            torch.testing.assert_close(captured, prior_logits, rtol=0, atol=1e-4)
            prior_predictions = [json.loads(line) for line in (args.reference / "predictions.jsonl").read_text().splitlines()]
            assert predictions == prior_predictions, "Fresh-process greedy predictions differ"
            report["fresh_process_reload"] = "passed"
        else:
            report["fresh_process_reload"] = "capture only; run a second process with --reference"
        torch.cuda.synchronize()
        report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("status", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
