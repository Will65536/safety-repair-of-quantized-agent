"""Generate and permanently cache responses to prepared behavioral prompts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from safe_quant.benchmarks.behavioral import read_jsonl, sha256_file
from safe_quant.config import load_benchmark_config, load_model_config
from safe_quant.model_loaders import load_model


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Behavioral model inference must run inside a Slurm GPU allocation")


def _versions() -> Dict[str, Any]:
    versions = {}
    for name in ("safe-quant", "torch", "transformers", "datasets", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def validate_protocol(config: Dict[str, Any]) -> Dict[str, Any]:
    protocol = config["protocol"]
    common = {
        "do_sample": False,
        "messages": "user_only_native_chat_template",
    }
    supported = {
        "behavioral_greedy_v1": {
            **common, "max_new_tokens": 512, "use_cache": False,
        },
        "behavioral_greedy_cached_v1": {
            **common, "max_new_tokens": 512, "use_cache": True,
        },
        "behavioral_greedy_cached_256_v1": {
            **common, "max_new_tokens": 256, "use_cache": True,
        },
        "behavioral_greedy_cached_128_smoke_v1": {
            **common, "max_new_tokens": 128, "use_cache": True,
        },
    }
    version = protocol.get("version")
    expected = {"version": version, **supported.get(version, {})}
    if (
        config["loader"] != "behavioral_generation"
        or version not in supported
        or protocol != expected
    ):
        raise ValueError("Behavioral protocol is not a supported frozen protocol")
    return protocol


def termination_metadata(new_ids: list[int], eos_token_id: Any, limit: int) -> Dict[str, Any]:
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    ended_with_eos = bool(new_ids) and new_ids[-1] in eos_ids
    if ended_with_eos:
        reason = "eos_token"
    elif len(new_ids) >= limit:
        reason = "max_new_tokens"
    else:
        reason = "generation_stopped_other"
    return {
        "termination_reason": reason,
        "ended_with_eos": ended_with_eos,
        "last_generated_token_id": new_ids[-1] if new_ids else None,
    }


def select_rows(
    rows: list[Dict[str, Any]],
    max_samples: Optional[int],
    sample_indices: Optional[List[int]],
) -> list[Dict[str, Any]]:
    if max_samples is not None and sample_indices is not None:
        raise ValueError("Use only one of --max-samples or --sample-indices")
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive")
        return rows[:max_samples]
    if sample_indices is None:
        return rows
    if not sample_indices or len(set(sample_indices)) != len(sample_indices):
        raise ValueError("sample_indices must be non-empty and unique")
    if min(sample_indices) < 0 or max(sample_indices) >= len(rows):
        raise ValueError("sample_indices are outside the prepared prompt range")
    return [rows[index] for index in sample_indices]


def render_ids(tokenizer: Any, prompt: str) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if not isinstance(ids, list) or not ids:
        raise ValueError("Native chat template did not return token IDs")
    return ids


def left_pad_generation_inputs(
    sequences: List[List[int]], pad_token_id: Optional[int]
) -> tuple[List[List[int]], List[List[int]]]:
    """Pad already-rendered IDs without re-tokenizing or changing prompt content."""
    if not sequences or any(not ids for ids in sequences):
        raise ValueError("Generation needs nonempty prompt token sequences")
    width = max(len(ids) for ids in sequences)
    if pad_token_id is None and any(len(ids) < width for ids in sequences):
        raise ValueError("Mixed-length batched generation requires a pad token")
    inputs = [[pad_token_id] * (width - len(ids)) + ids for ids in sequences]
    masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in sequences]
    return inputs, masks


def trim_after_first_eos(new_ids: List[int], eos_token_id: Any) -> List[int]:
    """Exclude padding added while other batch members continue; retain EOS itself."""
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])
    for index, token in enumerate(new_ids):
        if token in eos_ids:
            return new_ids[:index + 1]
    return new_ids


def run(args: argparse.Namespace) -> None:
    _require_slurm_gpu()
    batch_size = getattr(args, "batch_size", 1)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import torch
    from transformers import GenerationConfig

    model_path = args.model_config.resolve()
    benchmark_path = args.benchmark_config.resolve()
    prompts_path = args.prompts.resolve()
    model_config = load_model_config(model_path)
    benchmark_config = load_benchmark_config(benchmark_path)
    protocol = validate_protocol(benchmark_config)
    if batch_size != 1 and benchmark_config["id"] != "advbench_directrequest":
        raise ValueError("Batched behavioral generation is currently enabled only for AdvBench DirectRequest")
    prompt_manifest_path = prompts_path.parent / "prompt-manifest.json"
    prompt_manifest = json.loads(prompt_manifest_path.read_text(encoding="utf-8"))
    if prompt_manifest["benchmark"] != benchmark_config:
        raise ValueError("Prepared prompts used a different benchmark configuration")
    if sha256_file(prompts_path) != prompt_manifest["prompts_sha256"]:
        raise ValueError("Prepared prompt file changed")
    rows = select_rows(read_jsonl(prompts_path), args.max_samples, args.sample_indices)
    if not rows:
        raise ValueError("No prepared prompts selected")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial = args.output_dir / "responses.partial.jsonl"
    final = args.output_dir / "responses.jsonl"
    manifest_path = args.output_dir / "run-manifest.json"
    manifest: Dict[str, Any] = {
        "status": "running",
        "started_at": _now(),
        "model": model_config,
        "benchmark": benchmark_config,
        "prepared_prompts": prompt_manifest,
        "prompts_sha256": prompt_manifest["prompts_sha256"],
        "selected_num_samples": len(rows),
        "is_full_evaluation": args.max_samples is None and args.sample_indices is None,
        "selection": {
            "max_samples": args.max_samples,
            "sample_indices": args.sample_indices,
        },
        "runtime": {"slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "command": sys.argv, "package_versions": _versions()},
        "config_sha256": {"model": sha256_file(model_path), "benchmark": sha256_file(benchmark_path)},
    }
    _write_json(manifest_path, manifest)

    try:
        load_started = time.perf_counter()
        model, tokenizer = load_model(model_config)
        manifest["model_runtime"] = getattr(model, "_safe_quant_runtime", None)
        torch.cuda.synchronize()
        model_load_seconds = time.perf_counter() - load_started
        manifest["runtime"]["timing"] = {"model_load_seconds": model_load_seconds}
        base_generation_path = Path(model_config.get(
            "generation_config_path",
            str(Path(model_config["model_path"]) / "generation_config.json"),
        ))
        if not base_generation_path.is_file():
            raise FileNotFoundError(f"Native generation configuration not found: {base_generation_path}")
        base_generation_defaults = json.loads(base_generation_path.read_text(encoding="utf-8"))
        shared_generation_config = GenerationConfig.from_pretrained(
            str(base_generation_path.parent), local_files_only=True
        )
        shared_generation_config.do_sample = protocol["do_sample"]
        shared_generation_config.max_new_tokens = protocol["max_new_tokens"]
        shared_generation_config.num_beams = 1
        shared_generation_config.use_cache = protocol["use_cache"]
        shared_generation_config.pad_token_id = tokenizer.pad_token_id
        shared_generation_config.temperature = 1.0
        shared_generation_config.top_p = 1.0
        generation = {
            "do_sample": protocol["do_sample"],
            "max_new_tokens": protocol["max_new_tokens"],
            "num_beams": 1,
            "use_cache": protocol["use_cache"],
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": shared_generation_config.eos_token_id,
            "chat_template": "native",
            "messages": "user_only",
            "add_generation_prompt": True,
            "batch_size": batch_size,
            "native_generation_config_sha256": sha256_file(base_generation_path),
            "effective_generation_config": shared_generation_config.to_dict(),
        }
        context = int(model.config.max_position_embeddings)
        encoded_rows = []
        too_long = []
        for row in rows:
            ids = render_ids(tokenizer, row["prompt"])
            if len(ids) + protocol["max_new_tokens"] > context:
                too_long.append({"sample_id": row["sample_id"], "prompt_tokens": len(ids), "context": context})
            encoded_rows.append((row, ids))
        if too_long:
            _write_json(args.output_dir / "truncation-report.json", {"rejected_prompts": too_long})
            raise ValueError(f"{len(too_long)} prompts exceed context; see truncation-report.json")
        manifest["generation_config"] = generation
        manifest["native_generation_defaults"] = base_generation_defaults
        manifest["context_window"] = context
        if batch_size > 1:
            manifest["runtime"]["batching"] = {
                "padding_side": "left",
                "positions_and_cache": "native model.generate; fresh cache per batch",
                "post_eos_padding": "excluded from responses and token counts",
                "per_example_timing": "amortized batch wall time / actual batch size; not individual latency",
            }
        _write_json(manifest_path, manifest)
        total_generation_seconds = 0.0
        total_generated_tokens = 0
        with partial.open("x", encoding="utf-8") as handle, torch.inference_mode():
            for start in range(0, len(encoded_rows), batch_size):
                batch = encoded_rows[start:start + batch_size]
                padded_ids, masks = left_pad_generation_inputs(
                    [ids for _, ids in batch], tokenizer.pad_token_id
                )
                input_ids = torch.tensor(padded_ids, dtype=torch.long, device="cuda:0")
                attention_mask = torch.tensor(masks, dtype=torch.long, device="cuda:0")
                input_width = len(padded_ids[0])
                torch.cuda.synchronize()
                generation_started = time.perf_counter()
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=shared_generation_config,
                )
                output_rows = output_ids.tolist()
                torch.cuda.synchronize()
                batch_seconds = time.perf_counter() - generation_started
                if output_ids.shape[0] != len(batch) or output_ids.shape[1] < input_width:
                    raise ValueError("Native generation returned unexpected batch/sequence dimensions")
                generation_seconds = batch_seconds / len(batch)
                total_generation_seconds += batch_seconds
                for (row, ids), output in zip(batch, output_rows):
                    # Generated tokens begin after the PADDED input width, not
                    # after this row's shorter, original prompt length.
                    new_ids = output[input_width:]
                    if batch_size > 1:
                        new_ids = trim_after_first_eos(new_ids, shared_generation_config.eos_token_id)
                    termination = termination_metadata(
                        new_ids, shared_generation_config.eos_token_id, protocol["max_new_tokens"],
                    )
                    total_generated_tokens += len(new_ids)
                    raw = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    record = {
                        "benchmark_id": benchmark_config["id"],
                        "benchmark_version": protocol["version"],
                        "sample_id": row["sample_id"],
                        "source_row": row["source_row"],
                        "prompt": row["prompt"],
                        "metadata": row["metadata"],
                        "model_id": model_config["id"],
                        "model_config": model_config,
                        "generation_config": generation,
                        "rendered_prompt_sha256": hashlib.sha256(bytes(str(ids), "utf-8")).hexdigest(),
                        "prompt_token_count": len(ids),
                        "raw_response": raw,
                        "generated_token_count": len(new_ids),
                        **termination,
                        "generation_seconds": generation_seconds,
                        "generated_tokens_per_second": len(new_ids) / generation_seconds,
                        "response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "run_id": args.output_dir.name,
                        "generated_at": _now(),
                    }
                    if batch_size > 1:
                        record.update({
                            "batch_index": start // batch_size,
                            "actual_batch_size": len(batch),
                            "padded_prompt_token_count": input_width,
                            "batch_generation_seconds": batch_seconds,
                            "generation_seconds_interpretation": "amortized batch wall time, not individual latency",
                        })
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                del output_ids, input_ids, attention_mask
        partial.replace(final)
        manifest["status"] = "complete"
        manifest["completed_at"] = _now()
        manifest["runtime"]["timing"].update({
            "generation_seconds": total_generation_seconds,
            "generated_tokens": total_generated_tokens,
            "generated_tokens_per_second": (
                total_generated_tokens / total_generation_seconds
            ),
        })
        manifest["outputs"] = {"responses": final.name, "responses_sha256": sha256_file(final)}
        _write_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failed_at"] = _now()
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Generation batch size (default 1); larger batches currently supported only for AdvBench")
    parser.add_argument(
        "--sample-indices",
        type=lambda value: [int(item) for item in value.split(",")],
        help="Comma-separated zero-based prepared-prompt row indices",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
