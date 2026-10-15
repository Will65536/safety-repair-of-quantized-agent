"""Evaluate token perplexity on the pinned WikiText-2 test corpus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

from safe_quant.config import load_benchmark_config, load_model_config
from safe_quant.model_loaders import load_model


WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions(names: Iterable[str]) -> Dict[str, Any]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "WikiText-2 model evaluation is GPU work and must run inside a "
            "Slurm job on Delta"
        )


def _validate_protocol(config: Dict[str, Any]) -> None:
    if config.get("loader") != "wikitext2_perplexity":
        raise ValueError("WikiText-2 evaluator requires loader=wikitext2_perplexity")
    expected_dataset = {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-raw-v1",
        "revision": WIKITEXT_REVISION,
        "split": "test",
        "text_field": "text",
        "join_separator": "\n\n",
    }
    expected_protocol = {
        "version": "wikitext2_contiguous_2048_v1",
        "tokenizer_add_special_tokens": True,
        "sequence_length": 2048,
        "stride": 2048,
        "drop_final_incomplete_sequence": True,
        "predicted_tokens_per_sequence": 2047,
        "batch_size": 1,
        "use_cache": False,
        "metric": "token_perplexity",
    }
    if config.get("protocol", {}).get("version") == "wikitext2_contiguous_2048_cached_v1":
        expected_protocol.update(version="wikitext2_contiguous_2048_cached_v1", use_cache=True)
    if config.get("dataset") != expected_dataset:
        raise ValueError("Frozen WikiText-2 dataset settings changed")
    if config.get("protocol") != expected_protocol:
        raise ValueError("Frozen WikiText-2 perplexity protocol changed")


def _perplexity(total_nll: float, num_predicted_tokens: int) -> float:
    if num_predicted_tokens < 1:
        raise ValueError("num_predicted_tokens must be positive")
    return math.exp(total_nll / num_predicted_tokens)


def run(args: argparse.Namespace) -> None:
    _require_slurm_gpu()
    if args.max_blocks is not None and args.max_blocks < 1:
        raise ValueError("max_blocks must be positive")

    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    model_path = Path(args.model_config).resolve()
    benchmark_path = Path(args.benchmark_config).resolve()
    model_config = load_model_config(model_path)
    benchmark_config = load_benchmark_config(benchmark_path)
    _validate_protocol(benchmark_config)
    dataset_config = benchmark_config["dataset"]
    protocol = benchmark_config["protocol"]

    dataset = load_dataset(
        dataset_config["path"],
        dataset_config["name"],
        split=dataset_config["split"],
        revision=dataset_config["revision"],
        token=os.environ.get("HF_TOKEN") or True,
    )
    dataset_metadata = {
        **dataset_config,
        "num_rows": len(dataset),
        "fingerprint": dataset._fingerprint,
    }

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / "blocks.partial.jsonl"
    blocks_path = output_dir / "blocks.jsonl"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "run-manifest.json"
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model_config,
        "benchmark": benchmark_config,
        "dataset": dataset_metadata,
        "runtime": {
            "max_blocks": args.max_blocks,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "command": sys.argv,
            "package_versions": _package_versions(
                ["safe-quant", "torch", "transformers", "datasets"]
            ),
        },
        "config_sha256": {
            "model": _sha256(model_path),
            "benchmark": _sha256(benchmark_path),
        },
    }
    _write_json(manifest_path, manifest)

    model, tokenizer = load_model(model_config)
    manifest["model_runtime"] = getattr(model, "_safe_quant_runtime", None)
    manifest["cache_policy"] = "fresh cache per independent block; no past_key_values reused"
    _write_json(manifest_path, manifest)
    corpus = dataset_config["join_separator"].join(dataset[dataset_config["text_field"]])
    tokenized = tokenizer(
        corpus,
        return_tensors="pt",
        add_special_tokens=protocol["tokenizer_add_special_tokens"],
    )["input_ids"]
    sequence_length = int(protocol["sequence_length"])
    total_corpus_tokens = int(tokenized.numel())
    available_blocks = total_corpus_tokens // sequence_length
    evaluated_blocks = (
        available_blocks
        if args.max_blocks is None
        else min(args.max_blocks, available_blocks)
    )
    if evaluated_blocks == 0:
        raise ValueError("WikiText-2 corpus produced zero complete sequences")

    total_nll = 0.0
    total_predicted_tokens = 0
    start_time = time.perf_counter()
    loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    with partial_path.open("x", encoding="utf-8") as output:
        with torch.inference_mode():
            for block_index in tqdm(
                range(evaluated_blocks), desc=f"{model_config['id']} on WikiText-2"
            ):
                start = block_index * sequence_length
                end = start + sequence_length
                input_ids = tokenized[:, start:end].to("cuda:0", non_blocking=True)
                result = model(
                    input_ids=input_ids,
                    use_cache=protocol["use_cache"],
                    return_dict=True,
                )
                shift_logits = result.logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]
                token_losses = loss_function(
                    shift_logits.permute(0, 2, 1), shift_labels
                ).float()
                block_nll = float(token_losses.sum().item())
                predicted_tokens = int(token_losses.numel())
                mean_nll = block_nll / predicted_tokens
                record = {
                    "block_index": block_index,
                    "start_token_offset": start,
                    "end_token_offset_exclusive": end,
                    "num_input_tokens": sequence_length,
                    "num_predicted_tokens": predicted_tokens,
                    "negative_log_likelihood_sum": block_nll,
                    "mean_negative_log_likelihood": mean_nll,
                    "perplexity": math.exp(mean_nll),
                }
                output.write(json.dumps(record) + "\n")
                output.flush()
                total_nll += block_nll
                total_predicted_tokens += predicted_tokens
                del result, shift_logits, shift_labels, token_losses, input_ids

    elapsed = time.perf_counter() - start_time
    blocks_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(blocks_path)
    mean_nll = total_nll / total_predicted_tokens
    metrics = {
        "perplexity": _perplexity(total_nll, total_predicted_tokens),
        "mean_negative_log_likelihood": mean_nll,
        "negative_log_likelihood_sum": total_nll,
        "num_predicted_tokens": total_predicted_tokens,
        "num_evaluated_blocks": evaluated_blocks,
        "num_available_complete_blocks": available_blocks,
        "total_corpus_tokens": total_corpus_tokens,
        "num_dropped_tail_tokens": total_corpus_tokens - available_blocks * sequence_length,
        "is_full_evaluation": args.max_blocks is None,
        "runtime": {
            "elapsed_seconds": elapsed,
            "blocks_per_second": evaluated_blocks / elapsed if elapsed else 0.0,
        },
    }
    _write_json(metrics_path, metrics)
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["tokenization"] = {
        "add_special_tokens": protocol["tokenizer_add_special_tokens"],
        "total_corpus_tokens": total_corpus_tokens,
    }
    manifest["outputs"] = {
        "blocks": blocks_path.name,
        "metrics": metrics_path.name,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output_dir": str(output_dir), **metrics}, indent=2))


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--benchmark-config", default="configs/benchmarks/wikitext2.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-blocks", type=int)
    return parser.parse_args(argv)


def _record_failure(args: argparse.Namespace, error: Exception) -> None:
    manifest_path = Path(args.output_dir) / "run-manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = _read_json(manifest_path)
    except (OSError, json.JSONDecodeError):
        return
    if manifest.get("status") != "running":
        return
    manifest["status"] = "failed"
    manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = {"type": type(error).__name__, "message": str(error)}
    _write_json(manifest_path, manifest)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        _record_failure(args, error)
        raise


if __name__ == "__main__":
    main()
