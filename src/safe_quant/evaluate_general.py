"""Run the frozen four-task general evaluation with lm-eval-harness."""

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
from typing import Any, Dict, Iterable

from safe_quant.config import load_benchmark_config, load_model_config
from safe_quant.model_loaders import load_model


TASKS = ("arc_easy", "arc_challenge", "piqa", "winogrande")
HARNESS_VERSION = "0.4.8"


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _task_definition_hashes() -> Dict[str, str]:
    """Hash the lm-eval files that define the four frozen tasks."""
    import lm_eval

    package_root = Path(lm_eval.__file__).resolve().parent
    relative_paths = (
        "tasks/arc/arc_easy.yaml",
        "tasks/arc/arc_challenge.yaml",
        "tasks/piqa/piqa.yaml",
        "tasks/winogrande/default.yaml",
        "tasks/winogrande/preprocess_winogrande.py",
    )
    hashes = {}
    for relative_path in relative_paths:
        path = package_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"lm-eval task definition not found: {path}")
        hashes[relative_path] = _sha256(path)
    return hashes


def _require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "General benchmark model evaluation is GPU work and must run "
            "inside a Slurm job on Delta"
        )


def _validate_protocol(config: Dict[str, Any]) -> None:
    expected = {
        "version": "general_4task_lm_eval_v1",
        "harness_version": HARNESS_VERSION,
        "tasks": list(TASKS),
        "num_fewshot": 0,
        "apply_chat_template": False,
        "fewshot_as_multiturn": False,
        "max_length": 2048,
        "add_bos_token": False,
        "logits_cache": True,
        "batch_size": 8,
        "bootstrap_iters": 0,
        "random_seed": 0,
        "numpy_random_seed": 1234,
        "torch_random_seed": 1234,
        "fewshot_random_seed": 1234,
        "confirm_run_unsafe_code": True,
        "primary_metric": "acc",
        "aggregate": "unweighted_mean_of_four_task_accuracies",
    }
    if config.get("protocol", {}).get("version") == "general_4task_lm_eval_cached_bs1_v1":
        expected.update(version="general_4task_lm_eval_cached_bs1_v1", batch_size=1, use_cache=True)
    if config.get("loader") != "lm_eval_harness":
        raise ValueError("General evaluator requires loader=lm_eval_harness")
    if config.get("dataset") != {"implementation": "lm-evaluation-harness"}:
        raise ValueError("Frozen general benchmark implementation changed")
    if config.get("protocol") != expected:
        raise ValueError("Frozen general four-task protocol changed")


def _extract_summary(
    harness_results: Dict[str, Any], limit: Any
) -> Dict[str, Any]:
    tasks = {}
    for task in TASKS:
        result = harness_results["results"].get(task)
        if result is None or "acc,none" not in result:
            raise KeyError(f"lm-eval did not return acc,none for {task}")
        task_summary = {
            "accuracy": float(result["acc,none"]),
            "num_samples": harness_results["n-samples"][task],
        }
        if "acc_norm,none" in result:
            task_summary["accuracy_normalized_diagnostic"] = float(
                result["acc_norm,none"]
            )
        tasks[task] = task_summary
    general_average = sum(value["accuracy"] for value in tasks.values()) / len(tasks)
    return {
        "tasks": tasks,
        "general_4task_avg": general_average,
        "average_definition": "unweighted arithmetic mean of four acc values",
        "is_full_evaluation": limit is None,
        "limit_per_task": limit,
    }


def run(args: argparse.Namespace) -> None:
    _require_slurm_gpu()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    installed_harness = importlib.metadata.version("lm-eval")
    if installed_harness != HARNESS_VERSION:
        raise RuntimeError(
            f"lm-eval must be exactly {HARNESS_VERSION}; found {installed_harness}"
        )

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    model_path = Path(args.model_config).resolve()
    benchmark_path = Path(args.benchmark_config).resolve()
    model_config = load_model_config(model_path)
    benchmark_config = load_benchmark_config(benchmark_path)
    _validate_protocol(benchmark_config)
    protocol = benchmark_config["protocol"]

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    harness_path = output_dir / "harness-results.json"
    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "run-manifest.json"
    manifest = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model": model_config,
        "benchmark": benchmark_config,
        "runtime": {
            "limit_per_task": args.limit,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "command": sys.argv,
            "package_versions": _package_versions(
                [
                    "safe-quant",
                    "lm-eval",
                    "torch",
                    "transformers",
                    "datasets",
                    "accelerate",
                    "peft",
                ]
            ),
        },
        "config_sha256": {
            "model": _sha256(model_path),
            "benchmark": _sha256(benchmark_path),
        },
    }
    _write_json(manifest_path, manifest)

    model, tokenizer = load_model(model_config)
    if "use_cache" in protocol:
        # Pinned harness likelihood calls inherit model.config.use_cache.
        model.config.use_cache = protocol["use_cache"]
    manifest["model_runtime"] = getattr(model, "_safe_quant_runtime", None)
    manifest["runtime"]["effective_use_cache"] = model.config.use_cache
    manifest["runtime"]["cache_policy"] = "fresh cache per independent likelihood forward"
    _write_json(manifest_path, manifest)
    harness_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        backend="causal",
        batch_size=int(protocol["batch_size"]),
        max_length=int(protocol["max_length"]),
        add_bos_token=bool(protocol["add_bos_token"]),
        logits_cache=bool(protocol["logits_cache"]),
        truncation=False,
        parallelize=False,
    )
    start_time = time.perf_counter()
    harness_results = simple_evaluate(
        model=harness_model,
        tasks=list(protocol["tasks"]),
        num_fewshot=int(protocol["num_fewshot"]),
        batch_size=int(protocol["batch_size"]),
        device="cuda:0",
        limit=args.limit,
        bootstrap_iters=int(protocol["bootstrap_iters"]),
        log_samples=True,
        apply_chat_template=bool(protocol["apply_chat_template"]),
        fewshot_as_multiturn=bool(protocol["fewshot_as_multiturn"]),
        random_seed=int(protocol["random_seed"]),
        numpy_random_seed=int(protocol["numpy_random_seed"]),
        torch_random_seed=int(protocol["torch_random_seed"]),
        fewshot_random_seed=int(protocol["fewshot_random_seed"]),
        confirm_run_unsafe_code=bool(protocol["confirm_run_unsafe_code"]),
    )
    elapsed = time.perf_counter() - start_time
    if harness_results is None:
        raise RuntimeError("lm-eval returned no results")
    samples = harness_results.pop("samples")
    with samples_path.open("x", encoding="utf-8") as output:
        for task in TASKS:
            for sample in samples[task]:
                output.write(
                    json.dumps(
                        {"task": task, **sample},
                        ensure_ascii=False,
                        default=_json_default,
                    )
                    + "\n"
                )
    summary = _extract_summary(harness_results, args.limit)
    summary["runtime"] = {"elapsed_seconds": elapsed}
    _write_json(harness_path, harness_results)
    _write_json(metrics_path, summary)

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["harness_metadata"] = {
        "git_hash": harness_results.get("git_hash"),
        "task_versions": harness_results.get("versions"),
        "num_fewshot": harness_results.get("n-shot"),
        "task_configs": harness_results.get("configs"),
        "task_definition_sha256": _task_definition_hashes(),
    }
    manifest["outputs"] = {
        "samples": samples_path.name,
        "harness_results": harness_path.name,
        "metrics": metrics_path.name,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


def parse_args(argv: Any = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--benchmark-config", default="configs/benchmarks/general_4task.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
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


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception as error:
        _record_failure(args, error)
        raise


if __name__ == "__main__":
    main()
