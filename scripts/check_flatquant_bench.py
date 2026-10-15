"""CPU-only dependency/config/offline-data preflight; never construct a model."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path

from safe_quant.benchmarks.behavioral import read_jsonl, sha256_file
from safe_quant.config import load_benchmark_config


GENERATION_INPUTS = (
    ("harmbench_directrequest_cached_256.yaml", "harmbench_directrequest-cached-256-v1", [0, 1]),
    ("xstest_cached_256.yaml", "xstest-cached-256-v1", [0, 25]),
    ("strongreject_cached_256.yaml", "strongreject-cached-256-v1", [0, 156]),
)


def check(group: str) -> dict:
    # This child process may run on a login node or inside a GPU job. Its checks
    # are deliberately CPU-only, including dataset/task construction.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    expected = {
        "torch": "2.7.1+cu128", "transformers": "4.45.0", "tokenizers": "0.20.3",
        "accelerate": "0.34.2", "huggingface-hub": "0.25.2", "numpy": "1.26.4",
        "safetensors": "0.4.5", "triton": "3.3.1", "flash-attn": "2.8.3.post2",
        "fast-hadamard-transform": "1.1.0", "datasets": "2.20.0",
        "pyarrow": "15.0.2", "lm-eval": "0.4.8", "peft": "0.13.2",
    }
    found = {name: version(name) for name in expected}
    if found != expected:
        raise RuntimeError(f"Unexpected FlatQuant benchmark environment: {found}")
    root = Path(__file__).resolve().parents[1]
    report = {"group": group, "versions": found, "status": "running", "model_inference": "not_run"}
    if group in {"scoring", "general", "finguard", "wildguard", "all"}:
        from datasets import config as datasets_config

        report["dataset_cache"] = str(datasets_config.HF_DATASETS_CACHE)
    configs = root / "configs/benchmarks"
    if group in {"scoring", "finguard", "all"}:
        from safe_quant.benchmarks.finguard import load_finguard
        from safe_quant.evaluate import _validate_protocol as validate_finguard

        finguard = load_benchmark_config(configs / "finguard_cached.yaml")
        validate_finguard(finguard)
        rows, _ = load_finguard(finguard, max_samples=4)
        report["finguard_available_smoke_rows"] = len(rows)
    if group in {"scoring", "wildguard", "all"}:
        from safe_quant.benchmarks.wildguard import load_wildguard_prompts
        from safe_quant.evaluate_wildguard import _validate_benchmark_config as validate_wildguard

        wildguard = load_benchmark_config(configs / "wildguard_prompt_cached.yaml")
        validate_wildguard(wildguard)
        rows, metadata = load_wildguard_prompts(wildguard)
        report["wildguard_filtered_rows"] = len(rows)
        report["wildguard_fingerprint"] = metadata["filtered_fingerprint"]
    if group in {"scoring", "all"}:
        from datasets import load_dataset
        from safe_quant.evaluate_wikitext2 import _validate_protocol as validate_wikitext

        wiki = load_benchmark_config(configs / "wikitext2_cached.yaml")
        validate_wikitext(wiki)
        data = wiki["dataset"]
        corpus = load_dataset(data["path"], data["name"], split=data["split"],
                              revision=data["revision"], token=os.environ.get("HF_TOKEN") or True)
        report["wikitext_rows"] = len(corpus)
        report["wikitext_fingerprint"] = corpus._fingerprint
    if group in {"scoring", "general", "all"}:
        from lm_eval.tasks import TaskManager, get_task_dict
        from safe_quant.evaluate_general import TASKS, _validate_protocol as validate_general

        validate_general(load_benchmark_config(configs / "general_4task_cached_bs1.yaml"))
        tasks = get_task_dict(list(TASKS), task_manager=TaskManager())
        report["general_task_rows"] = {
            name: len(task.eval_docs) for name, task in tasks.items()
        }
    if group in {"generation", "all"}:
        from safe_quant.evaluate_behavioral import validate_protocol

        inputs = []
        for filename, directory, indices in GENERATION_INPUTS:
            config = load_benchmark_config(root / "configs/benchmarks" / filename)
            validate_protocol(config)
            prompts = root / "outputs/behavioral/prompts" / directory / "prompts.jsonl"
            manifest = json.loads((prompts.parent / "prompt-manifest.json").read_text())
            if manifest["benchmark"] != config or sha256_file(prompts) != manifest["prompts_sha256"]:
                raise ValueError(f"Prepared prompts differ from the pinned benchmark: {prompts}")
            rows = read_jsonl(prompts)
            if len(rows) != manifest["num_samples"] or max(indices) >= len(rows):
                raise ValueError(f"Invalid prepared prompt selection: {prompts}")
            if config["id"] == "xstest" and {rows[i]["metadata"]["subset"] for i in indices} != {"safe", "unsafe_contrast"}:
                raise ValueError("XSTest smoke must include both safe and unsafe-contrast subsets")
            inputs.append({"benchmark": config["id"], "num_available": len(rows),
                           "sample_indices": indices, "prompts_sha256": manifest["prompts_sha256"]})
        report["generation_inputs"] = inputs
    report["status"] = "passed"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("scoring", "general", "finguard", "wildguard", "generation", "all"), default="all")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = check(args.group)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        if args.output_json.exists():
            raise FileExistsError(args.output_json)
        args.output_json.write_text(rendered)
    print(rendered, end="", flush=True)


if __name__ == "__main__":
    main()
