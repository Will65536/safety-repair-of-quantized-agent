"""Run native fake_quant/main.py with recorded paper-style W4A4KV4 settings.

No custom parity gates, quantization implementation, or normalization patch.
All GPU/model work is guarded by Slurm. Outputs never overwrite existing runs.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/bf16")
ARTIFACTS = BASE.parent / "quarot-w4a4kv4"


def verify_native_source():
    # Permit only the two reviewed device-placement additions in GPTQ.
    # Removing them must recover the upstream file byte-for-byte.
    device_patches = (
        "    # Shared RoPE lives outside decoder layers in Transformers 4.44.\n"
        "    if hasattr(model.model, 'rotary_emb'):\n"
        "        model.model.rotary_emb = model.model.rotary_emb.to(dev)\n",
        "    if hasattr(model.model, 'rotary_emb'):\n"
        "        model.model.rotary_emb = model.model.rotary_emb.cpu()\n",
    )
    for name in ("model_utils.py", "rotation_utils.py", "quant_utils.py", "gptq_utils.py",
                 "main.py", "data_utils.py", "hadamard_utils.py", "monkeypatch.py"):
        expected = subprocess.check_output(["git", "show", f"HEAD:fake_quant/{name}"], cwd=ROOT)
        actual = (ROOT / "fake_quant" / name).read_bytes()
        if name == "gptq_utils.py":
            for patch in device_patches:
                encoded = patch.encode()
                assert actual.count(encoded) == 1, "Missing/altered approved GPTQ RoPE device patch"
                actual = actual.replace(encoded, b"", 1)
        assert actual == expected, f"Unexpected numerical source change: {name}"


def native_arguments(output):
    return ["--model", str(BASE), "--seed", "0", "--rotate", "--rotate_mode", "hadamard",
            "--w_bits", "4", "--a_bits", "4", "--k_bits", "4", "--v_bits", "4", "--w_clip",
            "--w_groupsize", "-1", "--a_groupsize", "-1", "--k_groupsize", "128", "--v_groupsize", "128",
            "--no-w_asym", "--no-a_asym", "--k_asym", "--v_asym",
            "--no-w_rtn", "--no-act_order", "--no-int8_down_proj", "--fp32_had",
            "--a_clip_ratio", "0.9", "--k_clip_ratio", "0.95", "--v_clip_ratio", "0.95",
            "--cal_dataset", "wikitext2", "--nsamples", "128", "--percdamp", "0.01",
            "--eval_dataset", "wikitext2", "--bsz", "1",
            "--save_name", output.name, "--save_qmodel_path", str(output / "qmodel.pt")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show-command", action="store_true", help="Print native recipe without loading anything")
    args = parser.parse_args()
    if args.show_command:
        import shlex
        print(shlex.join(["python", "-u", "-B", "fake_quant/main.py", *native_arguments(ARTIFACTS / "paperstyle-RUN_ID")]))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Use a user-requested Slurm GPU allocation; never run quantization on a login node")
    os.environ["HF_HOME"] = str(ROOT / "local_setup/hf-cache")
    os.environ["HF_DATASETS_CACHE"] = str(ROOT / "local_setup/hf-cache/datasets")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    # Verify the cached native data stream before expensive GPU work.
    subprocess.run([sys.executable, "-B", str(ROOT / "local_setup/prepare_wikitext.py"), "--offline-check"], check=True)
    sys.path.insert(0, str(ROOT / "fake_quant"))
    import torch
    import transformers
    import main as native
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU visible in the Slurm allocation")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires BF16 support")
    assert Path(native.__file__).resolve() == ROOT / "fake_quant/main.py"
    verify_native_source()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ARTIFACTS / f"paperstyle-{os.environ['SLURM_JOB_ID']}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    arguments = native_arguments(output)
    metadata = {"status": "running", "method": "QuaRot W4A4KV4", "base_model": str(BASE),
        "recipe": "paper-style",
        "paper_reference": "https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf",
        "reproduction_scope": "paper-style quantization settings on BF16 Llama-3-8B-Instruct, not exact published model/precision",
        "underlying_dtype": "BF16 from base config; original upstream normalization",
        "gptq_compatibility_patch": "move shared model-level RoPE to capture device and back; no dtype/math change",
        "artifact_type": "native fake-quantized state, not packed INT4 or standalone Hugging Face",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "transformers": transformers.__version__,
        "native_arguments": arguments, "native_entrypoint": str(ROOT / "fake_quant/main.py"),
        "w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4,
        "w_symmetric": True, "a_symmetric": True, "k_symmetric": False, "v_symmetric": False,
        "w_groupsize": -1, "a_groupsize": -1, "k_groupsize": 128, "v_groupsize": 128,
        "a_clip_ratio": 0.9, "k_clip_ratio": 0.95, "v_clip_ratio": 0.95,
        "seed": 0, "nsamples": 128, "seqlen": 2048, "percdamp": 0.01,
        "w_clip": True, "gptq": True, "fp32_had": True, "int8_down_proj": False,
        "eval_batch_size": 1, "eval_dataset": "wikitext2/test",
        "upstream_weight_save_precedes_activation_and_kv_configuration": True,
        "kv_semantics": "native fake quantization; low-bit numerical values may be stored as BF16 tensors, not packed cache",
        "load_only_adapter_status": "pending; preserve arguments to reconstruct runtime",
        "checkpoint_usable_for_safe_quant": False, "custom_parity_gates": "not_run",
        "prior_parity_failure": "recorded separately; user chose native workflow without gating on it"}
    for name in ("requirements.lock.txt",):
        shutil.copyfile(ROOT / "local_setup" / name, output / name)
    shutil.copyfile(ROOT / "local_setup/data/wikitext2_manifest.json", output / "calibration_manifest.json")
    for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json"):
        if (BASE / name).is_file():
            shutil.copyfile(BASE / name, output / name)
    (output / "source_changes.patch").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    metadata["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted((ROOT / "fake_quant").glob("*.py"))}
    started = time.monotonic()

    def save():
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        metadata["peak_host_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    print(json.dumps({"native_quarot_run": "starting", "artifact_directory": str(output),
                      "native_arguments": arguments}), flush=True)
    try:
        sys.argv = [str(ROOT / "fake_quant/main.py"), *arguments]
        native.main()
        checkpoint = output / "qmodel.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError("Native workflow returned without a saved checkpoint")
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        metadata["checkpoint_bytes"] = checkpoint.stat().st_size
        metadata["checkpoint_sha256"] = digest.hexdigest()
        metrics = json.loads((output / "ppl_metrics.json").read_text())
        import math
        if not math.isfinite(metrics["perplexity"]) or metrics["perplexity"] <= 0:
            raise RuntimeError("Native evaluation produced invalid perplexity")
        metadata["native_perplexity"] = metrics["perplexity"]
        metadata["status"] = "native_quantization_and_ppl_completed"
        save()
        print(json.dumps(metadata, indent=2), flush=True)
    except Exception:
        metadata["status"] = "failed"
        metadata["checkpoint_exists"] = (output / "qmodel.pt").is_file()
        metadata["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
