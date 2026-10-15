"""Load an existing real-quantized FlatQuant checkpoint."""

from __future__ import annotations

import json
import functools
import importlib.util
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Tuple

from safe_quant.benchmarks.behavioral import sha256_file


def _guard_cache_contract(model: Any) -> None:
    """Refuse requests the published runtime would silently override."""
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        if kwargs.get("use_cache", model.config.use_cache) is False:
            raise RuntimeError(
                "FlatQuant's published runtime forces its INT4 cache and cannot honor "
                "use_cache=False. Do not silently change the frozen benchmark protocol."
            )
        return original(*args, **kwargs)

    model.forward = forward


def _load_local_runtime(config: Dict[str, Any], model_path: Path) -> Any:
    """Use the exact checkpoint's deployment code and locally rebuilt extension."""
    settings = config["flatquant"]
    manifest = json.loads(Path(settings["checkpoint_manifest"]).read_text())
    if (manifest["revision"] != settings["checkpoint_revision"]
            or Path(manifest["directory"]).resolve() != model_path.resolve()):
        raise ValueError("FlatQuant snapshot manifest does not match the pinned artifact")
    helper_path = Path(settings["runtime_support"])
    runtime = helper_path.parent / "runtime"
    for name, expected in manifest["files"].items():
        path = model_path / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"FlatQuant immutable checkpoint changed: {path}")
        if name.startswith("deploy/") and name.endswith(".py"):
            if sha256_file(runtime / name) != expected["sha256"]:
                raise ValueError(f"FlatQuant deployment Python differs from the pinned snapshot: {name}")
    spec = importlib.util.spec_from_file_location("safe_quant_flatquant_runtime", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if helper.MODEL.resolve() != model_path.resolve() or helper.RUNTIME.resolve() != runtime.resolve():
        raise ValueError("FlatQuant helper points to a different artifact/runtime")
    with tempfile.TemporaryDirectory(prefix="safe-quant-flatquant-load-") as temporary:
        audit_path = Path(temporary) / "load.json"
        model = helper.load_model(audit_path)
        model._flatquant_load_audit = json.loads(audit_path.read_text())
    import torch
    import deploy
    linears = [m for m in model.modules() if isinstance(m, deploy.nn.Linear4bit)]
    if len(linears) != 224 or any(
            m.weight.dtype != torch.uint8 or m.weight.shape[-1] != m.in_features // 2
            for m in linears):
        raise ValueError("FlatQuant expected 224 packed INT4 linear layers")
    model.config.use_cache = True  # Published behavior, not an honored False request.
    _guard_cache_contract(model)
    model._safe_quant_runtime = {
        "method": "FlatQuant W4A4KV4", "checkpoint_revision": manifest["revision"],
        "packed_linear_count": len(linears),
        "checkpoint_manifest_sha256": sha256_file(Path(settings["checkpoint_manifest"])),
        "runtime_support_sha256": sha256_file(helper_path),
        "runtime_extension_sha256": sha256_file(Path(deploy._CUDA.__file__)),
        "underlying_precision": "FP16 published runtime (not BF16)",
        "attention_implementation": "flash_attention_2 published runtime",
        "cache_storage": "packed INT4 / torch.uint8",
        "runtime_forces_custom_kv_cache": True,
        "use_cache_false_policy": "reject, never silently override",
        "calibration_or_offline_rotation_run": False,
    }
    return model


def _validate_artifact(config: Dict[str, Any]) -> Path:
    if config.get("artifact_format") != "flatquant_real_safetensors":
        raise ValueError(
            "FlatQuant loading requires artifact_format=flatquant_real_safetensors"
        )

    precision = config.get("precision", {})
    actual = (
        int(precision.get("weights", -1)),
        int(precision.get("activations", -1)),
        int(precision.get("kv_cache", -1)),
    )
    if actual != (4, 4, 4):
        raise ValueError(f"FlatQuant config must describe W4A4KV4, got {actual}")

    if not bool(config.get("local_files_only", True)):
        raise ValueError("FlatQuant evaluation requires a local immutable checkpoint")
    if not bool(config.get("trust_remote_code", False)):
        raise ValueError("FlatQuant's published checkpoint requires trust_remote_code=true")

    model_path = Path(config["model_path"])
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"FlatQuant config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        artifact_config = json.load(handle)
    quantization = artifact_config.get("quantization_config", {})
    expected = {
        "quant_method": "flatquant",
        "w_bits": 4,
        "a_bits": 4,
        "kv_bits": 4,
        "w_is_symmetric": True,
        "a_is_symmetric": True,
        "kv_is_symmetric": False,
    }
    mismatches = {
        key: {"expected": wanted, "found": quantization.get(key)}
        for key, wanted in expected.items()
        if quantization.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Unexpected FlatQuant artifact metadata: {mismatches}")
    return model_path


def load_flatquant_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    """Load only a published FlatQuant artifact; never calibrate or quantize."""

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("FlatQuant model loading requires a Slurm GPU allocation")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = _validate_artifact(config)
    local_runtime = bool(config.get("flatquant", {}).get("runtime_support"))
    if local_runtime:
        import transformers
        if torch.__version__.split("+")[0] != "2.7.1" or transformers.__version__ != "4.45.0":
            raise RuntimeError("Use flatquant-eval (torch 2.7.1, transformers 4.45.0)")
        if (config.get("torch_dtype") != "float16"
                or config.get("attention_implementation") != "flash_attention_2"):
            raise ValueError("Published FlatQuant requires explicit FP16 / flash_attention_2 runtime metadata")
    tokenizer_path = Path(config["tokenizer_path"])
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"FlatQuant tokenizer not found: {tokenizer_path}")

    token = os.environ.get("HF_TOKEN")
    common = {
        "revision": config.get("revision"),
        "token": token,
        "local_files_only": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **common)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if local_runtime:
        model = _load_local_runtime(config, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **common,
        )
    model = model.to("cuda:0")
    model.eval()

    # Preserve the legacy Llama-3.1 exploratory adapter. The exact Llama-3
    # adapter instead exposes the true cache behavior and rejects False calls.
    if not local_runtime:
        model.config.use_cache = False  # Preserve the existing Llama-3.1 exploratory adapter.
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
