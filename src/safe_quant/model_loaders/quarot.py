"""Inference-only reconstruction of the saved native QuaRot paper-style state.

Weights already contain norm fusion, offline rotations and GPTQ. Never repeat
those operations here. Online Hadamards and activation/K/V quantizers remain
part of inference, exactly as in the producing native workflow.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_artifact(config: Dict[str, Any]) -> Tuple[Path, Dict[str, Any]]:
    if config.get("artifact_format") != "quarot_qmodel_state":
        raise ValueError("QuaRot requires artifact_format=quarot_qmodel_state")
    precision = config["precision"]
    if tuple(precision.get(k) for k in ("weights", "activations", "kv_cache")) != (4, 4, 4):
        raise ValueError("QuaRot config must describe W4A4KV4")
    if not config.get("local_files_only", True):
        raise ValueError("QuaRot requires a local immutable checkpoint")
    if config.get("torch_dtype", "bfloat16") != "bfloat16":
        raise ValueError("The saved QuaRot condition uses BF16; do not relabel its runtime precision")
    settings = config["quarot"]
    directory = Path(settings["artifact_path"])
    manifest = json.loads((directory / "run_manifest.json").read_text())
    expected = {
        "status": "native_quantization_and_ppl_completed",
        "method": "QuaRot W4A4KV4", "recipe": "paper-style",
        "w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4,
        "w_symmetric": True, "a_symmetric": True,
        "k_symmetric": False, "v_symmetric": False,
        "w_groupsize": -1, "a_groupsize": -1,
        "k_groupsize": 128, "v_groupsize": 128,
        "a_clip_ratio": 0.9, "k_clip_ratio": 0.95, "v_clip_ratio": 0.95,
        "fp32_had": True, "int8_down_proj": False,
        "upstream_weight_save_precedes_activation_and_kv_configuration": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Unsupported QuaRot manifest {key}: {manifest.get(key)!r}")
    if Path(manifest["base_model"]).resolve() != Path(config["model_path"]).resolve():
        raise ValueError("QuaRot base model does not match the producing manifest")
    checkpoint = directory / "qmodel.pt"
    if checkpoint.stat().st_size != manifest["checkpoint_bytes"]:
        raise ValueError("QuaRot checkpoint size differs from completed artifact")
    if manifest["checkpoint_sha256"] != settings["checkpoint_sha256"]:
        raise ValueError("QuaRot checkpoint SHA256 differs from pinned configuration")
    return checkpoint, manifest


def _native_modules(source_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    names = ("utils", "model_utils", "hadamard_utils", "quant_utils", "rotation_utils", "monkeypatch")
    source = source_root / "fake_quant"
    for name in names:
        path = source / f"{name}.py"
        if _sha256(path) != manifest["source_sha256"][f"fake_quant/{name}.py"]:
            raise ValueError(f"QuaRot inference source changed: {path}")
        existing = sys.modules.get(name)
        if existing is not None and Path(getattr(existing, "__file__", "")).resolve() != path.resolve():
            raise RuntimeError(f"Conflicting {name} import; use the isolated QuaRot environment/process")
    sys.path.insert(0, str(source))
    return {name: importlib.import_module(name) for name in names}


def load_quarot_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("QuaRot model loading requires a Slurm GPU allocation")
    import torch
    import transformers
    from accelerate import init_empty_weights
    from transformers.modeling_utils import no_init_weights
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    if torch.__version__.split("+")[0] != "2.2.1" or transformers.__version__ != "4.44.2":
        raise RuntimeError("Use the pinned quarot environment (torch 2.2.1, transformers 4.44.2)")
    checkpoint, manifest = _validate_artifact(config)
    # torch.save includes trusted native WeightQuantizer objects. Verify the
    # pinned, self-produced file BEFORE deserializing it, never arbitrary files.
    if _sha256(checkpoint) != config["quarot"]["checkpoint_sha256"]:
        raise ValueError("QuaRot checkpoint SHA256 verification failed")
    native = _native_modules(Path(config["quarot"]["source_root"]), manifest)
    mu, qu = native["model_utils"], native["quant_utils"]
    hu, ru = native["hadamard_utils"], native["rotation_utils"]
    cfg = transformers.AutoConfig.from_pretrained(config["model_path"], local_files_only=True)
    if (cfg.model_type, cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.rms_norm_eps) != (
            "llama", 4096, 14336, 32, 32, 8, 1e-5):
        raise ValueError("This saved-state loader supports only the validated Llama-3-8B architecture")
    dtype_old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        # Keep nonpersistent RoPE buffers real; they are NOT in the checkpoint.
        with init_empty_weights(include_buffers=False), no_init_weights():
            model = transformers.AutoModelForCausalLM.from_config(
                cfg, attn_implementation=config.get("attention_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
            )
            mu.replace_modules(model, LlamaRMSNorm,
                               lambda _: mu.RMSN(cfg.hidden_size), replace_layers=False)
            qu.add_actquant(model)
    finally:
        torch.set_default_dtype(dtype_old)
    wrappers = qu.find_qlayers(model, layers=[qu.ActQuantWrapper])
    for name, wrapper in wrappers.items():
        if "down_proj" in name:
            wrapper.had_K, wrapper.K = hu.get_hadK(cfg.intermediate_size)
            wrapper.online_full_had = True
            wrapper.fp32_had = manifest["fp32_had"]
        if "o_proj" in name:
            wrapper.had_K, wrapper.K = hu.get_hadK(cfg.num_attention_heads)
            wrapper.online_partial_had = True
            wrapper.had_dim = cfg.hidden_size // cfg.num_attention_heads
            wrapper.fp32_had = manifest["fp32_had"]

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weight_quantizers = saved["w_quantizers"]
    if len(weight_quantizers) != 224 or any(q.bits != 4 for q in weight_quantizers.values()):
        raise ValueError("Expected 224 saved four-bit GPTQ weight quantizers")
    model.load_state_dict(saved["model"], strict=True, assign=True)
    # assign=True can create distinct Parameter objects for native alias keys.
    # Rebind aliases before .to(cuda), avoiding duplicated GPU weight storage.
    for wrapper in wrappers.values():
        if wrapper.weight.data_ptr() != wrapper.module.weight.data_ptr():
            raise ValueError("Saved native wrapper weight alias is inconsistent")
        wrapper.weight = wrapper.module.weight
        wrapper.bias = wrapper.module.bias
    del saved, weight_quantizers

    # Native saving precedes runtime activation/K/V configuration. Restore it
    # AFTER strict loading so missing/mismatched saved state cannot be hidden.
    for name, wrapper in wrappers.items():
        wrapper.quantizer.configure(bits=16 if "lm_head" in name else 4,
                                    groupsize=-1, sym=True, clip_ratio=manifest["a_clip_ratio"])
        if "v_proj" in name:
            wrapper.out_quantizer.configure(bits=4, groupsize=manifest["v_groupsize"],
                                             sym=False, clip_ratio=manifest["v_clip_ratio"])
    for layer in mu.get_layers(model):
        ru.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn, mu.get_rope_function_name(model), config=cfg,
            k_bits=4, k_groupsize=manifest["k_groupsize"],
            k_sym=False, k_clip_ratio=manifest["k_clip_ratio"],
        )
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("QuaRot reconstruction left unmaterialized tensors")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config = transformers.GenerationConfig.from_pretrained(config["model_path"], local_files_only=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Explicit benchmark/generation requests still override this.
    model.seqlen = manifest["seqlen"]
    model._safe_quant_runtime = {
        "method": "QuaRot W4A4KV4", "checkpoint_sha256": manifest["checkpoint_sha256"],
        "strict_state_load": True, "weight_quantizers": 224,
        "activation_quantizers": sum(w.quantizer.bits == 4 for w in wrappers.values()),
        "value_quantizers": sum(w.out_quantizer.bits == 4 for w in wrappers.values()),
        "key_quantizers": sum(isinstance(m, ru.QKRotationWrapper) and m.k_bits == 4 for m in model.modules()),
        "underlying_precision": "BF16; upstream RMSN unchanged",
        "cache_storage": "fake quantized floating point, not packed INT4",
        "calibration_or_offline_rotation_run": False,
    }
    if tuple(model._safe_quant_runtime[k] for k in (
            "activation_quantizers", "value_quantizers", "key_quantizers")) != (224, 32, 32):
        raise RuntimeError("QuaRot runtime quantizers are incomplete")
    return model.eval().to("cuda:0"), tokenizer
