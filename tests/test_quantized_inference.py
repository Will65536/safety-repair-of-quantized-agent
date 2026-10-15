import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from safe_quant.config import load_model_config
from safe_quant.model_loaders import quarot
from safe_quant.model_loaders.flatquant import _guard_cache_contract, load_flatquant_model


def _quarot_config(tmp_path):
    payload = b"unit-test placeholder, never deserialized"
    (tmp_path / "qmodel.pt").write_bytes(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    manifest = {
        "status": "native_quantization_and_ppl_completed",
        "method": "QuaRot W4A4KV4", "recipe": "paper-style",
        "w_bits": 4, "a_bits": 4, "k_bits": 4, "v_bits": 4,
        "w_symmetric": True, "a_symmetric": True,
        "k_symmetric": False, "v_symmetric": False,
        "w_groupsize": -1, "a_groupsize": -1, "k_groupsize": 128, "v_groupsize": 128,
        "a_clip_ratio": 0.9, "k_clip_ratio": 0.95, "v_clip_ratio": 0.95,
        "fp32_had": True, "int8_down_proj": False,
        "upstream_weight_save_precedes_activation_and_kv_configuration": True,
        "base_model": str(tmp_path / "bf16"),
        "checkpoint_bytes": len(payload), "checkpoint_sha256": checksum,
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    return {
        "artifact_format": "quarot_qmodel_state", "model_path": str(tmp_path / "bf16"),
        "precision": {"weights": 4, "activations": 4, "kv_cache": 4},
        "quarot": {"artifact_path": str(tmp_path), "checkpoint_sha256": checksum},
    }


def test_quarot_validates_completed_pinned_artifact(tmp_path):
    config = _quarot_config(tmp_path)
    checkpoint, manifest = quarot._validate_artifact(config)
    assert checkpoint == tmp_path / "qmodel.pt"
    assert quarot._sha256(checkpoint) == manifest["checkpoint_sha256"]


@pytest.mark.parametrize("key,value", [
    ("status", "running"), ("k_bits", 16), ("recipe", "code-default"),
    ("fp32_had", False), ("v_groupsize", -1), ("a_clip_ratio", 1.0),
    ("base_model", "/wrong/base"), ("checkpoint_bytes", 1),
    ("checkpoint_sha256", "different"),
])
def test_quarot_rejects_incompatible_manifest(tmp_path, key, value):
    config = _quarot_config(tmp_path)
    path = tmp_path / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        quarot._validate_artifact(config)


def test_quarot_rejects_wrong_requested_precision_and_downloads(tmp_path):
    config = _quarot_config(tmp_path)
    config["precision"]["kv_cache"] = 16
    with pytest.raises(ValueError, match="W4A4KV4"):
        quarot._validate_artifact(config)
    config = _quarot_config(tmp_path)
    config["local_files_only"] = False
    with pytest.raises(ValueError, match="immutable"):
        quarot._validate_artifact(config)


def test_quarot_refuses_silent_dtype_change(tmp_path):
    config = _quarot_config(tmp_path)
    config["torch_dtype"] = "float16"
    with pytest.raises(ValueError, match="BF16"):
        quarot._validate_artifact(config)


def test_quarot_rejects_changed_native_source_before_import(tmp_path):
    source = tmp_path / "fake_quant"
    source.mkdir()
    (source / "utils.py").write_text("raise AssertionError('must not import')")
    with pytest.raises(ValueError, match="inference source changed"):
        quarot._native_modules(tmp_path, {"source_sha256": {"fake_quant/utils.py": "wrong"}})


@pytest.mark.parametrize("loader", [quarot.load_quarot_model, load_flatquant_model])
def test_loaders_refuse_login_before_importing_torch(monkeypatch, loader):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="Slurm GPU allocation"):
        loader({})


def test_flatquant_does_not_silently_override_use_cache_false():
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs)
        return "result"
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True), forward=original)
    _guard_cache_contract(model)
    with pytest.raises(RuntimeError, match="cannot honor use_cache=False"):
        model.forward(input_ids="placeholder", use_cache=False)
    assert not calls
    assert model.forward(input_ids="placeholder", use_cache=True) == "result"
    assert calls == [{"input_ids": "placeholder", "use_cache": True}]
    model.config.use_cache = False
    with pytest.raises(RuntimeError, match="cannot honor use_cache=False"):
        model.forward(input_ids="placeholder")


def test_inference_loader_never_calls_quantization_or_offline_weight_transforms():
    tree = ast.parse(Path(quarot.__file__).read_text())
    forbidden = {"rotate_model", "fuse_layer_norms", "gptq_fwrd", "rtn_fwrd", "get_loaders", "get_model"}
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not calls.intersection(forbidden)
    assert "load_state_dict" in calls


@pytest.mark.parametrize("method,loader", [("quarot", "quarot_saved_state"), ("flatquant", "flatquant_prequantized")])
def test_exact_llama3_instruct_configs(monkeypatch, method, loader):
    monkeypatch.setenv("SAFE_QUANT_MODEL_ROOT", "/models")
    root = Path(__file__).resolve().parents[1]
    config = load_model_config(root / "configs/models" / f"llama3_8b_instruct_{method}_w4a4kv4.yaml")
    assert config["loader"] == loader
    assert config["family"] == "llama3"
    assert config["tokenizer_path"] == "/models/llama-3-8b-instruct/bf16"
    assert config["precision"] == {"weights": 4, "activations": 4, "kv_cache": 4}
    if method == "flatquant":
        assert config["generation_config_path"] == "/models/llama-3-8b-instruct/bf16/generation_config.json"
        assert config["flatquant"]["use_cache_false_policy"] == "reject"
