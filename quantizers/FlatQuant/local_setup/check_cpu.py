"""No model construction, inference, or GPU queries; validate files/imports/tokenizer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
from collections import Counter

import torch
import flash_attn_2_cuda
import fast_hadamard_transform_cuda
from safetensors import safe_open
from transformers import AutoTokenizer
from runtime_support import SETUP, MODEL, BASE, RUNTIME


def main():
    manifest = json.loads((SETUP / "checkpoint_manifest.json").read_text())
    assert manifest["revision"] == "65380056097d262d882e427359c9c8be2d63abf7"
    for name, info in manifest["files"].items():
        path = MODEL / name
        assert path.stat().st_size == info["bytes"], name
        if not name.endswith(".safetensors"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"], name
        if name.startswith("deploy/") and name.endswith(".py"):
            assert (RUNTIME / name).read_bytes() == path.read_bytes(), name
    assert not list(MODEL.rglob("*.so"))
    config = json.loads((MODEL / "config.json").read_text())
    quant = config["quantization_config"]
    assert (quant["w_bits"], quant["a_bits"], quant["kv_bits"]) == (4, 4, 4)
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    dtypes = Counter()
    for shard in set(index["weight_map"].values()):
        with safe_open(MODEL / shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dtypes[handle.get_slice(name).get_dtype()] += 1
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    reference = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    texts = ["Hello, world!", " Yes", " No", "A short compatibility check."]
    for text in texts:
        assert tokenizer.encode(text) == reference.encode(text), text
    chat = [{"role": "user", "content": "Hello."}]
    assert tokenizer.apply_chat_template(chat, add_generation_prompt=True) == reference.apply_chat_template(chat, add_generation_prompt=True)
    assert tokenizer.get_vocab() == reference.get_vocab()
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id) == (reference.bos_token_id, reference.eos_token_id)
    assert tokenizer.encode(" Yes", add_special_tokens=False) == [7566]
    assert tokenizer.encode(" No", add_special_tokens=False) == [2360]
    # Triton 3.3 initializes its CUDA driver while decorating autotuned
    # functions, so importing the full deploy runtime requires the Slurm GPU
    # job. Import only the compiled pybind extension here, with GPUs hidden.
    binaries = list((RUNTIME / "deploy").glob("_CUDA*.so"))
    assert len(binaries) == 1
    spec = importlib.util.spec_from_file_location("_CUDA", binaries[0])
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    assert all(hasattr(extension, name) for name in ("matmul", "sym_quant", "sym_dequant", "batch_decode_i4"))
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "triton", "flash-attn", "fast-hadamard-transform", "huggingface-hub")}
    print(json.dumps({"cpu_checks": "passed", "python": platform.python_version(),
                      "versions": versions, "cuda_runtime": torch.version.cuda,
                      "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                      "checkpoint_files": len(manifest["files"]), "tensor_dtypes": dict(dtypes),
                      "tokenizer_and_chat": "identical_to_local_base",
                      "compiled_extension_import": "passed",
                      "full_triton_runtime_import": "deferred_to_slurm_gpu_job",
                      "model_loading_and_gpu_execution": "not_run"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
