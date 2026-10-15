"""Check QuaRot imports, exact tokenizer, and interfaces without model inference."""

import ast
import importlib
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
from unittest.mock import patch

BASE_MODEL = Path(
    "/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/bf16"
)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True
    source_root = Path(__file__).resolve().parents[1]
    fake_quant = source_root / "fake_quant"
    sys.path.insert(0, str(fake_quant))

    import torch
    import transformers
    import fast_hadamard_transform_cuda
    from transformers.models.llama import modeling_llama

    entrypoint = importlib.import_module("main")
    utils = importlib.import_module("utils")
    assert Path(entrypoint.__file__).resolve() == fake_quant / "main.py"

    config = transformers.AutoConfig.from_pretrained(BASE_MODEL, local_files_only=True)
    assert config.model_type == "llama"
    assert (config.hidden_size, config.num_attention_heads,
            config.num_key_value_heads, config.vocab_size) == (4096, 32, 8, 128256)
    assert config.torch_dtype == torch.bfloat16

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, use_fast=False
    )
    fast_tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_MODEL, local_files_only=True, use_fast=True
    )
    for prompt in ("Hello world!", " Yes", " No", "A short tokenizer check."):
        assert tokenizer.encode(prompt) == fast_tokenizer.encode(prompt)
    assert len(tokenizer) == config.vocab_size
    assert tokenizer.bos_token_id == config.bos_token_id
    assert tokenizer.eos_token_id == config.eos_token_id
    messages = [{"role": "user", "content": "Hello."}]
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert chat_ids == fast_tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert chat_ids[0] == config.bos_token_id
    assert tokenizer.encode(" Yes", add_special_tokens=False) == [7566]
    assert tokenizer.encode(" No", add_special_tokens=False) == [2360]

    # Parse without creating directories or log files, and check the local-path fix.
    cli = ["quarot-compatibility", "--model", str(BASE_MODEL),
           "--save_name", "compatibility-check", "--w_bits", "4",
           "--a_bits", "4", "--k_bits", "4", "--v_bits", "4",
           "--k_groupsize", "128", "--v_groupsize", "128"]
    with patch.object(sys, "argv", cli), patch.object(utils.os, "makedirs"), \
            patch.object(utils, "config_logging"):
        args = utils.parser_gen()
    assert Path(args.save_path).is_relative_to(fake_quant / "experiments")
    assert not Path(args.save_path).is_relative_to(BASE_MODEL)

    interfaces = {}
    for attention in (modeling_llama.LlamaAttention, modeling_llama.LlamaSdpaAttention):
        parameters = inspect.signature(attention.forward).parameters
        assert {"position_ids", "attention_mask", "past_key_value", "use_cache"} <= parameters.keys()
        source = inspect.getsource(attention.forward)
        tree = ast.parse(textwrap.dedent(source))
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "apply_rotary_pos_emb" for node in ast.walk(tree))
        assert attention.forward.__globals__["apply_rotary_pos_emb"] is modeling_llama.apply_rotary_pos_emb
        interfaces[attention.__name__] = "direct rotary-call interface found"

    report = {
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in (
            "torch", "transformers", "accelerate", "datasets", "tokenizers",
            "numpy", "fast-hadamard-transform")},
        "torch_cuda": torch.version.cuda,
        "compiled_architectures": torch._C._cuda_getArchFlags(),
        "hadamard_extension": fast_hadamard_transform_cuda.__file__,
        "quarot_import": "passed",
        "local_model_config": "passed",
        "tokenizer_and_chat_template": "passed",
        "yes_no_token_ids": "passed",
        "local_model_cli_and_output_path": "passed",
        "attention_interfaces": interfaces,
        "not_tested": ["model loading/inference", "GPU kernels", "BF16 RMSNorm parity",
                       "rotated model parity", "GPTQ", "KV cache numerics", "checkpoint reload"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
