"""Loader for standard Transformers checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple


def load_transformers_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_names = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype_name = config.get("torch_dtype", "bfloat16")
    if dtype_name not in dtype_names:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name!r}")

    token = os.environ.get("HF_TOKEN")
    common = {
        "revision": config.get("revision"),
        "token": token,
        "local_files_only": bool(config.get("local_files_only", True)),
    }
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], **common)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(common)
    model_kwargs.update(
        {
            "torch_dtype": dtype_names[dtype_name],
            "low_cpu_mem_usage": True,
        }
    )
    attention = config.get("attention_implementation")
    if attention:
        model_kwargs["attn_implementation"] = attention
    model = AutoModelForCausalLM.from_pretrained(config["model_path"], **model_kwargs)
    model = model.to("cuda:0")
    model.eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
