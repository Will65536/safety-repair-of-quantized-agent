"""Load an existing SpinQuant state without running quantization."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Tuple


def _required_file(value: str, description: str) -> str:
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return str(path)


def _runtime_args(config: Dict[str, Any]) -> SimpleNamespace:
    if config.get("artifact_format") != "spinquant_qmodel_state":
        raise ValueError(
            "SpinQuant saved-state loading requires "
            "artifact_format=spinquant_qmodel_state"
        )
    options = config["spinquant"]
    precision = config["precision"]
    actual = (
        int(precision["weights"]),
        int(precision["activations"]),
        int(precision["kv_cache"]),
    )
    if actual != (4, 4, 4):
        raise ValueError(f"SpinQuant config must describe W4A4KV4, got {actual}")

    rotation_path = _required_file(options["rotation_path"], "SpinQuant rotation")
    state_path = _required_file(
        options["quantized_state_path"], "SpinQuant quantized state"
    )
    return SimpleNamespace(
        seed=int(options.get("seed", 0)),
        rotate=True,
        rotate_mode=options.get("rotate_mode", "hadamard"),
        rotation_seed=int(options.get("rotation_seed", -1)),
        optimized_rotation_path=rotation_path,
        fp32_had=bool(options.get("fp32_had", False)),
        w_bits=4,
        w_groupsize=int(options.get("w_groupsize", -1)),
        w_asym=bool(options.get("w_asym", False)),
        w_rtn=False,
        w_clip=False,
        nsamples=0,
        percdamp=0.01,
        act_order=False,
        a_bits=4,
        a_groupsize=int(options.get("a_groupsize", -1)),
        a_asym=bool(options.get("a_asym", True)),
        a_clip_ratio=float(options.get("a_clip_ratio", 1.0)),
        int8_down_proj=bool(options.get("int8_down_proj", False)),
        v_bits=4,
        v_groupsize=int(options.get("v_groupsize", 128)),
        v_asym=bool(options.get("v_asym", True)),
        v_clip_ratio=float(options.get("v_clip_ratio", 1.0)),
        k_bits=4,
        k_groupsize=int(options.get("k_groupsize", 128)),
        k_asym=bool(options.get("k_asym", True)),
        k_pre_rope=False,
        k_clip_ratio=float(options.get("k_clip_ratio", 1.0)),
        load_qmodel_path=state_path,
        save_qmodel_path=None,
        export_to_et=False,
        capture_layer_io=False,
        layer_idx=0,
        bsz=1,
    )


@contextmanager
def _skip_overwritten_preload_transforms(spinquant_main: Any) -> Iterator[None]:
    """Skip weight transforms that a complete saved state immediately replaces.

    SpinQuant's ``ptq_model`` must still construct its activation and KV
    quantization wrappers.  For ``load_qmodel_path``, however, the preceding
    layer-norm fusion and rotation only modify parameters that the subsequent
    strict ``load_state_dict`` replaces.  Avoiding those transforms preserves
    the saved model state while removing expensive float64 matrix operations
    and CPU/GPU transfers from every load.
    """

    fuse_module = spinquant_main.fuse_norm_utils
    rotation_module = spinquant_main.rotation_utils
    original_fuse = fuse_module.fuse_layer_norms
    original_rotate = rotation_module.rotate_model
    fuse_module.fuse_layer_norms = lambda model: None
    rotation_module.rotate_model = lambda model, args: None
    try:
        yield
    finally:
        fuse_module.fuse_layer_norms = original_fuse
        rotation_module.rotate_model = original_rotate


def load_spinquant_model(config: Dict[str, Any]) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoTokenizer

    try:
        from eval_utils import main as spinquant_main
        from eval_utils.modeling_llama import LlamaForCausalLM
        from utils.quant_utils import WeightQuantizer
    except ImportError as error:
        raise RuntimeError(
            "The SpinQuant checkout must be available in PYTHONPATH when using "
            "the spinquant loader"
        ) from error

    ptq_args = _runtime_args(config)
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

    model_config = AutoConfig.from_pretrained(config["model_path"], **common)
    clone_embeddings = bool(model_config.tie_word_embeddings)
    if clone_embeddings:
        model_config.tie_word_embeddings = False

    model = LlamaForCausalLM.from_pretrained(
        config["model_path"],
        config=model_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **common,
    )
    if clone_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()

    model = model.to("cuda:0")
    model.eval()

    # load_qmodel_path is mandatory above, so SpinQuant loads existing weights and
    # cannot enter its GPTQ or RTN creation branches.
    def load_saved_state() -> Any:
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            return spinquant_main.ptq_model(
                ptq_args,
                model,
                SimpleNamespace(input_model=config["model_path"]),
            )
        # PyTorch 2.6+ defaults torch.load to weights_only=True. SpinQuant's
        # official saved state includes WeightQuantizer instances in addition
        # to tensors, so allow only that checkpoint class during this load.
        with safe_globals([WeightQuantizer]):
            return spinquant_main.ptq_model(
                ptq_args,
                model,
                SimpleNamespace(input_model=config["model_path"]),
            )

    fast_saved_state_load = bool(
        config["spinquant"].get("skip_overwritten_preload_transforms", False)
    )
    if fast_saved_state_load:
        with _skip_overwritten_preload_transforms(spinquant_main):
            model = load_saved_state()
    else:
        model = load_saved_state()
    model = model.to("cuda:0")
    non_cuda = [
        name for name, parameter in model.named_parameters()
        if parameter.device.type != "cuda"
    ]
    if non_cuda:
        raise RuntimeError(f"SpinQuant parameters remain off GPU: {non_cuda[:20]}")

    model.eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
