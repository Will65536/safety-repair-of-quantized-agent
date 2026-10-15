"""Local, pinned inference imports and audited loading for smoke tests only."""
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

SETUP = Path(__file__).resolve().parent
MODEL = Path("/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/flatquant-w4a4kv4")
BASE = MODEL.parent / "bf16"
RUNTIME = SETUP / "runtime"


def import_runtime():
    sys.path.insert(0, str(RUNTIME))
    import deploy
    assert Path(deploy.__file__).resolve() == RUNTIME / "deploy/__init__.py"
    import deploy._CUDA
    assert Path(deploy._CUDA.__file__).resolve().is_relative_to(RUNTIME)
    spec = importlib.util.spec_from_file_location("pinned_flatquant_model", MODEL / "modeling_llama.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(report_path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model loading/smoke inference must run in a Slurm GPU allocation")
    import torch
    module = import_runtime()
    original = torch.nn.Module.load_state_dict
    records, covered, expected = [], set(), set()

    def audited_load(instance, state, *args, **kwargs):
        expected.update(instance.state_dict())
        covered.update(state)
        result = original(instance, state, *args, **kwargs)
        records.append({"provided": len(state), "unexpected": result.unexpected_keys,
                        "missing_this_stage": result.missing_keys})
        return result

    dtype_old = torch.get_default_dtype()
    try:
        with patch.object(torch.nn.Module, "load_state_dict", audited_load):
            model = module.FlatQuantLlamaForCausalLM.from_pretrained(str(MODEL), local_files_only=True)
    finally:
        torch.set_default_dtype(dtype_old)

    missing = sorted(expected - covered)
    # Upstream replaces shared input matrices after loading. diag_scale is not
    # read by OnlineTrans.forward. Separate Quantizers receive already-packed
    # tensors and return them unchanged in this FlatQuant matmul configuration.
    aliases = r"model\.layers\.\d+\.(self_attn\.inp_trans_[qkv]|mlp\.inp_trans_[ug])\.(left_matrix|right_matrix)"
    bypassed = r"model\.layers\.\d+\.(self_attn\.quantizer_[qkv]|mlp\.quantizer_[gu]|self_attn\.o_proj\.0|mlp\.down_proj\.1)\.clip_factor_a_(max|min)"
    unresolved = [k for k in missing if not (re.fullmatch(aliases, k) or re.fullmatch(bypassed, k) or k.endswith(".diag_scale"))]
    # Retained calibration-only fields in the source artifact are not runtime
    # state: packed weights already incorporate weight clipping/inverse transforms.
    unused = r".*\.(clip_factor_w_(max|min)|left_matrix_inv|right_matrix_inv|right_matrix_inv_t|trans_matrix_v_inv_t)"
    unexpected = sorted({k for r in records for k in r["unexpected"]})
    unknown = [k for k in unexpected if not re.fullmatch(unused, k)]
    report = {"load_calls": records, "missing_after_both_stages": missing,
              "unresolved_runtime_state": unresolved, "unrecognized_checkpoint_state": unknown}
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    if unresolved or unknown:
        raise RuntimeError(f"Unverified checkpoint loading: missing={unresolved[:12]}, unknown={unknown[:12]}; see {report_path}")
    assert model.cache_dtype == "int4"
    return model.eval().to(device="cuda:0")
