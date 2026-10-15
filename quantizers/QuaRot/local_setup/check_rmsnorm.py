"""CPU-only upstream RMSNorm diagnostic; differences from Llama are reported."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fake_quant"))

import torch
import model_utils
from transformers.models.llama.modeling_llama import LlamaRMSNorm


def check(device="cpu"):
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(2, 8, 4096, generator=generator)
    results = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for scale in (0.01, 1.0, 1000.0):
            x = (values * scale).to(device=device, dtype=dtype)
            reference_norm = LlamaRMSNorm(4096, eps=1e-5).to(device=device, dtype=dtype)
            norm = model_utils.RMSN(4096, eps=1e-5).to(device=device)
            with torch.inference_mode():
                reference = reference_norm(x)
                actual = norm(x)
                assert actual.dtype == dtype
                # Require the exact upstream expression, not Llama equivalence.
                upstream_input = x.float() if dtype == torch.float16 else x
                expected = (upstream_input * torch.rsqrt(
                    upstream_input.pow(2).sum(-1, keepdim=True) / 4096 + 1e-5)).to(dtype)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
            results.append({"dtype": str(dtype), "scale": scale,
                            "upstream_expression_max_abs_error": (actual.float() - expected.float()).abs().max().item(),
                            "llama_max_abs_error": (actual.float() - reference.float()).abs().max().item(),
                            "llama_exact_match": torch.equal(actual, reference)})
    return results


if __name__ == "__main__":
    print(json.dumps({"upstream_rmsnorm_check": "passed", "device": "cpu",
                      "llama_difference_policy": "report_only", "cases": check()}, indent=2))
