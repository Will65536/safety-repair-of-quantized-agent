"""Check actual QuaRot Hadamard helpers inside Slurm; no model loading."""

import json
import os
from pathlib import Path
import sys


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run this check inside a user-submitted Slurm allocation.")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fake_quant"))

    import torch
    import hadamard_utils

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access a CUDA GPU.")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The GPU does not support the requested BF16 runtime.")
    torch.manual_seed(0)
    results = []
    with torch.inference_mode():
        for dtype in (torch.float32, torch.bfloat16):
            for width in (128, 4096, 14336):
                values = torch.randn(2, width).to(dtype)
                reference = hadamard_utils.matmul_hadU(values.float())
                had_k, factor = hadamard_utils.get_hadK(width)
                actual = hadamard_utils.matmul_hadU_cuda(
                    values.cuda(), had_k, factor
                ).float().cpu()
                tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
                torch.testing.assert_close(actual, reference,
                                           atol=tolerance, rtol=tolerance)
                results.append({"dtype": str(dtype), "width": width,
                                "factor": factor,
                                "max_abs_error": (actual - reference).abs().max().item()})
    properties = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": properties.name,
                      "memory_gib": round(properties.total_memory / 2**30, 1),
                      "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "quarot_hadamard_fp32_bf16": "passed", "cases": results,
                      "model_loading_and_inference": "not_run"}, indent=2))


if __name__ == "__main__":
    main()
