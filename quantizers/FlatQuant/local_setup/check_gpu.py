"""Single-GPU inference-only smoke. Not a benchmark or a quantization run."""
import os
if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use a user-submitted Slurm GPU job; never run on a login node")

import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from runtime_support import SETUP, MODEL, import_runtime, load_model


def main():
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("Slurm job has no accessible CUDA GPU")
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    import_runtime()
    import deploy
    from deploy.functional.quantization import pack_i4, unpack_i4
    from flash_attn import flash_attn_func
    from fast_hadamard_transform import hadamard_transform

    with torch.inference_mode():
        # Integer inputs and unit scales avoid rounding ambiguity in this test.
        a = (torch.arange(16 * 128).reshape(16, 128) % 16 - 8).to(torch.int8)
        b = (torch.arange(32 * 128).reshape(32, 128) % 15 - 7).to(torch.int8)
        packed_a = deploy.sym_quant(a.cuda().half(), torch.ones(16, device="cuda", dtype=torch.float16))
        torch.testing.assert_close(unpack_i4(packed_a).cpu(), a.int(), rtol=0, atol=0)
        result = deploy.matmul(packed_a, pack_i4(b).cuda())
        expected = a.int() @ b.int().T
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
        restored = deploy.sym_dequant(result, torch.ones(16, device="cuda", dtype=torch.float16),
                                    torch.ones(32, device="cuda", dtype=torch.float16))
        torch.testing.assert_close(restored.cpu(), expected.half(), rtol=0, atol=0)
        q = torch.randn(1, 8, 4, 128, dtype=torch.float16, device="cuda")
        flash = flash_attn_func(q, q, q, causal=True)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2), q.float().transpose(1, 2), q.float().transpose(1, 2), is_causal=True).transpose(1, 2)
        torch.testing.assert_close(flash.float(), reference, rtol=0.01, atol=0.01)
        values = torch.randn(2, 128)
        reference_had = values.clone()
        stride = 1
        while stride < 128:
            blocks = reference_had.reshape(2, -1, 2, stride)
            left, right = blocks[:, :, 0].clone(), blocks[:, :, 1].clone()
            reference_had = torch.stack((left + right, left - right), dim=2).reshape(2, 128)
            stride *= 2
        torch.testing.assert_close(hadamard_transform(values.cuda(), scale=128 ** -0.5).cpu(),
                                   reference_had * 128 ** -0.5, rtol=1e-5, atol=1e-5)
        print(json.dumps({"kernel_checks": "passed"}), flush=True)

        job = os.environ["SLURM_JOB_ID"]
        report_path = SETUP / "logs" / f"load-{job}.json"
        model = load_model(report_path)
        linears = [m for m in model.modules() if isinstance(m, deploy.nn.Linear4bit)]
        assert len(linears) == 32 * 7
        assert all(m.weight.dtype == torch.uint8 and m.weight.shape[-1] == m.in_features // 2 for m in linears)
        assert all(torch.isfinite(m.weight_scales).all() and (m.weight_scales > 0).all() for m in linears)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": "Say hello briefly."}],
                                           add_generation_prompt=True, return_tensors="pt").cuda()
        model._expected_max_length = ids.shape[1] + 4
        out = model(ids, use_cache=True, return_dict=True)
        assert torch.isfinite(out.logits).all()
        cache = out.past_key_values
        inner = cache.inner
        assert not inner.disable_quant and inner.pages.dtype == torch.uint8
        assert inner.pages.shape[-1] == 64
        generated = []
        for _ in range(4):
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(int(token.item()))
            out = model(token, past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            assert torch.isfinite(out.logits).all()
        assert cache.get_seq_length() == ids.shape[1] + 4
        torch.cuda.synchronize()
        print(json.dumps({"inference_smoke": "passed", "gpu": torch.cuda.get_device_name(0),
                          "packed_linear_count": len(linears), "cache_dtype": model.cache_dtype,
                          "cache_storage": str(inner.pages.dtype), "nonpacked_precision": "FP16 published runtime",
                          "generated_token_ids": generated, "generated_text": tokenizer.decode(generated),
                          "elapsed_seconds": time.monotonic() - started,
                          "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                          "load_audit": str(report_path),
                          "matched_evaluation_compatibility": "not_yet_validated"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
