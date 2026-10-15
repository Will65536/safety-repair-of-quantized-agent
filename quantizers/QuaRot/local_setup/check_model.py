"""Slurm-only BF16 QuaRot model smoke; no GPTQ, datasets, or low-bit quantization."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/bf16")


def compare(reference, actual):
    import torch
    assert reference.shape == actual.shape
    assert torch.isfinite(reference).all() and torch.isfinite(actual).all()
    ref, alt = reference.double(), actual.double()
    centered_ref = ref - ref.mean(-1, keepdim=True)
    centered_alt = alt - alt.mean(-1, keepdim=True)
    nrmse = ((centered_ref - centered_alt).square().mean() /
             centered_ref.square().mean().clamp_min(1e-12)).sqrt().item()
    lp, lq = ref.log_softmax(-1), alt.log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    return {"centered_logit_nrmse": nrmse, "mean_kl_ref_to_test": max(0.0, kl.mean().item()),
            "max_position_kl": max(0.0, kl.max().item()),
            "max_abs_logit_error": (ref - alt).abs().max().item(),
            "top1_agreement": (ref.argmax(-1) == alt.argmax(-1)).double().mean().item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nrmse", type=float, default=0.03,
                        help="Diagnostic smoke gate, not a scientific equivalence tolerance")
    parser.add_argument("--max-mean-kl", type=float, default=0.01)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Run only inside a user-requested Slurm GPU allocation")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "fake_quant"))
    import torch
    import transformers
    from transformers.cache_utils import DynamicCache
    import model_utils
    import rotation_utils
    import quant_utils
    import hadamard_utils

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This test requires a BF16-capable Slurm CUDA GPU")
    torch.cuda.set_device(0)
    transformers.set_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "local_setup/logs" / f"model-{os.environ['SLURM_JOB_ID']}-{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "output": str(output), "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "host": os.uname().nodename, "gpu": torch.cuda.get_device_name(0),
              "base_model": str(BASE), "seed": 0, "torch": torch.__version__,
              "transformers": transformers.__version__, "attention_implementation": "sdpa",
              "weight_dtype": "bfloat16", "low_bit_quantization": "disabled",
              "normalization": "unmodified upstream QuaRot RMSN",
              "rotation_mode": "hadamard", "fp32_had": False,
              "gates": vars(args), "gate_interpretation": "diagnostic only; not benchmark acceptance",
              "stages": {}, "comparisons": {}}
    examples = []

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["peak_gpu_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["peak_host_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "predictions.json").write_text(json.dumps(examples, indent=2) + "\n")

    def gate(name, reference, actual):
        metrics = compare(reference, actual)
        passed = metrics["centered_logit_nrmse"] <= args.max_nrmse and metrics["mean_kl_ref_to_test"] <= args.max_mean_kl
        metrics["passed"] = passed
        report["comparisons"][name] = metrics
        save()
        print(json.dumps({"comparison": name, **metrics}), flush=True)
        if not passed:
            raise RuntimeError(f"Diagnostic parity gate failed: {name}; inspect report before changing tolerances")

    try:
        report["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        (output / "source_changes.patch").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
        report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__).resolve(), ROOT / "fake_quant/model_utils.py",
                                             ROOT / "fake_quant/rotation_utils.py", ROOT / "fake_quant/quant_utils.py",
                                             ROOT / "local_setup/requirements.lock.txt"]}
        report["base_config_sha256"] = hashlib.sha256((BASE / "config.json").read_bytes()).hexdigest()
        save()
        print(json.dumps({"stage": "gpu_bf16_rmsnorm_diagnostic"}), flush=True)
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        with torch.inference_mode():
            norm_inputs = torch.randn(2, 8, 4096, device="cuda", dtype=torch.bfloat16)
            rmsn = model_utils.RMSN(4096, eps=1e-5).cuda()
            llama_norm = LlamaRMSNorm(4096, eps=1e-5).cuda().bfloat16()
            actual_norm, reference_norm = rmsn(norm_inputs), llama_norm(norm_inputs)
            upstream_norm = norm_inputs * torch.rsqrt(
                norm_inputs.pow(2).sum(-1, keepdim=True) / 4096 + 1e-5)
            torch.testing.assert_close(actual_norm, upstream_norm, atol=0, rtol=0)
            assert torch.isfinite(actual_norm).all() and torch.isfinite(reference_norm).all()
            report["gpu_bf16_rmsnorm_diagnostic"] = {
                "upstream_expression_match": True, "llama_difference_policy": "report_only",
                "llama_max_abs_error": (actual_norm.float() - reference_norm.float()).abs().max().item(),
                "llama_exact_match": torch.equal(actual_norm, reference_norm)}
            del norm_inputs, rmsn, llama_norm, actual_norm, reference_norm, upstream_norm
        print(json.dumps(report["gpu_bf16_rmsnorm_diagnostic"]), flush=True)
        save()
        print(json.dumps({"stage": "load_local_bf16_model", "output": str(output)}), flush=True)
        # Do not use get_model(): its loader globally disables torch initializers.
        model = transformers.LlamaForCausalLM.from_pretrained(
            BASE, local_files_only=True, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
        assert (model.config.hidden_size, model.config.num_attention_heads,
                model.config.num_key_value_heads, len(model.model.layers)) == (4096, 32, 8, 32)
        assert model.config.rms_norm_eps == 1e-5
        assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        tokenizer = transformers.AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
        prompts = ["Say hello briefly.", "What is two plus two? Answer with one short sentence."]
        inputs = [tokenizer.apply_chat_template([{"role": "user", "content": text}],
                  add_generation_prompt=True, return_tensors="pt") for text in prompts]
        continuation = tokenizer.encode(" Hello.", add_special_tokens=False)[:2]
        assert len(continuation) == 2
        (output / "inputs.json").write_text(json.dumps({"prompts": prompts,
            "input_ids": [ids.tolist() for ids in inputs], "teacher_forced_continuation": continuation}, indent=2) + "\n")

        def snapshot(stage):
            print(json.dumps({"stage": stage, "status": "running"}), flush=True)
            stage_started = time.monotonic()
            model.to("cuda:0")
            outputs = []
            with torch.inference_mode():
                for index, ids in enumerate(inputs):
                    gpu_ids = ids.cuda()
                    logits = model(gpu_ids, attention_mask=torch.ones_like(gpu_ids),
                                   use_cache=False, return_dict=True).logits.float().cpu()[0]
                    assert torch.isfinite(logits).all()
                    outputs.append(logits)
                    examples.append({"stage": stage, "example": index,
                                     "last_token_argmax": int(logits[-1].argmax()),
                                     "yes_no_logits": [float(logits[-1, 7566]), float(logits[-1, 2360])]})
                # Left-padding/masking check against each unpadded reference.
                width = max(ids.shape[1] for ids in inputs)
                padded = torch.full((len(inputs), width), tokenizer.eos_token_id, dtype=torch.long, device="cuda")
                mask = torch.zeros_like(padded)
                for i, ids in enumerate(inputs):
                    padded[i, -ids.shape[1]:] = ids[0].cuda()
                    mask[i, -ids.shape[1]:] = 1
                positions = (mask.cumsum(-1) - 1).clamp_min(0)
                batch = model(padded, attention_mask=mask, position_ids=positions,
                              use_cache=False, return_dict=True).logits.float().cpu()
                for i, ids in enumerate(inputs):
                    gate(f"{stage}/left_padding_{i}", outputs[i], batch[i, -ids.shape[1]:])
                del batch
                # Teacher-forced tokens keep every stage's cache comparison identical.
                prefix = inputs[0].cuda()
                cache = DynamicCache()
                prefill = model(prefix, past_key_values=cache, use_cache=True, return_dict=True)
                cache = prefill.past_key_values
                del prefill
                decode_logits = []
                for token_id in continuation:
                    token = torch.tensor([[token_id]], device="cuda")
                    prefix = torch.cat((prefix, token), dim=-1)
                    result = model(token, attention_mask=torch.ones_like(prefix),
                                   past_key_values=cache, use_cache=True, return_dict=True)
                    cache = result.past_key_values
                    cached_logits = result.logits[:, -1].float().cpu()
                    full_logits = model(prefix, attention_mask=torch.ones_like(prefix),
                                        use_cache=False, return_dict=True).logits[:, -1].float().cpu()
                    gate(f"{stage}/cache_step_{len(decode_logits)}", full_logits, cached_logits)
                    decode_logits.append(cached_logits)
                    del result
                assert cache.get_seq_length() == prefix.shape[1]
                assert len(cache.key_cache) == len(cache.value_cache) == 32
                for key, value in zip(cache.key_cache, cache.value_cache):
                    assert key.shape[1] == value.shape[1] == 8
                    assert key.shape[-1] == value.shape[-1] == 128
                    assert key.dtype == value.dtype == torch.bfloat16
                    assert torch.isfinite(key).all() and torch.isfinite(value).all()
            report["stages"][stage] = {"status": "passed", "seconds": time.monotonic() - stage_started,
                "parameter_dtypes": dict(Counter(str(p.dtype) for p in model.parameters())),
                "cache_dtype": "BF16; K/V quantization disabled", "gqa_kv_heads": 8}
            save()
            return outputs, torch.cat(decode_logits)

        with torch.inference_mode():
            reference, reference_decode = snapshot("original")
            torch.save({"prefill": reference, "cached_decode": reference_decode}, output / "reference_logits.pt")
            model.cpu()
            torch.cuda.empty_cache()
            print(json.dumps({"stage": "fuse_norms", "note": "includes upstream embedding centering"}), flush=True)
            rotation_utils.fuse_layer_norms(model)
            fused, _ = snapshot("fused")
            for i in range(len(inputs)):
                gate(f"fused_vs_original/{i}", reference[i], fused[i])
            del fused
            model.cpu()
            torch.cuda.empty_cache()
            print(json.dumps({"stage": "offline_rotation", "seed": 0}), flush=True)
            transformers.set_seed(0)  # Rotation identity must not depend on loading/testing RNG consumption.
            rotation_utils.rotate_model(model, SimpleNamespace(rotate_mode="hadamard"))
            quant_utils.add_actquant(model)
            wrappers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            assert len(wrappers) == 32 * 7 + 1
            for name, layer in wrappers.items():
                assert layer.quantizer.bits == layer.out_quantizer.bits == 16
                if "down_proj" in name:
                    layer.had_K, layer.K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    layer.online_full_had = True
                if "o_proj" in name:
                    layer.had_K, layer.K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    layer.online_partial_had = True
                    layer.had_dim = model.config.hidden_size // model.config.num_attention_heads
                layer.fp32_had = False
            # At K16 the wrapper still applies Q/K rotation after RoPE but does
            # not quantize keys. Head-sized grouping preserves GQA geometry.
            for block in model.model.layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    block.self_attn, model_utils.get_rope_function_name(model), config=model.config,
                    k_bits=16, k_groupsize=128, k_sym=False, k_clip_ratio=1.0)
            assert all(m.bits == 16 for m in model.modules() if isinstance(m, quant_utils.ActQuantizer))
            rotated, rotated_decode = snapshot("rotated_with_online_transforms")
            for i in range(len(inputs)):
                gate(f"rotated_vs_original/{i}", reference[i], rotated[i])
            gate("rotated_vs_original/cached_decode", reference_decode, rotated_decode)
        report["status"] = "passed"
        report["not_tested"] = ["W4A4KV4", "GPTQ/calibration", "checkpoint export/reload",
                                "long contexts", "quantization memory budget", "scientific evaluation"]
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
