# Llama 3 8B Instruct — FlatQuant W4A4KV4

Built with Meta Llama 3.

Research artifact handoff for [Safety Repair of Quantized Agent](https://github.com/Will65536/safety-repair-of-quantized-agent).
The original quantized checkpoint bytes are unchanged. No distillation or safety
repair has been applied to this baseline. This is not an official release by Meta
or the quantizer authors.

Base model: [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
Obtain the BF16 teacher/base model and tokenizer separately under its applicable
access conditions. See `LICENSE`, `NOTICE`, `artifact_manifest.json`, and
`SHA256SUMS`. Use only trusted native `.pt` checkpoints and deployment code.

## Code and runtime

The project includes `src/safe_quant/model_loaders/` and a source snapshot at
`quantizers/FlatQuant/`, with upstream commits, source hashes, licenses and local
patches recorded in `quantizers/SOURCES.json`. Current launchers and some manifests
contain Delta-specific paths: downloading this model alone is not a fully portable
installation. Follow `SHARING.md` and `docs/QUANTIZATION_GUIDE.md` in the code repo.
Use a compatible isolated environment, rebuild native extensions for the target
PyTorch/CUDA ABI, and do not quantize or alter this baseline while loading.
GPU smoke checks require an allocated GPU; on Delta, use Slurm rather than login
nodes. No model inference was performed as part of this upload.

## Training caveat

This is an inference/evaluation artifact, not a validated distillation or training
checkpoint. End-to-end gradients and quantization-aware parameter updates have
not been validated. Preserve the original checkpoint and save refinements to a
new artifact directory. Native runtime differences are intentional and must be
reported for matched comparisons.

## Mirror provenance and precision

This is a byte-preserving mirror of
[Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant](https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant)
at revision `65380056097d262d882e427359c9c8be2d63abf7`, not a model quantized
by this project. Credit the original checkpoint publisher and FlatQuant authors.
The original `README.md` and all 46 downloaded source-artifact files are
preserved unchanged so the project's pinned integrity checks remain valid.

Native deployment uses packed INT4 weights and KV cache, **FP16** nonpacked
tensors, and FlashAttention-2. It is not a BF16 runtime. Rebuild the matching
deployment extension; do not use a bundled or mismatched CUDA binary. The
project's inference-only environment is not a validated FlatQuant quantization
environment. Packed deployment weights are not directly trainable parameters.
`use_cache=False` is unsupported by the published runtime; do not silently
change scientific evaluation settings. Native prefill attention and cached
autoregressive decode have different KV paths.

Upstream quantization code: https://github.com/ruikangliu/FlatQuant
at `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`. Pinned deployment code comes
from the checkpoint revision above and can differ from this upstream checkout.
