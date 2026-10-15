---
license: llama3
base_model: meta-llama/Meta-Llama-3-8B-Instruct
tags:
- quantized
- spinquant
- w4a4kv4
---

# Llama 3 8B Instruct — SpinQuant W4A4KV4

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
`quantizers/SpinQuant/`, with upstream commits, source hashes, licenses and local
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

## Artifact format

Native SpinQuant fake-quantized saved state: `qmodel.pt` and `R.bin`, approximately
16.14 GB together. W4A4KV4 describes the numerical quantization condition, not
packed four-bit checkpoint storage. Nonquantized portions use the project's BF16
base/runtime. This is **not** a standalone `AutoModel.from_pretrained(repo_id)`
model. Load with the project's `spinquant_saved_state` adapter and exact native
SpinQuant modules; do not repeat calibration or rotation on the loaded state.

Upstream code: https://github.com/facebookresearch/SpinQuant
at `8f47aa3f00e8662caf1a484153920a07e5281c3a`, plus the documented local patch.
The quantizer code retains its CC BY-NC 4.0 license; see `LICENSE.spinquant-code`.
The full checkpoint-producing calibration history has not been reconstructed
for this handoff; do not infer it from the current inference-only loader flags.
