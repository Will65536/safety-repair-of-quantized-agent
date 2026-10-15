---
license: llama3
base_model: meta-llama/Meta-Llama-3-8B-Instruct
tags:
- quantized
- quarot
- w4a4kv4
---

# Llama 3 8B Instruct — QuaRot W4A4KV4

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
`quantizers/QuaRot/`, with upstream commits, source hashes, licenses and local
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

## Artifact format and provenance

Completed native QuaRot paper-style run:
`paperstyle-22914989-20261015T081845171928Z`. Native fake-quantized BF16
`qmodel.pt`, not packed INT4 and not a standalone Hugging Face model. The project
loader restores online transforms and activation/K/V quantizers from the frozen
recipe; do not repeat GPTQ or offline rotation. Native WikiText-2 PPL is
`10.027104377746582` on 141 length-2048 sequences. This is not an exact
reproduction of the paper's model/precision condition. Fresh-process evaluation
loader validation remains pending; no benchmark readiness is implied.

Upstream code: https://github.com/spcl/QuaRot
at `5008669b08c1f11f9b64d52d16fddd47ca754c5a`, plus the included source patch.
The original run manifest, source patch, environment lock, calibration manifest,
and native PPL outputs are preserved as producing provenance. The run manifest's
absolute paths and historical adapter-status fields are not rewritten.
