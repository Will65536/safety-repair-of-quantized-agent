# Download the published baseline checkpoints

All three student repositories are public. Quantized checkpoint bytes are
unchanged; file hashes were verified against the pinned remote revisions.
See `sharing/PUBLICATION.json` for the upload receipts and `SHARING.md` for
native runtime, build and portability limitations. Downloads need no GPU.

Use the HF CLI in a separate transfer environment rather than upgrading any
working quantizer environment. Choose your storage directory first:

```bash
export SAFE_QUANT_MODEL_ROOT="/path/to/models/safe-quant"
```

## SpinQuant W4A4KV4

```bash
hf download wliu62/Llama-3-8B-Instruct-SpinQuant-W4A4KV4 \
  --revision bcd12ad54d175106b359c4bc9c8e3b23d736aa80 \
  --local-dir "$SAFE_QUANT_MODEL_ROOT/llama-3-8b-instruct/spinquant-w4a4kv4"
```

Both `qmodel.pt` and `R.bin` are required. This native fake-quantized saved
state needs the project's loader and native SpinQuant source; it is not a
standalone packed INT4 Hugging Face model.

## FlatQuant W4A4KV4

```bash
hf download wliu62/Llama-3-8B-Instruct-FlatQuant-W4A4KV4 \
  --revision 8bbd40bcda79f2bb0c2864f4b94509604f30e27e \
  --local-dir "$SAFE_QUANT_MODEL_ROOT/llama-3-8b-instruct/flatquant-w4a4kv4"
```

This preserves all 46 files of the original publisher's pinned snapshot and
adds attribution/license/integrity handoff files. Read `HANDOFF.md`; the original
`README.md` remains unchanged for integrity compatibility. Runtime uses FP16
nonpacked tensors and packed INT4 weights/cache. Rebuild the matching deployment
extension; no compiled CUDA binary is provided. Credit the original publisher
`Hyun9junn` and the FlatQuant authors.

## QuaRot W4A4KV4

```bash
hf download wliu62/Llama-3-8B-Instruct-QuaRot-W4A4KV4 \
  --revision 4f81b87fc399027c3fdb8911486065b8767ab266 \
  --local-dir "$SAFE_QUANT_MODEL_ROOT/llama-3-8b-instruct/quarot-w4a4kv4/paperstyle-22914989-20261015T081845171928Z"
```

This is only the completed paper-style native run, not the interrupted runs.
Its native PPL is 10.027104377746582. Fresh-process benchmark-loader validation
is still pending; uploading the artifact does not establish training readiness.
Online activation/K/V settings must be reconstructed by the native-compatible
loader rather than inferred from `qmodel.pt` alone.

## BF16 teacher/base model

BF16 weights are not included in the three student uploads. All ten local base
files were matched against the official Meta repository at revision
`8afb486c1db24fe5011ec46dfbe5b5dccdb575c2`, using remote LFS SHA256/size or Git
blob SHA1/size as appropriate. Their independent SHA256 hashes are recorded in
`sharing/base_model_manifest.json` and `sharing/base_model_SHA256SUMS`.

Obtain your own authorized access to Meta's gated repository and authenticate
locally. The include list avoids downloading the additional legacy weight copy:

```bash
hf auth login

hf download meta-llama/Meta-Llama-3-8B-Instruct \
  --revision 8afb486c1db24fe5011ec46dfbe5b5dccdb575c2 \
  --include "config.json" "generation_config.json" "model*.safetensors" \
    "model.safetensors.index.json" "tokenizer*.json" "special_tokens_map.json" \
  --local-dir "$SAFE_QUANT_MODEL_ROOT/llama-3-8b-instruct/bf16"
```

## Verify and load

From each downloaded student directory, run `sha256sum -c SHA256SUMS`. From
the downloaded BF16 directory, run `sha256sum -c` with the absolute path to
the checkout's `sharing/base_model_SHA256SUMS`. File verification is not inference.

Downloading into these directories preserves the expected model layout, but
does **not** automatically fix Delta-specific paths in runtime helpers,
launchers, configs, or producing manifests. Follow the explicit path-adaptation
notes in `SHARING.md`; do not silently rewrite historical provenance or bypass
integrity checks. Rebuild extensions in separate compatible environments.
Run GPU loading/smoke tests in a Slurm allocation on Delta, never on login nodes.
Distillation/backpropagation support remains unvalidated for all three artifacts.
