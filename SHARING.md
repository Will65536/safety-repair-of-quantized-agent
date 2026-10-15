# Sharing code and quantized artifacts

This document describes the public monorepo and model-artifact handoff.
Publication receipts are recorded only after transfers have been verified;
distillation has not been implemented or validated.
Never commit credentials, model weights, environments, caches, or evaluation
prediction directories to GitHub. Keep quantized checkpoints unchanged.

## Code repositories

The user selected one public repository:
https://github.com/Will65536/safety-repair-of-quantized-agent.
Evaluation code remains at the root; audited quantizer working-tree snapshots
are under `quantizers/SpinQuant`, `quantizers/QuaRot`, and `quantizers/FlatQuant`.
These are vendored snapshots, not nested Git repositories or complete upstream
histories. `quantizers/SOURCES.json` preserves upstream commits and every selected
source file's hash. Local tracked changes are also retained as
`LOCAL_CHANGES.patch` where present. Original external checkouts, their history,
and their remote settings are unchanged.

| Component | Local directory | Existing upstream commit |
| --- | --- | --- |
| Evaluation | `/work/nvme/bgxm/wliu12/safe-quant` | No Git repository existed at the initial sharing audit |
| SpinQuant | `/work/nvme/bgxm/wliu12/repos/SpinQuant` | `8f47aa3f00e8662caf1a484153920a07e5281c3a` |
| QuaRot | `/work/nvme/bgxm/wliu12/repos/QuaRot` | `5008669b08c1f11f9b64d52d16fddd47ca754c5a` |
| FlatQuant | `/work/nvme/bgxm/wliu12/repos/FlatQuant` | `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1` |

The evaluation repository needs `src/`, `configs/`, `scripts/`, `slurm/`,
`tests/`, `requirements/`, `reference-files/`, `pyproject.toml`, `README.md`,
`AGENTS.md`, `.gitignore`, and this handoff. The two reference scripts are legacy
evaluation references rather than the current primary entry points.

The canonical operational guide remains outside the evaluation repository at
`/work/nvme/bgxm/wliu12/repos/QUANTIZATION_GUIDE.md`. A 2026-10-15 handoff
snapshot is included at `docs/QUANTIZATION_GUIDE.md`.

The root `.gitmodules` records six pinned dependency gitlinks from the two
upstream quantizers. Initialize only the dependencies needed for your selected
build; a plain source clone does not download them. For the FlatQuant inference
extension's CUTLASS headers, run from the monorepo root:

```bash
git submodule update --init --depth 1 -- quantizers/FlatQuant/third-party/cutlass
```

Each quantizer's archived upstream `.gitmodules` is kept as source provenance;
Git uses the root `.gitmodules` for this monorepo.

### Quantizer-specific additions

- SpinQuant: retain the tracked source and the change to
  `utils/process_args.py`; review `scripts/10_optimize_rotation_single_gpu.sh`
  and `run_rotation.slurm` for inclusion. The untracked `CAQ/` experiments,
  standalone legacy evaluation scripts, `.bak` files, caches, and results are
  not included in this three-quantizer handoff. The exact
  working SpinQuant environment is recorded in
  `requirements/spinquant-bench.freeze.txt`; upstream `requirement.txt` alone
  is not an environment lock. The local Hadamard build is recorded by version
  rather than an installation URL pointing to the author's filesystem.
- QuaRot: retain the tracked changes in `fake_quant/eval_utils.py`,
  `fake_quant/gptq_utils.py`, and `fake_quant/utils.py`. Include the source
  scripts and requirements from `local_setup/`, but not cached data, model
  artifacts, logs, or compiled files. Custom parity diagnostics are historical,
  not a prerequisite for the native quantization workflow.
- FlatQuant: retain the upstream quantization source and the source scripts,
  requirements, checkpoint manifest, and `runtime_support.py` in `local_setup/`.
  Include the pinned `local_setup/runtime/deploy/` Python/CUDA/header sources,
  excluding binaries and build caches. Its `checkpoint_manifest.json` must not
  be omitted by the upstream blanket `*.json` ignore rule.

Share source to rebuild native extensions, not an environment-specific `.so`.
FlatQuant quantization itself was not run locally: the evaluated artifact is a
published checkpoint with a locally rebuilt deployment runtime. Do not describe
the inference-only FlatQuant environment as a validated quantization environment.

## Weight artifacts

The authenticated Hugging Face account at the initial audit is `wliu62`.
The following public destinations were authorized by the user. Consult
`sharing/PUBLICATION.json` for verified uploaded revisions when present:

| Condition | HF repository | Exact local source |
| --- | --- | --- |
| SpinQuant W4A4KV4 | `wliu62/Llama-3-8B-Instruct-SpinQuant-W4A4KV4` | `models/safe-quant/llama-3-8b-instruct/spinquant-w4a4kv4/` |
| FlatQuant W4A4KV4 | `wliu62/Llama-3-8B-Instruct-FlatQuant-W4A4KV4` | `models/safe-quant/llama-3-8b-instruct/flatquant-w4a4kv4/` |
| QuaRot W4A4KV4 | `wliu62/Llama-3-8B-Instruct-QuaRot-W4A4KV4` | `models/safe-quant/llama-3-8b-instruct/quarot-w4a4kv4/paperstyle-22914989-20261015T081845171928Z/` |

Local model paths in this table are relative to `/work/nvme/bgxm/wliu12/`.
Exclude `.cache/`, bytecode, compiled libraries, unfinished QuaRot runs, and any
other condition. Do not overwrite an existing remote repository without checking
its contents and revision first. GitHub and these three HF repositories are
public by the user's explicit choice. No account token is included in the code,
model cards, logs or manifests; authentication remains local to the user's tools.

### Artifact identity and provenance

- SpinQuant: transfer both `qmodel.pt` (16,072,578,047 bytes) and `R.bin`
  (69,214,550 bytes). This is a native fake-quantized state, not packed INT4.
  Verified SHA256 values:
  - `qmodel.pt`: `22f81827a8e2fe6a708eae28e78a93acaf14a844a031b89da8904d58f7bb3b7c`
  - `R.bin`: `1abb326819d9486fbbb6fbfb85eb5374540c789a84931c729f16402929d22807`
- QuaRot: transfer only the completed paper-style run selected above, including
  its native `qmodel.pt` (16,072,004,244 bytes), run/calibration manifests,
  requirements, source patch, tokenizer/config files, and native PPL results.
  The expected checkpoint SHA256 is
  `fa0c5ce028544bed329941ef91f44b586a7bf0ad6ad003313055a7f27808cd42`.
  Native WikiText-2 perplexity is `10.027104377746582`; this is not an exact
  reproduction of a published BF16 Instruct result. Fresh-process evaluation
  integration remains pending validation.
- FlatQuant: this is a mirror, not a newly quantized model. Preserve attribution
  to `Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant` at revision
  `65380056097d262d882e427359c9c8be2d63abf7`. Preserve its original model card
  separately if adding a handoff card. Transfer both safetensor shards, their
  index, model/config/tokenizer files, and deployment source. The evaluated
  native runtime uses FP16 nonpacked tensors and packed INT4 weights/cache.

Every model repository needs the applicable Meta Llama 3 agreement, attribution
`NOTICE`, a clearly attributed model card, file checksums, and runtime/source
provenance. Preserve applicable third-party code licenses as well. Never relabel
native fake-quantized artifacts as standalone Hugging Face packed models.

## Recipient setup and remaining limitations

1. Install separate compatible environments; do not combine the three quantizer
   dependency sets. Upload tools should also be isolated from inference envs.
2. Obtain the exact BF16 base model/tokenizer, or download the official base
   revision and verify it matches the producing files. None of the three student
   uploads alone supplies a complete BF16 teacher.
3. Set `SAFE_QUANT_MODEL_ROOT` and place downloaded artifacts in the directory
   layout expected by `configs/models/`. Record the final remote commit hashes.
4. Adapt only filesystem paths and runtime build locations. Existing Delta paths
   remain in launchers, manifests, configs, and FlatQuant helpers. The current
   loaders strictly check some producing paths and source hashes: merely setting
   a model-root variable is not yet a complete portable loading solution. Do not
   edit immutable checkpoints or provenance to bypass checks silently.
5. Put the exact SpinQuant source root and `safe-quant/src` on `PYTHONPATH` for
   SpinQuant loading. QuaRot requires its exact source checkout; FlatQuant needs
   its pinned deployment source plus a freshly built compatible extension.
6. Re-prepare benchmark inputs from their pinned sources; ignored `outputs/`
   catalogs are not present after cloning. Preserve prompts, splits, generation
   limits, batch settings, and judge definitions for matched comparisons.
7. Run GPU smoke tests only inside a Slurm allocation, never on a Delta login
   node. The user submits any Slurm jobs. The handoff itself must not launch
   quantization or model inference.

The current model loaders are inference-oriented and return evaluation-mode
models. End-to-end gradients, quantization-aware parameter updates, optimizer
state, teacher/student placement, and on-policy distillation are not validated.
FlatQuant's packed deployment weights in particular must not be advertised as
directly trainable parameters. Keep all refinement outputs separate from these
immutable baseline artifacts.

## Publication status

Remote uploaded HF revisions and verification results are recorded in
`sharing/PUBLICATION.json` only after they actually succeed. A local Git commit
alone does not imply a successful GitHub push. The initial code commit uses the
verified public profile name `Will Liu` and GitHub noreply address
`127075923+Will65536@users.noreply.github.com` as repository-local identity;
no global Git identity is changed.
