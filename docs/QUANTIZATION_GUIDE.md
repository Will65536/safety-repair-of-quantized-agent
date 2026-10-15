# Quantization workspace: Llama-3-8B-Instruct

Last updated: 2026-10-15.

This is the living record of checkpoint preparation for safe-quant. Update it
after setup changes, bug fixes, GPU tests, and completed quantization runs.
Keep commands, observed results, and proposals distinguishable.
This shared guide is the maintained setup documentation for all quantizers;
do not create duplicate local setup READMEs. Keep upstream project READMEs.

## Scope and decisions

- Exact base model: `meta-llama/Meta-Llama-3-8B-Instruct`, not the base model or
  Llama 3.1. Reuse the existing local BF16 weights and original tokenizer.
- Target conditions: SpinQuant W4A4KV4, FlatQuant W4A4KV4, OSTQuant W4A4KV4,
  and QuaRot W4A4KV4, alongside the BF16 reference.
- DuQuant was removed from the plan. W4A4KV16 is no longer the target.
- Quantization and method environments live outside safe-quant. That repository
  consumes saved, immutable artifacts for evaluation.
- GPU work must run inside a user-requested Slurm allocation. Never run model
  inference or GPU experiments on login nodes. The user submits all jobs.
- Preserve identical downstream evaluation settings across conditions. Record
  method-specific calibration and quantization choices explicitly.
- Keep per-example predictions separate from aggregate metrics. Any later
  refinement produces a separate artifact rather than overwriting a baseline.
- A fake-quantized checkpoint reproduces low-bit numerical behavior while
  potentially storing floating-point tensors. Packed INT4 storage is a separate
  property; do not infer it from the W4A4KV4 label or checkpoint filename.

## Current status

| Method | Artifact / implementation status | Next step |
|---|---|---|
| BF16 reference | Existing unquantized Llama-3-8B-Instruct baseline; original evaluation results preserved; AdvBench generation/judge launchers now support BF16 | User runs the matched batch-4 AdvBench smoke, full generation and classifier judgment |
| SpinQuant W4A4KV4 | Existing `R.bin` and `qmodel.pt`; already integrated into safe-quant | Preserve and reuse the existing condition |
| FlatQuant W4A4KV4 | Full PPL/general/FinGuard/WildGuard/HarmBench/StrongREJECT and AdvBench generation+string matching completed; the old interrupted combined run is preserved separately | Finish/check XSTest separately; run the new AdvBench HarmBench-classifier judge; WildGuard score-shift investigation remains open |
| OSTQuant W4A4KV4 | Official source cloned; isolated environment installed; CPU/import checks passed; no checkpoint produced | Await batch environment-check results, then model smoke test |
| QuaRot W4A4KV4 | Paper-style run completed: WikiText-2 PPL 10.027104; immutable checkpoint saved; inference-only safe-quant loader added | User-run saved-state generation/EOS/fresh-process checks before benchmark runs |

No OSTQuant optimization, GPTQ, model inference, or GPU environment check has
been executed by the assistant. The user supplied passing QuaRot GPU-helper
and FlatQuant model-smoke results from gpua056; OSTQuant results remain pending.

## How safe-quant works: entry point and handoff

The maintained evaluation overview is the first section of
[/work/nvme/bgxm/wliu12/safe-quant/README.md](/work/nvme/bgxm/wliu12/safe-quant/README.md).
It describes the source layout, model dispatch, benchmark sizes, current batch
sizes, environments, output formats, commands and restart rules. This shared
guide remains the quantizer/checkpoint preparation record; do not put
quantization inside the evaluation repository or create per-quantizer setup
README duplicates.

The execution stages are:

1. Load an immutable external checkpoint with its model YAML/native runtime.
2. Select a frozen benchmark YAML and the same source/prompt file for every model.
3. Run CPU preparation and user-run GPU smoke checks before full evaluation.
4. Direct scoring benchmarks produce metrics immediately; behavioral generation
   saves raw responses first, then a separate judge produces the final score.
5. Compare matched examples/settings and judge provenance, retaining raw
   responses, individual judgments and aggregate metrics separately.

Current runtime boundaries: BF16 and SpinQuant use the existing Transformers
4.44.2 environment; BF16 uses the plain unquantized HF loader, not SpinQuant
wrappers. FlatQuant uses its native FP16/FlashAttention-2 deployment in the
separate Transformers 4.45.0 environment. The same chat/decoding/judge settings
do not make those floating-point/runtime paths identical. FlatQuant prefill
attention must not be described as an INT4-cache readback just because its
native runtime allocated a packed cache.

Current FlatQuant batches: general/PPL=1, FinGuard=4, WildGuard=2,
HarmBench/XSTest/StrongREJECT generation=1. Matched AdvBench runs use batch 4
via the explicit final launcher argument; the shared generation default is
still 1, and larger batches remain enabled only for AdvBench.

Judging is benchmark-specific:

- HarmBench: separate local GPU classifier after generation.
- XSTest: FlatQuant launcher automatically runs the CPU string matcher;
  safe and unsafe-contrast refusal rates remain separate.
- StrongREJECT: separate pinned paid API rubric; no local GPU is needed.
- AdvBench: automatic CPU string-matching non-refusal proxy, plus separate
  optional local GPU HarmBench-classifier ASR. These are distinct metrics.

User GPU execution must use Slurm, normally one model process per GPU. The
assistant never submits jobs or runs inference on login nodes. CPU-only data
and output inspection/tests are permitted without a GPU. Keep API keys out of
files/chat. Existing launchers activate their own prepared environment; do not
apply old BF16/SpinQuant installation snippets to FlatQuant or judge environments.

Use fresh output tags. If a run stopped, first verify its job is inactive,
then archive only that exact unfinished run if reusing its name is necessary.
No current runner resumes partial generation/judgment automatically. Preserve
partial evidence and completed baselines; restarting paid judging may repeat
API charges. A `running` manifest alone is not proof of a live job, and a
completed generation manifest is not proof that judging finished.

### Verified evaluation snapshot, 2026-10-15

These are observed runs, not strictly runtime-controlled quantization-only
conclusions. See each immutable run manifest for the complete settings.

| FlatQuant result | Observed full score / condition |
|---|---|
| WikiText-2 | PPL 9.011714; 141 blocks; batch 1 |
| General tasks | ARC-E 81.36%, ARC-C 50.09%, PIQA 77.15%, WinoGrande 71.51%; mean raw accuracy 70.02%; batch 1 |
| FinGuard | Accuracy 78.63%, attack F1 73.00%; batch 4 |
| WildGuard batch 4 | Accuracy 55.33%, harmful F1 66.28%; retained condition |
| WildGuard batch 2 | Accuracy 54.91%, harmful F1 66.14%; current matched-batch condition |
| HarmBench DirectRequest | HarmBench-classifier ASR 8.81% (14/159) |
| StrongREJECT full | Rubric mean 0.0047923 on 313 responses; continuous [0,1], not ASR |
| AdvBench DirectRequest | Original string-match non-refusal proxy 1.92% (10/520), batch 4; new classifier score pending |

SpinQuant W4A4KV4's matched AdvBench batch-4 proxy is 3.46% (18/520).
The new HarmBench-classifier scores for AdvBench and the requested BF16
AdvBench baseline are not yet present in the inspected outputs. Do not label
the existing proxy values as semantic classifier ASR. WildGuard's excess
false positives remain unexplained; moving FlatQuant from batch 4 to batch 2
did not close the gap, and no threshold/prompt/kernel was changed to improve it.
XSTest full completion was not verified in this documentation update.

Latest code verification: 160 CPU tests passed. Those tests stub native model
and classifier execution; real GPU validation status is recorded separately.

## Workspace locations

| Purpose | Absolute path |
|---|---|
| This guide | `/work/nvme/bgxm/wliu12/repos/QUANTIZATION_GUIDE.md` |
| Evaluation repository | `/work/nvme/bgxm/wliu12/safe-quant` |
| BF16 model and tokenizer | `/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/bf16` |
| Existing SpinQuant artifact | `/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/spinquant-w4a4kv4` |
| OSTQuant source | `/work/nvme/bgxm/wliu12/repos/OSTQuant` |
| OSTQuant environment | `/work/nvme/bgxm/wliu12/envs/ostquant` |
| OSTQuant managed Python | `/work/nvme/bgxm/wliu12/envs/.python` |
| OSTQuant setup files | `/work/nvme/bgxm/wliu12/repos/OSTQuant/local_setup` |
| OSTQuant test logs | `/work/nvme/bgxm/wliu12/repos/OSTQuant/local_setup/logs` |
| Proposed production artifact parent | `/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/ostquant-w4a4kv4` |
| QuaRot source | `/work/nvme/bgxm/wliu12/repos/QuaRot` |
| QuaRot environment | `/work/nvme/bgxm/wliu12/envs/quarot` |
| QuaRot setup files and logs | `/work/nvme/bgxm/wliu12/repos/QuaRot/local_setup` and its `logs` directory |
| FlatQuant source | `/work/nvme/bgxm/wliu12/repos/FlatQuant` |
| FlatQuant inference environment | `/work/nvme/bgxm/wliu12/envs/flatquant-eval` |
| Published FlatQuant artifact | `/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/flatquant-w4a4kv4` |
| FlatQuant setup, runtime, and logs | `/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup`, with `runtime` and `logs` subdirectories |

The production artifact path is proposed; it has not been populated. Keep
smoke artifacts separate and retain the managed Python directory on which the
virtual environment depends.

## OSTQuant environment: completed work

Official source: <https://github.com/BrotherHappy/OSTQuant>.
Pinned upstream commit: `ab64362da147291612d077accaab5d3ed7b508b6`.
Upstream algorithm source has not been modified; additions are in `local_setup`.

| Component | Installed version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.2.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| Transformers | 4.44.0 |
| Accelerate | 0.33.0 |
| Datasets | 2.17.1 |
| lm-eval | 0.4.4 |
| Geoopt | 0.5.0 |
| NumPy / SciPy | 1.26.4 / 1.10.1 |
| fast-hadamard-transform | Official 1.0.4.post1 CUDA 12.2 / torch 2.2 / CPython 3.10 wheel, C++11 ABI false |

Upstream `requirements.txt` is a formatted package listing, not a directly
installable requirements file. We created minimal direct pins in
`OSTQuant/local_setup/requirements.txt` and resolved all dependencies in
`OSTQuant/local_setup/requirements.lock.txt`.

The Hadamard wheel avoids compiling against the login node's CUDA 13.2 toolkit.
Its import passed; its numerical CUDA behavior still requires the GPU test.

To reinstall OSTQuant's dependencies into its existing environment:

```bash
cd /work/nvme/bgxm/wliu12/repos/OSTQuant
uv pip install --python /work/nvme/bgxm/wliu12/envs/ostquant/bin/python \
  --cache-dir /work/nvme/bgxm/wliu12/pip-cache/uv-ostquant \
  --index-url https://pypi.org/simple -r local_setup/requirements.lock.txt
```

The former `OSTQuant/local_setup/README.md` has been consolidated into this
guide and deleted. Its environment, validation, batch/interactive, and
subsequent-model instructions are preserved here.

Checks passed on 2026-10-15:

- Installed dependency consistency.
- OSTQuant entrypoint and compiled Hadamard extension imports with GPUs hidden.
- CLI argument parser initialization (`main.py --help`), without model loading.
- Python and batch-script syntax.
- Independent CPU Hadamard reference against an explicit Hadamard matrix.

The user reported an A100-SXM4-40GB with driver 595.71.05 and `nvidia-smi`
CUDA version 13.2. This is compatible at the driver/architecture level with
our CUDA 12.1 PyTorch build, which includes A100's `sm_80` architecture.
The CUDA version shown by `nvidia-smi` describes driver support and does not
need to equal `torch.version.cuda`. Actual kernel execution is still pending.

## How to test: sbatch (preferred)

The user now prefers a batch job because interactive allocation waits are long.
Submit from the login shell; no prior environment activation is needed because
the script activates OSTQuant itself:

```bash
sbatch /work/nvme/bgxm/wliu12/repos/OSTQuant/local_setup/check_env_4gpu.sbatch
```

The script requests one node, four A100 GPUs, 16 CPUs, 64 GB host memory, and
15 minutes with account `bgxm-delta-gpu` and partition `gpuA100x4`.
It runs the CPU import check, Hadamard checks, and four-rank NCCL check described
below. It does not quantize or load Llama. Fifteen minutes is the requested
runtime limit for this environment test; the job exits when its checks finish.
Queue waiting time is separate from the requested runtime limit. Batch mode
does not guarantee a shorter wait for the same GPU resources.

Slurm prints the submitted job ID. Replace `12345678` below with that ID:

```bash
quant_env_job=12345678
squeue -j "$quant_env_job"
sacct -j "$quant_env_job" --format=JobID,State,Elapsed,ExitCode
tail -n 80 "/work/nvme/bgxm/wliu12/repos/OSTQuant/local_setup/logs/env4-${quant_env_job}.out"
tail -n 80 "/work/nvme/bgxm/wliu12/repos/OSTQuant/local_setup/logs/env4-${quant_env_job}.err"
```

The log files may not exist until the job starts. Success requires the passed
CPU report, four passed GPU rank reports, and job state `COMPLETED` with
`ExitCode` `0:0`. Do not submit a duplicate interactive or batch test while this
job is pending or running. The assistant does not submit jobs.

## How to test: interactive Slurm mode (optional)

### 1. Obtain the allocation

Request one node with four A100 GPUs through Slurm using your usual interactive
allocation command. Once assigned, use the compute-node shell belonging to
that allocation. The earlier `gpua048` output showed only one visible GPU;
that was not sufficient for the four-rank test below.

Inside the allocation, confirm that `SLURM_JOB_ID` is set and four GPUs are
visible before launching four processes:

```bash
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-unset}"
nvidia-smi -L
```

### 2. Activate OSTQuant and run the CPU/import check

```bash
source /work/nvme/bgxm/wliu12/envs/ostquant/bin/activate
cd /work/nvme/bgxm/wliu12/repos/OSTQuant
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
unset PYTHONPATH

mkdir -p local_setup/logs
quant_check_run="interactive-env4-$(date +%Y%m%d-%H%M%S)"
set -o pipefail
python -B local_setup/check_cpu.py 2>&1 | tee "local_setup/logs/${quant_check_run}-cpu.log"
```

Expected: a JSON report with `"ostquant_import": "passed"`,
`"torch_cuda": "12.1"`, and `"nccl_built": true`, with exit status zero.
Stop and inspect the error if this command fails. This check hides GPUs inside
its own Python process and does not change the shell's GPU visibility.
The CPU/import check can also run on a login node because it loads no model
and executes no GPU work.

### 3. Run the four-GPU environment check

In the same interactive Slurm shell:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  local_setup/check_gpu.py 2>&1 | tee "local_setup/logs/${quant_check_run}-gpu.log"
```

This script requires `SLURM_JOB_ID` and exactly four torchrun ranks. Each rank
uses one GPU. It verifies:

- BF16 support.
- FP32 and BF16 Hadamard transforms at widths 128 and 4096 against the CPU
  reference. FP32 tolerances are `atol=rtol=1e-5`; BF16 tolerances are
  `atol=rtol=0.02`.
- Four-rank NCCL all-reduce: rank values 1, 2, 3, 4 must sum to 10.

Expected: four JSON reports for ranks 0 through 3, each containing
`"hadamard_fp32_bf16": "passed"` and `"nccl": "passed"`, followed by a
successful command exit. GPU names and per-device memory are also reported.

This is an environment test. It does not load Llama, download calibration data,
run optimization, run GPTQ, or prove that the full model fits in memory.
An environment pass does not yet establish a working quantized model.

Record the Slurm job ID, GPU types, test log paths, and outcome in the run
record below. Preserve the complete traceback when a check fails.

## How to run quantization: pending model validation

There is no validated production quantization launcher yet. Do not treat the
environment-check commands as model-quantization commands.

After the environment check passes, the next stages are:

1. Verify loading of the local Instruct model and original tokenizer, forward
   outputs with quantization disabled, transformation fusion, grouped-query
   attention, and cached generation in a Slurm model smoke test.
2. Check a few optimization steps for finite loss/gradients and measure GPU
   memory. Establish correct parameter placement and gradient synchronization
   across four GPUs.
3. Freeze the production recipe and add the exact launch commands
   here. Then optimize transformations and perform final GPTQ weight
   quantization in the external workspace.
4. Save learned transformations, complete quantized state, tokenizer/config,
   resolved settings, software/source versions, and checksums.
5. Reload in a fresh process without calibration or optimization. Compare
   pre-save/reloaded outputs, inspect active W/A/K/V settings, and verify cached
   generation. Only then integrate an inference-only safe-quant loader.

### Proposed recipe, not yet frozen or executed

- BF16 base weights/runtime, with W4A4KV4 fake quantization enabled at inference.
- Official W4A4KV4 transformation choices and KL-Top loss; 100 optimizer steps.
- Unquantized weights during transformation optimization, 4-bit activations and
  K/V; final 4-bit GPTQ using 128 training samples.
- WikiText-2 training data, 2048-token sequences, seed 0; record sampling and
  immutable dataset revision when the recipe is finalized.
- Use validation data for development checks; reserve test data for the final
  diagnostic. This is a proposed change to upstream progress evaluation.
- Existing SpinQuant optimization launcher uses 512-token sequences. Retaining
  OSTQuant's 2048-token recipe creates a method-specific calibration difference
  that must be reported; it is not a calibration-controlled comparison.
- Four GPUs are the target. Upstream README/W4A16KV16 uses four processes;
  upstream `scripts/w4a4kv4.sh` uses eight processes and microbatch 4.
  To preserve its effective batch of 32, four GPUs could use microbatch 1 and
  eight gradient-accumulation steps, subject to model-memory validation.

No parameter changes, reduced sequence lengths, or altered optimization budgets
should be adopted silently to work around memory limits.

## QuaRot setup: completed environment and CPU compatibility checks

Official source: <https://github.com/spcl/QuaRot>.
Pinned upstream commit: `5008669b08c1f11f9b64d52d16fddd47ca754c5a`.

| Component | Installed version |
|---|---|
| Python | 3.10.20 |
| PyTorch / CUDA runtime | 2.2.1+cu121 / 12.1 |
| Transformers / Tokenizers | 4.44.2 / 0.19.1 |
| Accelerate / Datasets | 0.34.2 / 2.20.0 |
| NumPy / SciPy | 1.26.4 / 1.10.1 |
| fast-hadamard-transform | Official 1.0.4.post1 CUDA 12.2 / torch 2.2 / CPython 3.10 wheel, C++11 ABI false |

Direct pins are in `QuaRot/local_setup/requirements.txt`; resolved dependencies
are in `QuaRot/local_setup/requirements.lock.txt`. The environment is separate
from OSTQuant. Cached wheels were reused from `pip-cache/uv-ostquant`; sharing
that download cache does not share or modify the installed environments.

The initial environment includes the research quantization dependencies. Optional
WandB and native QuaRot `--lm_eval` dependencies have not been installed or
validated. Do not use those flags yet. Downstream scientific evaluation will
use safe-quant's frozen protocols rather than QuaRot's old harness integration.
Packed QuaRot deployment kernels have not been built.

### Compatibility changes and checks

Current upstream source changes are limited to `fake_quant/utils.py`,
`fake_quant/eval_utils.py`, and device placement in `fake_quant/gptq_utils.py`:

- Added the exact Instruct model ID and permitted existing local checkpoint
  directories in the parser.
- Fixed local-model output paths. Previously, an absolute `--model` path could
  redirect experiment logs into the immutable BF16 directory. Local model
  names now put logs under `QuaRot/fake_quant/experiments/<basename>/<run>`.
- The evaluator now detects the actual model class rather than looking for
  `meta` in the model path, so the local BF16 directory works. It also saves
  native aggregate perplexity and per-sequence losses separately when saving
  a checkpoint. No loss formula, sequence truncation, or aggregation changed.
- GPTQ and Llama evaluation now move the model-level `rotary_emb` to the
  capture device before collecting inputs, then back to CPU afterward. This
  module is outside decoder layers in Transformers 4.44. Without these moves,
  native sequential offloading left its frequency buffer on CPU while token
  positions were on CUDA. No dtype casts or numerical formulas were added.

The user explicitly chose **original upstream QuaRot normalization with the
BF16 base model**. `fake_quant/model_utils.py` has been restored to upstream;
its RMSNorm precision is not patched. Do not silently switch to FP16 or reapply
the BF16 upcast. Parser/local-path compatibility changes remain in place.

Verified on 2026-10-15 with GPUs hidden and Hugging Face offline mode:

- Dependency consistency and official research entrypoint/compiled Hadamard imports.
- Exact local model configuration: BF16 Llama, hidden size 4096, 32 query heads,
  8 KV heads, and vocabulary size 128256.
- Local tokenizer loading with both `use_fast=False` and `use_fast=True`, matching
  encodings, original chat template, BOS/EOS, and scoring tokens ` Yes`=7566 and
  ` No`=2360.
- Parsing the local checkpoint path without creating log directories or files;
  verified resulting output path stays inside QuaRot's experiment directory.
- Eager/SDPA attention signatures and direct rotary-call interfaces needed by
  QuaRot's wrapper. These are structural checks, not numerical model validation.

Those initial CPU checks loaded no model weights and ran no model inference.
The user subsequently passed the GPU Hadamard helper test on an A100-SXM4-40GB.
On 2026-10-15 a BF16-upcast compatibility patch was initially tested, but then
reverted at the user's explicit request to retain upstream numerical behavior.
The temporary variant matched installed Llama RMSNorm in nine CPU cases. The
current `local_setup/check_rmsnorm.py` instead checks the original upstream
expression exactly and reports differences from Llama without requiring them
to be zero. Tested dtypes are FP32/FP16/BF16 at scales 0.01, 1, and 1000.
The unpatched BF16 path showed maximum differences from Llama of 0.03125,
0.03125, and 0.015625 in these toy cases; not model accuracy measurements.
Some reduction kernels may use FP32 internally, but the upstream BF16 path
still has BF16 intermediate tensors instead of Llama's explicit FP32 upcast.
The user then ran the BF16 model smoke on gpua056, job 22913901. Loading,
fusion, online transforms, and within-stage mask/cache checks executed, but
rotated-vs-original prompt 0 exceeded the custom gates (NRMSE 0.12121, mean
KL 0.07777). The full run took 130.26 seconds with 15.00 GiB peak allocated
GPU memory and 29.17 GiB host RSS. These are not GPTQ resource measurements.
The user explicitly stopped further custom parity investigation and chose
the official quantization workflow. Do not relabel the failed test as passed,
modify original normalization, or silently loosen its thresholds.

### Run the QuaRot CPU compatibility check

This is safe on a login node: it loads only configuration/tokenizer data and
does not run model inference or GPU operations.

```bash
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
cd /work/nvme/bgxm/wliu12/repos/QuaRot
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
python -B local_setup/check_cpu.py
```

Expected: passed import/config/tokenizer/CLI reports, direct rotary-call interfaces
for `LlamaAttention` and `LlamaSdpaAttention`, and exit status zero. The JSON
also lists what has not been tested.

### Submit the QuaRot single-GPU environment check

The user can submit independently of the pending OSTQuant check:

```bash
sbatch /work/nvme/bgxm/wliu12/repos/QuaRot/local_setup/check_env_1gpu.sbatch
```

This requests one A100 GPU, 8 CPUs, 64 GB host memory, and 15 minutes, with
account `bgxm-delta-gpu` and partition `gpuA100x4`. The partition name does
not mean the job requests all four GPUs: `--gpus-per-node=1` is explicit.
The script activates only QuaRot's environment and runs the CPU check followed
by QuaRot's own CUDA Hadamard helper compared to its CPU implementation for
FP32/BF16 at widths 128, 4096, and 14336 (factor 28 for Llama 3's MLP).
Tolerances are `atol=rtol=1e-5` for FP32 and `0.02` for BF16.
It does not load or quantize Llama and does not test multi-GPU communication.

Logs: `QuaRot/local_setup/logs/env1-JOBID.out` and `.err`. Success requires
the passed CPU check, `"quarot_hadamard_fp32_bf16": "passed"`, and Slurm
state `COMPLETED` with exit code `0:0` for batch submissions. The user ran the
equivalent interactively on gpua056 and supplied passing output. Maximum FP32
error was 1.3113e-6; maximum BF16 error was 0.013823 across the tested widths.
The helper test did not load or run the model.

Inside an existing user-requested Slurm GPU allocation, the equivalent is:

```bash
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
cd /work/nvme/bgxm/wliu12/repos/QuaRot
python -B local_setup/check_cpu.py
python -B local_setup/check_gpu.py
```

To reinstall dependencies into QuaRot's existing environment:

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
uv pip install --python /work/nvme/bgxm/wliu12/envs/quarot/bin/python \
  --cache-dir /work/nvme/bgxm/wliu12/pip-cache/uv-ostquant \
  --index-url https://pypi.org/simple -r local_setup/requirements.lock.txt
```

## QuaRot model and quantization plan (remaining work)

Target: exact local Meta-Llama-3-8B-Instruct weights and tokenizer, with
W4A4KV4 fake-quantized behavior and BF16 underlying model weights/runtime
where supported. Keep inference settings matched to the BF16 evaluation
condition and document all runtime precision exceptions.

### Locations

- Source: `/work/nvme/bgxm/wliu12/repos/QuaRot`.
- Environment: `/work/nvme/bgxm/wliu12/envs/quarot`.
- Proposed artifact parent:
  `/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/quarot-w4a4kv4`.

The source/environment and artifact parent exist. Preserve the completed
code-default artifact and failed attempts. New paper-style runs use a distinct
`paperstyle-JOBID-UTC_TIMESTAMP` subdirectory. Reuse the BF16 input without
modifying it, and keep smoke outputs separate from production.

### 1. Environment and GPU-helper stage completed

The source/environment and CPU/tokenizer checks above are complete. Upstream
pins Transformers 4.38.0 and an older tokenizer stack; our isolated 4.44.2 setup
has been verified against the exact local tokenizer. The user passed GPU
helper checks and chose to proceed with native GPTQ/perplexity after stopping
the custom model-parity investigation.

### 2. Make minimal Llama 3 Instruct compatibility changes

- Exact Instruct/local-path parsing and tokenizer identity checks are complete.
- Check grouped-query attention, rotary interfaces, attention implementation,
  masks, and BF16 RMSNorm behavior against the original model.
- The custom unquantized-model comparison is historical diagnostic evidence,
  not an official reproduction acceptance gate. Do not require another custom
  parity run before the user-approved native quantization workflow.
- Preserve official rotation locations, including online MLP-output,
  attention-output, and Q/K transforms. Do not substitute SpinQuant with
  learned rotations disabled and assume it is identical QuaRot.
- Fix interface problems with small documented patches. Any numerical change
  needs an explicit rationale and validation.

### 3. Archived custom model smoke: not part of the current workflow

**Stopped by explicit user decision.** The instructions and thresholds below
are retained only as historical context. Do not run this as a prerequisite
for quantization. Its failure is recorded above; the current command is under
"Native quantization launch" below. The model smoke was not QuaRot's official
reproduction procedure and its thresholds were custom diagnostic choices.

Propose one A100 40 GB and a 15-minute smoke allocation. The official GPTQ
implementation quantizes layers sequentially on one device; `--distribute`
is used for evaluation rather than distributing GPTQ across four GPUs.
The current code does not provide four-rank GPTQ simply by using torchrun.

The current smoke deliberately performs **no low-bit quantization, calibration,
GPTQ, or checkpoint export**. Reduced calibration/export is a later separate
test, after model compatibility passes. Do not infer GPTQ memory requirements
from this model smoke or present diagnostic gates as scientific equivalence.

In a fresh user-requested interactive allocation (one A100, 15 minutes):

```bash
srun --account=bgxm-delta-gpu --partition=gpuA100x4 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  --mem=64G --time=00:15:00 --pty bash -l
```

Once the compute-node shell opens, or inside an existing allocation with
sufficient time left:

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
unset PYTHONPATH
python -u -B local_setup/check_model.py
```

Do not nest another `srun` in the interactive shell. Optional batch alternative,
submitted only by the user:

```bash
sbatch /work/nvme/bgxm/wliu12/repos/QuaRot/local_setup/check_model_1gpu.sbatch
```

The script refuses to run without `SLURM_JOB_ID`. It loads only local BF16
weights with the same SDPA attention for every comparison. It avoids the
upstream loader's global disabling of PyTorch initializers. Only one full
model is kept at a time, moved to CPU during weight transformations. Fixed
seed 0 is reset immediately before rotation. TF32 is disabled as in upstream.

The test sequence is:

1. Verify original QuaRot RMSNorm matches its upstream expression on GPU and
   report its BF16 difference from installed Llama RMSNorm. A nonzero difference
   here is informational, not by itself a failure.
2. Capture original-model logits for two fixed neutral prompts.
3. Apply official norm fusion, including upstream embedding centering, then
   compare against the original. If this fails, stop before rotating weights.
4. Apply official randomized Hadamard weight rotations, online MLP/output
   rotations, and post-RoPE Q/K rotations. All activation/key/value quantizers
   remain at 16 bits; underlying model weights/cache remain BF16.
5. Compare transformed logits with the original, and test left-padding/masks
   and two teacher-forced cached decode steps at each stage. Check eight GQA
   KV heads, 128-dimensional heads, finite BF16 cache values, and cache length.

Default **diagnostic** gates: centered-logit normalized RMSE <= 0.03 and mean
KL(reference || test) <= 0.01. These are screening thresholds, not guarantees
of benchmark equivalence or promises of bit-identical BF16 rotation. Report
absolute errors, maximum position KL, and top-1 agreement as well; top-1 is
reported rather than required because near-ties may change under rounding.
If any gate fails, inspect the stage and errors before changing tolerances.
The full-model gates remain unchanged after restoring upstream RMSNorm. A
failure may reflect upstream normalization and fusion behavior before 4-bit
quantization; it must not be dismissed by silently loosening the gates.
`--max-nrmse` and `--max-mean-kl` overrides are recorded in the report and should
not be loosened simply to obtain a pass. This does not freeze evaluation settings.

Outputs use a unique directory:
`QuaRot/local_setup/logs/model-JOBID-UTC_TIMESTAMP/`.

- `report.json`: stage status, comparisons, versions, source hashes, settings,
  elapsed time, peak allocated/reserved GPU memory, and peak host RSS.
- `predictions.json`: per-example scoring logits and top-token IDs, separate
  from aggregate diagnostics.
- `inputs.json`, `reference_logits.pt`: exact prompts/token IDs and reference
  tensors used for reproducibility. These are not a quantized checkpoint.
- `source_changes.patch`: tracked local patches at execution time.

Failures save a traceback and partial report. If Slurm kills the job before
exception handling, the most recent saved report/progress line identifies the
last reached stage; a `running` report is not success. The user supplied the
failed model-test result described above; the assistant ran only CPU checks.

### 4. Paper-style quantization recipe and native launch

The first successful run used the repository README example/defaults. The
user then explicitly approved switching to the **paper's experimental settings**
after learning that those defaults are not the published recipe. The current
launcher uses the settings below through native QuaRot CLI flags, without
changing the quantization implementation or normalization.

Source: [QuaRot, Section 5 and Appendix A.7](https://proceedings.neurips.cc/paper_files/paper/2024/file/b5b939436789f76f08b9d0da5e81af7c-Paper-Conference.pdf).
These are paper-style settings on **BF16 Meta-Llama-3-8B-Instruct**, not a claim
to reproduce the paper's base Llama-3-8B/FP16 numbers. The bit label alone also
does not mean the granularity/settings match SpinQuant or FlatQuant.

| Setting | Current paper-style recipe |
|---|---|
| Rotation | Official randomized Hadamard recipe, global seed 0, no learned-rotation training; FP32 online MLP/output Hadamard (`--fp32_had`) |
| Weight quantization | GPTQ INT4, clipping enabled, symmetric per-output-channel weights (`w_groupsize=-1`) |
| Activation quantization | 4-bit dynamic, symmetric, token-wise (`a_groupsize=-1`), clipping ratio 0.9; output head input stays 16-bit as upstream |
| KV quantization | Asymmetric K4 after RoPE and V4 at `v_proj` output, both groups of 128, clipping ratios 0.95 |
| Calibration | WikiText-2 training split, 128 samples, 2048 tokens per sample, seed 0 |
| Mixed precision exception | Keep `int8_down_proj` disabled; record embedding, norm, residual, and output-head precision |
| GPTQ settings | Native damping 0.01, no activation ordering, no RTN; upstream block size 128 |
| Evaluation | Native WikiText-2 test perplexity, sequence length 2048, batch size 1; no lm-eval/WandB |

Head-sized K grouping also avoids the unsupported native token-wise reshape
for Llama 3 GQA. V grouping now changes from -1 to 128 to follow the paper.
Symmetric weights, native GPTQ and its weight-clipping search remain unchanged.
No mixed INT8 down-projection is enabled. W/A group sizes stay -1; this is not
the paper's weight/activation-grouped "QuaRot-128G" variant.

The current native CLI differences from the preserved code-default run are:

```text
--a_clip_ratio 0.9
--k_asym --v_asym
--k_groupsize 128 --v_groupsize 128
--k_clip_ratio 0.95 --v_clip_ratio 0.95
--fp32_had
```

This needs a fresh quantization run because online Hadamard precision affects
the activations used by GPTQ. Do not relabel or overwrite the previous model.
Calibration split/windows, seed, sample count, base BF16 weights, tokenizer,
upstream RMSNorm, GPTQ damping/ordering, and PPL protocol remain unchanged.

The base checkpoint stays BF16; upstream RMSNorm, rotations, quantizers, and
GPTQ mathematics remain unmodified. GPTQ includes only the documented shared
RoPE device-placement additions. W4A4KV4 is **fake quantization**: low-bit numerical
values can be stored in BF16 tensors, not packed INT4 weights/cache. This is
the native research artifact, not a claim of optimized deployment kernels.
Safety benchmark examples are never calibration data. Full downstream
evaluation remains in safe-quant after artifact loading is validated.

#### Data preparation: completed on CPU

`local_setup/prepare_wikitext.py` calls the native data loader, without changing
its tokenizer, `\n\n` concatenation, seed, random-window selection, or sequence
length. It downloads WikiText-2 to the dedicated `local_setup/hf-cache` and
records exact calibration/evaluation tensors and hashes in:

- `local_setup/data/wikitext2_tokens.pt`.
- `local_setup/data/wikitext2_manifest.json`: dataset fingerprints, resolved
  download URLs/checksums/cache identity, 128 sampled-token hashes, evaluation
  token hash, and source/package identities.
- `local_setup/logs/prepare-wikitext.log` and `prepare-wikitext-offline.log`.

The current test encoding contains 289,077 tokens, of which the native
evaluator uses 141 complete 2048-token sequences and truncates the tail.
The native GPU workflow still calls `data_utils.get_loaders` itself. It does
not substitute a custom calibration sampler or read held-out evaluation data
for GPTQ. The wrapper checks cache reproducibility before GPU work and then
runs entirely offline. A different dataset/tokenization fingerprint is a
failure, not permission to overwrite the preparation record.

To restore/verify the cache (CPU-only; no model loading):

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
python -B local_setup/prepare_wikitext.py
python -B local_setup/prepare_wikitext.py --offline-check
```

#### Paper-style quantization launch

The user requests/submits the GPU resources. Full GPTQ plus perplexity is not
a 15-minute environment test. Start with **one A100, eight CPUs, 64 GB host
RAM, two hours**. The completed code-default run took 19.44 minutes. The new
paper-style runtime may differ; two hours remains a conservative scheduling
budget, not a completion guarantee.

Interactive allocation, requested from the login node:

```bash
srun --account=bgxm-delta-gpu --partition=gpuA100x4 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  --mem=64G --time=02:00:00 --pty bash -l
```

Once the compute-node shell opens, run directly without a nested `srun`:

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
unset PYTHONPATH
set -o pipefail
python -u -B local_setup/run_official.py 2>&1 | tee "local_setup/logs/quantize-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%S).log"
```

The launcher clears inherited offline/cache settings in favor of the dedicated
prepared cache, enforces offline loading, and calls **`fake_quant/main.py`'s
`main()` directly** with native CLI flags. It does not call custom parity
tests, replace GPTQ, or alter normalization. It verifies core numerical source
files match upstream before starting, allowing only the two reviewed shared
RoPE device-placement blocks in GPTQ. Removing those blocks must recover the
upstream file byte-for-byte; other GPTQ edits are rejected. Local changes are
parser/path compatibility, RoPE placement, result persistence, and provenance.

#### Restart after the shared-RoPE device error

The first native attempt, job **22914500** on gpua012, failed during calibration
input collection with `cpu and cuda:0` in `LlamaRotaryEmbedding.forward`.
Rotation completed, but no GPTQ weight quantization/checkpoint saving completed.
The preserved failed run is:
`quarot-w4a4kv4/official-22914500-20261015T071540842799Z`.
Its manifest records `status=failed`, `checkpoint_exists=false`, and 224.58
seconds elapsed. Do not delete it or attempt to resume from a nonexistent
checkpoint. WikiText-2 cache and base weights were unaffected.

The compatibility patch moves **only shared RoPE** alongside the modules native
QuaRot already offloads. It retains BF16, upstream RMSNorm, rotations, all bit
widths/groups/clipping, calibration, and the native PPL computation. Both GPTQ
and evaluation are patched to avoid the same failure at the later PPL stage.

`local_setup/check_rope_placement.py` passed CPU-only source-integrity and
placement-branch tests with recording stubs, including models without shared
RoPE. Syntax checks passed. These are not GPU verification or model inference.
No new GPU job has been run or submitted by the assistant.

Inside an active allocation with sufficient remaining time, restart using:

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
unset PYTHONPATH
python -u -B local_setup/run_official.py
```

If the previous allocation ended, request a fresh one-A100/two-hour allocation
using the command above first. No reinstall or data re-download is needed.
Each restart gets a new timestamped output directory and reruns from the
beginning, preserving the failed attempt. Success past input collection should
show `Layer 0:` and GPTQ projection names; that progress is not final completion.

Batch alternative (user submission only):

```bash
sbatch /work/nvme/bgxm/wliu12/repos/QuaRot/local_setup/quantize_official_1gpu.sbatch
```

Batch logs: `local_setup/logs/quantize-JOBID.out` / `.err`. To inspect the native
recipe without running anything: `python -B local_setup/run_official.py --show-command`.

Each run creates a unique directory under
`/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/quarot-w4a4kv4/paperstyle-JOBID-UTC_TIMESTAMP/`.
Existing artifacts are never overwritten. Outputs include:

- `qmodel.pt`: the native saved weight state and weight quantizers.
- `run_manifest.json`: exact native arguments, settings, source/package identity,
  dataset manifest, status, timing/memory, and checksum after native completion.
  New runs explicitly record `recipe=paper-style`, asymmetric KV, clipping
  ratios, FP32 online Hadamard, and the BF16/Instruct reproduction limitation.
- Copied base config/tokenizer, dependency lock, calibration manifest, and
  `source_changes.patch` for interpretation/reproducibility.
- `ppl_metrics.json`: native aggregate perplexity.
- `ppl_per_sequence.pt`: native per-sequence mean NLLs, separate from aggregates.

The native parser also creates its usual log under
`fake_quant/experiments/bf16/<run_name>/`. If evaluation fails after weight
saving, preserve the checkpoint and manifest; do not assume quantization must
be repeated. A `running` manifest after Slurm termination is not completion.
The wrapper marks `checkpoint_usable_for_safe_quant=False` until the separate
load-only adapter/reload check is completed. Native `qmodel.pt` alone does not
encode activation/KV runtime configuration. Do not pass it to AutoModel and
assume it is a standalone Hugging Face model.

#### Preserved successful code-default run

The user completed job 22914500 on gpua012 after the device-placement fix:

- Artifact directory: `quarot-w4a4kv4/official-22914500-20261015T072741204302Z`.
- Status: `native_quantization_and_ppl_completed`.
- Native WikiText-2 test perplexity: **10.464049339294434**, not 10.64.
- 141 sequences of length 2048, evaluation batch size 1.
- Elapsed: 1166.34 seconds (19.44 minutes); peak allocated GPU memory 8.89 GiB;
  peak host RSS 33.18 GiB.
- `qmodel.pt`: 16,072,004,244 bytes; saved SHA256
  `ccb8fd0c63081113ce2252c8caaebd47cd60b7573fbc3a0a79fefb721217b617`.
- Native log:
  `QuaRot/fake_quant/experiments/bf16/official-22914500-20261015T072741204302Z/official-22914500-20261015T072741204302Z.log`.

This was **code-default**, not paper-style: symmetric KV, K groups 128,
V groups -1, activation/KV clipping 1.0, and BF16 online MLP/output Hadamard.
Its saved manifest and checkpoint are preserved unchanged. It is not directly
comparable to the paper's Table 11 base Llama-3-8B numbers (8.16 for QuaRot,
7.36 for QuaRot-128G). No published exact Instruct/BF16 reference result was
verified. Fresh-process loading still needs validation for evaluation use.

### 5. Export a complete, reproducible research artifact

Official `fake_quant/main.py` supports `save_qmodel_path`/`load_qmodel_path`,
but saves weights before configuring activation/V quantization and adding the
K wrapper. A weight-state file alone does not encode the whole runtime.

Preserve quantized model state, weight-quantizer metadata, exact rotation
identity/configuration, tokenizer/config, quantization settings, source commit,
dependency lock, base-weight identity, calibration provenance, and checksums.
Add a load-only path that reconstructs wrappers and loads compatible state
strictly without GPTQ, RTN, calibration, or numerical re-rotation of weights
already stored in rotated form. This is a custom research artifact, not a
claim of standalone standard-Hugging-Face or packed INT4 compatibility.

### 6. Inference-only integration: implemented, GPU validation pending

The paper-style artifact inspected after the completed user run is:

`/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/quarot-w4a4kv4/paperstyle-22914989-20261015T081845171928Z`.

- Native PPL: 10.027104377746582; 141 sequences of 2048 tokens, batch 1.
- Native run: 1651.24 seconds; peak allocated GPU 7.961 GiB during the
  layerwise native workflow. This is not a full-model inference memory estimate.
- `qmodel.pt`: 16,072,004,244 bytes; SHA256
  `fa0c5ce028544bed329941ef91f44b586a7bf0ad6ad003313055a7f27808cd42`.
- This is the Instruct/BF16 condition, not an exact reproduction of the paper's
  base-model results. The earlier code-default artifact remains unchanged.

The new safe-quant loader verifies the completed manifest, checkpoint hash,
and pinned native inference source. It builds an empty model skeleton,
replaces norm *modules* with unchanged upstream RMSN, recreates the native
activation wrappers and online-transform metadata, and loads saved weights
strictly. It then configures activation/V quantizers and post-RoPE K wrappers,
matching the native save boundary. It does not import GPTQ/calibration code,
load base weights, fuse weights, or apply offline rotations again.

Expected runtime counts: 224 four-bit weight-quantizer records, 224 active
input quantizers, 32 V quantizers, 32 K quantizers; LM head input remains 16-bit.
BF16 fake-quantized storage is not a packed INT4 checkpoint/cache.

Fresh-process checks below compare two independent loads of this saved state,
not pre-save native logits (which were not captured), and not original-model
versus rotated-model parity. They test loading, finite outputs, active flags,
native EOS handling and repeatability, without the additional custom parity
investigation the user stopped.

## FlatQuant: published-checkpoint inference setup

This is an inference-only setup, not a new FlatQuant quantization run. No
calibration dataset, transformation training, GPTQ/RTN, or export is performed.
Do not run upstream `main.py` for this workflow. Activation and KV quantization
still happen dynamically during inference, as required by W4A4KV4.

### Pinned inputs and environment

- Official source: <https://github.com/ruikangliu/FlatQuant>.
- Source commit: `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`.
- CUTLASS build dependency: `ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e`.
- Published checkpoint:
  <https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant>.
- Checkpoint revision: `65380056097d262d882e427359c9c8be2d63abf7`.
- Snapshot: 46 files, 5,638,181,521 bytes excluding Hugging Face download metadata.
  SHA256/size manifest: `FlatQuant/local_setup/checkpoint_manifest.json`.
  Bundled `.so` and `.pyc` files are deliberately excluded.
- Published config: W4/A4/KV4, symmetric W/A and asymmetric KV. This does not
  establish that its quantization granularity matches the other methods.

| Component | Installed version |
|---|---|
| Python | 3.10.20, same managed interpreter directory as OSTQuant/QuaRot |
| PyTorch / CUDA runtime | 2.7.1+cu128 / 12.8, C++11 ABI true |
| Transformers / tokenizers | 4.45.0 / 0.20.3 |
| Accelerate / Hugging Face Hub | 0.34.2 / 0.25.2 |
| Triton | 3.3.1 |
| NumPy / safetensors | 1.26.4 / 0.4.5 |
| FlashAttention | Official 2.8.3.post2 CUDA 12 / torch 2.7 / ABI true / CPython 3.10 wheel |
| fast-hadamard-transform | Official 1.1.0 CUDA 12 / torch 2.7 / ABI true / CPython 3.10 wheel |
| CCCL build-only headers | NVIDIA nvidia-cuda-cccl-cu12 12.8.90 |
| Compiler | System nvcc 12.9.86 and /usr/bin/g++ 11.5.0; A100/sm80 only |

Direct dependencies and resolved lock are `local_setup/requirements.txt` and
`local_setup/requirements.lock.txt`. No datasets, lm-eval, plotting, or training
packages were installed for this workflow. GPU/kernel compatibility is not
proven by dependency resolution or successful compilation alone.

### Runtime layout and build fixes

The model snapshot is an immutable input, including its published Python code.
Its deployment code is copied mechanically into `local_setup/runtime/deploy`,
and its own CUDA sources are compiled into that separate runtime directory.
The checkpoint runtime is used deliberately: the current source checkout has
a different `block_matmul.py`. No numerical patches were applied to either.
The checkout provides the pinned CUTLASS headers and our setup scripts.

`local_setup/build_inference.py` builds only `deploy._CUDA`; it avoids the
upstream setup's automatic installation/build of unrelated dependencies.
Official FlashAttention/Hadamard wheels passed CPU import/ABI checks.

The initial extension build failed because module-provided `CPATH` selected
CUDA 13.2 headers and the system CUDA 12.9 toolkit lacked runtime/CCCL headers.
The fixed build clears inherited CUDA include/library paths, uses the installed
CUDA 12.8 runtime and CCCL headers, and supplies an isolated linker symlink to
the environment's libcudart. The 12.9 compiler / 12.8 PyTorch minor-version
warning is retained; numerical compatibility still needs the GPU test.

- First failed build log: `local_setup/logs/build-cuda.log`.
- Successful isolated build log: `local_setup/logs/build-cuda-isolated.log`.
- Compiled binary: `local_setup/runtime/deploy/_CUDA.cpython-310-x86_64-linux-gnu.so`.
- Only two compiler workers are used. Compilation does not execute GPU kernels.

To rebuild or restore dependencies, outside safe-quant:

```bash
cd /work/nvme/bgxm/wliu12/repos/FlatQuant
uv pip install --python /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/python \
  --index-strategy unsafe-best-match -r local_setup/requirements.lock.txt
# Needed only if restoring an absent snapshot; it never imports custom code:
CUDA_VISIBLE_DEVICES='' /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/python \
  -B local_setup/download_checkpoint.py
git submodule update --init --depth 1 third-party/cutlass
bash local_setup/build_inference.sh
```

The dependency lock contains both PyPI and the official PyTorch cu128 index;
exact package versions and direct official wheel URLs are pinned. Keep the
managed Python directory and model snapshot at their documented locations.

### CPU checks and one-GPU inference smoke

CPU checks validate download sizes and code/config hashes, safetensors headers,
W4A4KV4 metadata, exact tokenizer vocabulary/chat-template behavior, scoring
token IDs, wheel imports, and the locally compiled extension import. Weight
SHA256 values were computed during download; CPU checks read only tensor
headers, not full weight contents, and do not construct or run the model.
These checks passed on 2026-10-15. The saved headers contain 224 uint8 packed
weight tensors, alongside FP16, FP32, and integer metadata tensors. Tokenizer
vocabulary, tested chat rendering/token IDs, and BOS/EOS matched the local base.

Triton 3.3 initializes a CUDA driver while decorating autotuned functions.
An attempted full-runtime import with GPUs hidden therefore failed with
`0 active drivers`. Full deployment-runtime imports are intentionally deferred
to the Slurm GPU test, without mocking a driver or modifying upstream code.

Run CPU checks:

```bash
cd /work/nvme/bgxm/wliu12/repos/FlatQuant
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  TOKENIZERS_PARALLELISM=false \
  /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/python -B local_setup/check_cpu.py
```

The following test passed in a user-run interactive allocation on gpua056,
job 22913901. The command remains available for reproducing it in batch; the
assistant has not submitted it:

```bash
sbatch /work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup/check_inference_1gpu.sbatch
```

Requests **one A100, 64 GB host RAM, eight CPUs, and 15 minutes** on
`gpuA100x4` / `bgxm-delta-gpu`. The partition name does not mean four GPUs.
Native kernels are precompiled, but the first Triton execution may compile and
autotune kernels. This is a smoke allocation, not a proven full-run time/memory
budget. If it times out, inspect the last stage before changing the limit.

The job runs offline and checks full runtime import, INT4 pack/GEMM/dequant
against independent integer references, FlashAttention against PyTorch SDPA,
and Hadamard against a CPU butterfly reference. It then loads the exact saved
model, checks packed linear weights/scales, runs one prompt and four greedy
cached decode steps, and verifies actual uint8-packed KV storage. It reports
elapsed time and peak allocated GPU memory. It does not run a benchmark.

The published loader calls `load_state_dict(strict=False)` twice. Our smoke
wrapper records both results and checks combined state coverage, allowing only
documented shared aliases, unused fields, and calibration-only leftovers. It
fails on unrecognized missing/runtime state rather than silently evaluating
random initialization. This load audit itself remains to be exercised on GPU.
The wrapper also restores the process default dtype even if loading fails.
The saved audit for job 22913901 has no unresolved required state and no
unrecognized checkpoint state. The run detected 224 packed linear layers,
INT4 cache/uint8 storage, and passed cached decoding with finite logits.
Elapsed time was 114.2836 seconds and peak allocated GPU memory 5.4876 GiB
for this short smoke, including initial compilation; not a throughput result.
The four fixed decode steps intentionally ignore EOS to exercise the cache.
`Hello!<|eot_id|><|start_header_id|>` is therefore not evidence that normal EOS
stopping was tested. Full `generate()`, EOS behavior, fresh-load repeatability,
and matched-evaluation compatibility remain pending. Warnings about tensor
copying, future GenerationMixin inheritance, and logits dtype were nonfatal.

Logs/results:

- `local_setup/logs/cpu-check.log`: CPU check output.
- `local_setup/logs/infer1-JOBID.out` and `.err`: Slurm output and errors.
- `local_setup/logs/load-JOBID.json`: model-state loading audit.
- `local_setup/triton-cache`: runtime compilation/autotuning cache.

After submission, replace the placeholder with the returned ID:

```bash
flatquant_job=12345678
squeue -j "$flatquant_job"
sacct -j "$flatquant_job" --format=JobID,State,Elapsed,ExitCode
tail -n 100 "/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup/logs/infer1-${flatquant_job}.out"
tail -n 100 "/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup/logs/infer1-${flatquant_job}.err"
```

### Evaluation gate: do not silently change the scientific condition

The published config labels the model BF16, but the custom loader initializes
FP16 nonpacked tensors and the CUDA kernels use FP16. The runtime also forces
cache use, requires FlashAttention 2, and uses a different prefill path from
its quantized-cache decoding path (upstream documents unquantized Q/K during
prefill). Therefore passing a BF16 dtype or `use_cache=False` is not sufficient
to claim those settings were honored.

Keep the existing BF16 reference and SpinQuant W4A4KV4 condition unchanged.
Before safe-quant integration, validate the exact scoring/generation protocol,
masks, padding, sequence lengths, batching, and effective A/K/V quantization
boundaries. Record FP16 and cache/prefill differences explicitly. Any change to
baseline evaluation settings requires a separate discussion and agreement.
The exact Llama-3 safe-quant adapter now uses the previously validated local
runtime, audits shard loading and verifies snapshot/runtime Python hashes.
It explicitly rejects `use_cache=False`; it never changes a frozen protocol
to make it run. Existing Llama-3.1 exploratory configuration is not this model
and is preserved unchanged. Full matched-evaluation compatibility remains
unvalidated, even if the new inference smoke passes.

## Saved-checkpoint inference checks: QuaRot and FlatQuant

Integration files live in safe-quant:

- `src/safe_quant/model_loaders/quarot.py` and `flatquant.py`.
- `configs/models/llama3_8b_instruct_quarot_w4a4kv4.yaml`.
- `configs/models/llama3_8b_instruct_flatquant_w4a4kv4.yaml`.
- `scripts/check_quantized_inference.py`, its `.sh` launcher, and
  `slurm/check_quantized_inference.sbatch`.

The launcher activates each method's existing isolated environment and uses
safe-quant via `PYTHONPATH`; no package installation or quantization is needed
for these checks. Do not install safe-quant's `[eval]` extra into flatquant-eval:
that extra pins a different Transformers release. If a later benchmark needs
datasets/lm-eval dependencies, check and add only those dependencies separately
without replacing the pinned numerical runtime.

Both checks use one GPU. Run inside your existing interactive Slurm GPU
allocation (never on a login node):

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/check_quantized_inference.sh quarot "outputs/inference-checks/quarot-${SLURM_JOB_ID}"
bash scripts/check_quantized_inference.sh flatquant "outputs/inference-checks/flatquant-${SLURM_JOB_ID}"
```

Each command launches two separate Python processes, loading the immutable
saved artifact both times. Use a NEW output parent on retries; existing
directories are intentionally not overwritten. The script does not request
an allocation for you.

Alternatively, submit each as its own 15-minute, one-GPU job yourself:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
sbatch --account=bgxm-delta-gpu --partition=gpuA100x4 --output=quarot-inference-%j.out --error=quarot-inference-%j.err slurm/check_quantized_inference.sbatch quarot outputs/inference-checks/quarot-first
sbatch --account=bgxm-delta-gpu --partition=gpuA100x4 --output=flatquant-inference-%j.out --error=flatquant-inference-%j.err slurm/check_quantized_inference.sbatch flatquant outputs/inference-checks/flatquant-first
```

Fifteen minutes is an initial smoke-test limit, not a full evaluation budget.
If queued or interrupted, preserve logs and use a new output directory when
retrying. There is no need to delete or regenerate any quantized checkpoint.

Each output parent contains `capture/` and `reload/` with:

- `report.json`: pass/fail, effective method/runtime, quantizer/storage flags,
  provenance, timing and peak GPU memory. FlatQuant also saves its load audit.
- `predictions.jsonl`: two neutral prompts with native chat token IDs and
  greedy responses, separate from the aggregate report.
- `last-logits.pt`: fixed-input logits, compared after fresh-process loading
  with declared absolute tolerance 1e-4 and relative tolerance 0; greedy
  response token IDs must match exactly.

The smoke also forces the native EOT token in a separate synthetic generation
control test to check EOS stopping. This is not an accuracy test. FlatQuant's
actual packed cache and refusal of `use_cache=False` are checked explicitly.
Generation uses the BF16 base model's native EOS/generation configuration for
both methods, rather than inventing defaults missing from the published
FlatQuant directory. The behavioral evaluator supports an explicit
`generation_config_path` for that purpose; all existing default paths and
frozen scientific generation settings remain unchanged.

Success requires `reload/report.json` to report `status: passed` and
`fresh_process_reload: passed`. GPU results are pending user execution.
Afterward, run a small subset of the chosen existing benchmark before a full
run. QuaRot can honor cached and uncached requests. FlatQuant cannot run the
existing uncached FinGuard/WildGuard/PPL protocols as matched results; cached
behavioral protocols are possible candidates, but FP16/attention/prefill
differences still require explicit interpretation, not an automatic claim of
matched settings. Do not change the BF16 reference or SpinQuant W4A4KV4 to
accommodate FlatQuant without separate agreement.

## Interactive alternative: single-GPU QuaRot and FlatQuant tests

For the newer FlatQuant grouped benchmark smoke jobs, use the section below.
The older helper-only interactive commands in this section remain available.

The user can request a single A100 interactively instead of submitting the
prepared batch tests. Request the allocation from the login node:

```bash
srun --account=bgxm-delta-gpu --partition=gpuA100x4 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-node=1 \
  --mem=64G --time=00:15:00 --pty bash -l
```

Wait until the compute-node shell opens. Do not run the Python GPU tests on the
login node. Each test uses one GPU; `gpuA100x4` is the partition name. Run Python
directly below, without nesting another `srun` inside the interactive step.

QuaRot environment/helper test (does not load or quantize the model):

```bash
cd /work/nvme/bgxm/wliu12/repos/QuaRot
source /work/nvme/bgxm/wliu12/envs/quarot/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
unset PYTHONPATH
python -B local_setup/check_cpu.py
python -u -B local_setup/check_gpu.py
```

FlatQuant kernel/model-inference smoke (does not re-quantize weights):

```bash
cd /work/nvme/bgxm/wliu12/repos/FlatQuant
source /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/activate
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
export TRITON_CACHE_DIR="$PWD/local_setup/triton-cache"
export TORCH_EXTENSIONS_DIR="$PWD/local_setup/torch-extensions"
export HF_MODULES_CACHE="$PWD/local_setup/hf-modules"
unset PYTHONPATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
python -B local_setup/check_cpu.py
python -u -B local_setup/check_gpu.py
```

Both tests may run sequentially in the same allocation if time permits; do
not run them concurrently. FlatQuant's first Triton compilation/autotuning
may take several minutes, so request a fresh allocation if needed rather than
assuming the remaining time is sufficient. `exit` releases the interactive
shell/allocation. Capture the terminal output and Slurm job ID for the work log;
interactive jobs do not automatically create the batch `.out`/`.err` files.

## FlatQuant first: two grouped benchmark smoke jobs

The user approved keeping native implementations first, including published
FlatQuant FP16 tensors/kernels and its custom INT4 cache. No quantizer math,
checkpoint tensor, or deployment kernel was changed for these benchmarks.
The model's BF16 config label is not a guarantee of BF16 execution. Record
native runtime differences; do not describe these runs as a strictly
BF16-controlled quantization-only comparison.

Only the requested benchmarks are included:

| Smoke job | Evaluation subset |
|---|---|
| `scoring smoke` | WikiText-2: 2 contiguous 2048-token blocks; ARC-E, ARC-C, PIQA, WinoGrande: 2 examples per task; FinGuard: first 4 examples; WildGuardTest: first 4 examples after the original null-label filter |
| `generation smoke` | HarmBench: prepared rows 0,1; XSTest: verified rows 0,25 (one safe, one unsafe contrast; source rows are interleaved); full 313-row StrongREJECT source: rows 0,156 |

Scoring calls use explicitly versioned cache-enabled protocols. FinGuard uses
batch size 4 and WildGuard now uses batch size 2 in both smoke and full modes,
at the user's request. WildGuard batch size 2 matches the previous SpinQuant
W4A4KV4 run's batching, not its native precision/cache policy. WikiText-2 and
the four-task suite remain at batch size 1. Earlier batch-1 and batch-4 results
are preserved separately; they do not establish batch-2 numerical equivalence.
The four-task protocol records the batch-size change
from the earlier batch-size-8 version. Prompts, candidate IDs, metric formulas,
zero-shot settings and data revisions are unchanged. Existing uncached configs
and previous results are not overwritten. Future cross-model comparisons must
use the same new protocol, not silently mix old and new runs.

The classification/PPL cache is fresh for each independent forward/block;
past_key_values are not passed between examples. Native FlatQuant prefill
still uses K/V before cache quantization: allocating a cache is not evidence
that a single-forward score measures KV4 decoding. Generation remains native
chat, greedy, cache-enabled, batch size 1, with max_new_tokens=256 and the same
base-model native EOS configuration. No shorter output cap was substituted
just to make the smoke faster.

### Environment and CPU/data checks: prepared

Added datasets 2.20.0, pyarrow 15.0.2, lm-eval 0.4.8 and peft 0.13.2 to the
existing flatquant-eval environment. PyTorch 2.7.1+cu128, Transformers 4.45.0,
Triton 3.3.1, flash-attn 2.8.3.post2 and FHT 1.1.0 remain unchanged. The only
previously installed package replaced was fsspec, 2026.9.0 -> 2024.5.0, required
by pinned datasets. Dependency compatibility (`uv pip check`) passes.

Reproducibility files in safe-quant:

- `requirements/flatquant-bench.txt`: requested benchmark and protected runtime pins.
- `requirements/flatquant-before-bench.freeze.txt`: environment before this addition.
- `requirements/flatquant-bench.freeze.txt`: resolved installed environment afterward.
- `scripts/check_flatquant_bench.py`: CPU-only imports/config/data preflight;
  no model is constructed and deployment code is not imported.
- `outputs/flatquant/setup/cpu-preflight-20261015.json`: saved preflight report.

All requested datasets/prepared prompts are already cached. The preflight
loads the same pinned scoring splits/tasks on CPU and checks hashes for all
prepared generation files. Dataset cache defaults to
`/u/wliu12/.cache/huggingface/datasets`; cached PIQA/Winogrande Python modules
come from `/u/wliu12/.cache/huggingface/modules`. Jobs run offline rather than
downloading data or models on compute nodes.

New explicit configs: `wikitext2_cached.yaml`, `general_4task_cached_bs1.yaml`,
`finguard_cached.yaml`, `wildguard_prompt_cached.yaml`. The existing cached-256
generation configs/prepared prompts are reused unchanged. WildGuard gains a
test-only max-samples limit recorded in its manifest; the original BF16
WildGuardTrain validation selection/protocol remains unchanged.

### Submit exactly two smoke jobs yourself

From the login node, submit these commands (the assistant does not submit):

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
sbatch --job-name=flatquant-score-smoke slurm/flatquant_bench.sbatch scoring smoke
sbatch --job-name=flatquant-gen-smoke slurm/flatquant_bench.sbatch generation smoke
```

Each requests one A100, 8 CPUs, 64 GiB host memory and an initial 30-minute
wall-time limit. This is a conservative unmeasured smoke budget, not a full-run
estimate. The seven scoring benchmarks run sequentially in the first job;
the three generation benchmarks run sequentially in the second. Jobs are
independent. Both first run CPU preflight and the existing two-process
saved-load/EOS/reload check internally, so no third GPU job is needed.

To use an existing interactive GPU allocation instead:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/run_flatquant_bench.sh scoring smoke scoring-run1
bash scripts/run_flatquant_bench.sh generation smoke generation-run1
```

The launcher selects flatquant-eval automatically. Do not use the older
BF16/SpinQuant-only wrappers or install safe-quant[eval] into this environment.

### Outputs and interpretation

Batch outputs: `flatquant-JOBID.out` / `.err` in safe-quant.
Detailed outputs: `outputs/flatquant/scoring/smoke-JOBID/` and
`outputs/flatquant/generation/smoke-JOBID/`. Each contains per-stage logs,
`preflight.json`, and `inference/capture/` plus `inference/reload/` reports.
Each benchmark then gets its own directory. Classification predictions,
PPL block NLLs, harness per-example samples and generated responses remain
separate from aggregate metrics. Runtime metadata records FP16, actual cache
policy, precision/deployment exceptions and immutable artifact provenance.

The XSTest job also runs the existing CPU-only official-prefix string-match
judge on both smoke subsets. HarmBench's large classifier and StrongREJECT's
existing rubric judge are deliberately separate downstream stages: their
responses are saved, but no ASR/StrongREJECT score is claimed until judged.
No paid API judge is invoked automatically. These small subsets validate
execution, not benchmark quality; do not report smoke scores as full results.

The job stops at the first failing stage, with that stage's traceback in its
log. Keep failures and partial outputs, fix the cause, then rerun with a new
job ID or RUN_TAG. Existing outputs/checkpoints are not deleted or overwritten.
The user subsequently completed both grouped smoke runs on A100 allocation
22915865. Saved-load/EOS/reload checks passed, including exactly matching
fresh-process logits. All seven scoring smoke outputs completed; generation
saved two responses per benchmark and XSTest local judging completed. These
remain partial execution checks, not full-dataset results.

### Full runs after smoke success

The launcher also supports `scoring full`, `harmbench full`, `xstest full`,
`strongreject full`, and the independent scoring groups below.
`generation full` is intentionally rejected so the
three long generation runs remain separate jobs, as requested. Use fresh tags
and choose full-job wall times after inspecting smoke throughput; the default
30-minute batch limit is not a full-run recommendation. Preserve the existing
HarmBench and StrongREJECT judge protocols rather than introducing substitutes.

### Independent full scoring runs: current preferred grouping

The user requested three independent jobs/interactive runs for the remaining
FlatQuant scoring benchmarks:

| Launcher group | Benchmarks | Output root with RUN_TAG=run1 |
|---|---|---|
| `general` | ARC-E, ARC-C, PIQA, WinoGrande together | `outputs/flatquant/general/full-run1/` |
| `finguard` | FinGuard only | `outputs/flatquant/finguard/full-run1/` |
| `wildguard` | WildGuardTest only | `outputs/flatquant/wildguard/full-run1/` |

Inside a user-requested interactive Slurm GPU allocation, run one command per
allocation, or sequentially if the allocation has sufficient time remaining:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/run_flatquant_bench.sh general full run1
bash scripts/run_flatquant_bench.sh finguard full run1
bash scripts/run_flatquant_bench.sh wildguard full bs2-run1
```

Each uses the existing flatquant-eval environment and unchanged cache-enabled
benchmark configs. FinGuard uses batch size 4 and WildGuard uses batch size 2;
the four-task suite remains at batch size 1. This applies to their standalone groups and the combined
scoring group, in both smoke and full modes. Actual batch sizes are recorded in
run manifests. No prompt, candidate, scoring, few-shot, generation, checkpoint
or numerical-runtime setting is changed. Both earlier batch-4 smoke and full
classifier runs completed, but score changes relative to batch 1 were observed.
WildGuard batch 2 is pending GPU execution. Optional fresh smoke commands:

```bash
bash scripts/run_flatquant_bench.sh finguard smoke bs4-check2
bash scripts/run_flatquant_bench.sh wildguard smoke bs2-check
```

These smoke jobs check execution but do not by themselves establish numerical
equivalence to batch size 1. Existing results and already-started jobs are unchanged.
Preflight checks only the selected group's datasets. Per-stage logs, predictions
and metrics remain under each independent output root. Existing roots are not
overwritten. The same groups also accept smoke mode if a new check is desired;
the original two grouped smoke modes remain available.

Do not rerun PPL just to restart the other benchmarks. The completed result is
preserved at `outputs/flatquant/scoring/full-run1/wikitext2/`: PPL
9.011714050744322, all 141 complete 2048-token blocks, 288627 predicted tokens.
The four-task suite in that older combined run stopped at 4464/20398 likelihood
requests (~22%) when allocation 22915865 ended (FAILED, exit 130). Its stale
running manifest and log are preserved; no per-task predictions or final metrics
were saved. FinGuard/WildGuard had not started in that combined run.

The new general job must restart the entire four-task suite; it cannot resume
the previous in-memory requests. WinoGrande intentionally uses the pinned
lm-eval winogrande_xl dataset configuration and 1267-example validation split,
zero-shot; this is not model training. No GPU job has been submitted or run by
the assistant. New split-job GPU results are pending user execution.

CPU verification after adding these groups: 91 tests passed, including shell
routing with all actual evaluator commands stubbed out, original grouped-mode
compatibility, and unchanged frozen benchmark configurations. Shell syntax
passes; each selected group's offline data preflight is checked separately.

## AdvBench DirectRequest: prepared for user GPU smoke tests

Scope: BF16, FlatQuant and SpinQuant W4A4KV4. The user approved DirectRequest,
not GCG or another jailbreak attack. Submit all 520 original `goal` requests;
keep `target` as metadata only, never as assistant prefill. Official source:
https://github.com/llm-attacks/llm-attacks/tree/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench

Pinned source commit: `098262edf85f807224e70ecd87b9d83716bf6b73`.
CSV SHA-256: `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`.
CSV and exact original scorer sources are cached in safe-quant under
`outputs/behavioral/sources/advbench/`. Shared prepared prompts and their manifest
are under `outputs/behavioral/prompts/advbench_directrequest-cached-256-v1/`.

Generation: greedy, cache enabled, max_new_tokens 256, native user-only Llama-3
chat template and native EOS IDs. Batch size defaults to 1; the optional fourth
launcher argument enables batch 4 for AdvBench only. Both models use the same prepared
prompts, including fixed smoke indices `0,74,148,222,296,370,444,519`. Native
environments and checkpoint tensors are unchanged; FlatQuant still executes its
FP16 runtime and SpinQuant has BF16 floating-point portions. Record those
differences rather than claiming a precision-controlled comparison.

Run sequentially inside an interactive Slurm GPU allocation:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/run_advbench.sh flatquant smoke bs4-run1 4
bash scripts/run_advbench.sh spinquant smoke bs4-run1 4
```

Or submit the two smoke jobs yourself (one A100 each; default 30 minutes):

```bash
sbatch --job-name=advbench-flat-smoke slurm/advbench.sbatch flatquant smoke bs4-run1 4
sbatch --job-name=advbench-spin-smoke slurm/advbench.sbatch spinquant smoke bs4-run1 4
```

After reviewing smoke results, change `smoke` to `full` for all 520 prompts.
Choose a longer sbatch `--time=HH:MM:SS` based on smoke throughput; the default
30-minute request is not a full-run estimate. The assistant has submitted no
jobs and has run no GPU inference.

Each launcher activates the correct existing environment and automatically
runs the CPU-only original refusal-string scorer after generation. Outputs:
`outputs/behavioral/advbench_directrequest/{flatquant,spinquant}/{smoke,full}-RUN_TAG/`.
Raw responses are in `generation/`, individual judgments and metrics in
`judgments/strmatch/`, and step logs in `logs/`. Existing paths are never reused.

Report `non_refusal_rate` as a **non-refusal proxy**, not semantic harmful
compliance or GCG attack success. Original matching is case-sensitive substring
matching anywhere, not XSTest prefix matching. Exact lists and source hashes
are recorded. Empty responses retain original matching behavior and are counted
separately. No paid API or additional judge GPU is needed.

AdvBench paired comparison uses actual model IDs. It checks matching prompts,
scorer and generation settings, recording native metadata differences for the
4.44/4.45 Transformers serialization and unused assistant-decoding defaults/
cache-return representation. Full configs remain saved; existing benchmark
comparison rules are unchanged. See safe-quant `README.md`, section
"AdvBench DirectRequest", for preparation and paired-comparison commands.
GPU smoke validation is pending user execution.

CPU verification: 133 tests passed, including matching the actual pinned
upstream refusal expression, immutable prompt identity, CPU judging, paired
model comparison, and shell routing with evaluator commands stubbed. Shell
syntax passed. All 520 prompts fit context (maximum 39 prompt tokens plus 256
new tokens). Both native environments rendered identical prompt token IDs:
SHA-256 `050b0b67b33aa4303ce2460f0b34a7fac3bcf6f57a893c54f63d27b1cf6e03b2`.
No model was loaded for these CPU checks.

Batch-4 update: smoke runs use two batches of four prompts with left padding.
Native generation handles positions, sequence caches and EOS stopping. Cache
state is never reused across independent batches. Response extraction uses the
padded input width and trims each row after its first EOS so waiting pads do
not enter response text/token counts. Actual batch sizes and batch wall times
are recorded; per-example times are explicitly amortized, not individual
latencies. Existing batch-1 behavior remains the default, and other behavioral
benchmarks still reject larger batches. No checkpoint, deployment kernel or
quantization algorithm was changed. CPU stub tests cover padding, EOS,
short final batches and default-1 routing; GPU validation remains pending.
Use fresh tags to preserve older batch-1 outputs. Quantized output equivalence
between batch sizes is not assumed; the paired comparison requires matching
batch sizes for the two models.

SpinQuant smoke startup fix: the AdvBench launcher now adds
`/work/nvme/bgxm/wliu12/repos/SpinQuant` to `PYTHONPATH` as well as safe-quant's
`src`. The saved-state loader needs the upstream `eval_utils` and `utils`
packages; setting `SPINQUANT_ROOT` alone does not add Python import paths.
CPU preflight checks package discovery from that checkout without executing
model code. Preserve the failed `smoke-bs4-run1` outputs and retry inside a
Slurm GPU allocation with a fresh tag:

```bash
bash scripts/run_advbench.sh spinquant smoke bs4-run2 4
```

No checkpoint, environment package or generation setting was changed by this
import-path fix. FlatQuant keeps its isolated import path.

### AdvBench HarmBench-classifier ASR: separate cached-response judge

The user requested semantic judging in addition to the original refusal-string
proxy. A new `advbench_harmbench_classifier` backend reuses both saved full
520-response caches; no target generation, checkpoint loading, quantization or
environment package changes are needed. Original proxy scores remain intact.

Inside an interactive Slurm allocation with one free A100 40 GB, run sequentially:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/judge_advbench_harmbench.sh flatquant
bash scripts/judge_advbench_harmbench.sh spinquant
```

The launcher activates `/work/nvme/bgxm/wliu12/venvs/harmbench-judge`. It defaults
to generation tag `full-bs4-run1` and fresh judge tag `harmbench-run1`. Optional
syntax: `MODEL GENERATION_RUN_TAG JUDGE_TAG`, using a new judge tag for retries.
It never submits jobs, regenerates target responses or calls a paid API.

Each immutable AdvBench goal is adapted to a standard HarmBench Behavior with
its stable sample ID. Tags/ContextString are empty, so the original standard
classifier template is used. The adapter checks the pinned AdvBench source,
all prompt identities and full-set coverage, then judges every saved response,
not just heuristic non-refusals. The adapted CSV is written only inside the
new judgment folder. Original CSVs, responses and string-match results remain
unchanged.

Classifier snapshot: `bda705349d1144fa618770bea64d99ce54e3835b`, same as earlier
HarmBench runs. Official code revision:
`8e1604d1171fe8a48d8febecd22f600e462bdcdd`, with evaluator source hashes checked.
Unmodified native evaluation uses a 512-classifier-token response cap,
temperature 0, one output token and tensor parallel size 1. Invalid labels and
missing IDs fail closed. Judge code/model provenance, adapted inputs, native
results, individual judgments and aggregate metrics are saved.

Score: `asr = classifier_positive_count / 520`; multiply by 100 for percent.
Report **AdvBench DirectRequest, HarmBench-classifier ASR**, not HarmBench
dataset ASR or jailbreak-attack ASR. It remains an automated classifier result,
not human ground truth. Results for each model are in
`outputs/behavioral/advbench_directrequest/MODEL/full-bs4-run1/judgments/harmbench-run1/metrics.json`.
Logs are in the run's `logs/judge-harmbench-run1.log`. See safe-quant README for
the paired CPU comparison command. The comparison rejects mixing classifier
scores with the original refusal-string scores.

Both real caches passed CPU adapter/source-identity validation (520 each).
Classifier subprocesses are stubbed in unit tests; actual GPU classifier
execution remains pending user action.
Full CPU regression suite: 153 tests passed; launcher shell syntax passed.

### Added BF16 AdvBench baseline

The user requested a matched unquantized baseline. Both AdvBench generation
and classifier launchers now accept `bf16`. It selects the existing
`configs/models/llama3_8b_instruct_bf16.yaml` and unquantized local BF16 checkpoint.
The existing Transformers 4.44.2 environment is reused without adding SpinQuant
to BF16's Python import path; no rotation, quantization or quantized-state
loading occurs. BF16 tensors and its existing SDPA runtime are retained.

Inside an interactive Slurm allocation with one free A100 40 GB, run sequentially,
checking the smoke before starting the full generation:

```bash
cd /work/nvme/bgxm/wliu12/safe-quant
bash scripts/run_advbench.sh bf16 smoke bs4-run1 4
bash scripts/run_advbench.sh bf16 full bs4-run1 4
bash scripts/judge_advbench_harmbench.sh bf16
```

The same eight smoke rows/all 520 full goals, batch 4, greedy generation,
256-new-token limit, cache-enabled native user chat/EOS settings and pinned
HarmBench classifier are used. Outputs are separate under
`outputs/behavioral/advbench_directrequest/bf16/`. The final score is in
`full-bs4-run1/judgments/harmbench-run1/metrics.json` (`asr` times 100 for percent).
The original string-match proxy is automatically saved separately as before.

After all three classifier runs complete, use the two CPU-only commands in
safe-quant README's "BF16 baseline and matched three-model comparison" section.
They compare BF16 against FlatQuant and SpinQuant W4A4KV4, validating matching
prompt identities, generation settings and judge provenance and saving each
ASR difference plus paired label flips. Missing runs or mixed scorer backends
are rejected. Existing quantized response caches do not need regeneration.
No new environment was built, no model inference was run, and no job was
submitted by the assistant. BF16 GPU validation is pending user execution.
BF16-extension verification: all 160 CPU tests passed; generation/judge routing
and both BF16 paired comparisons are covered. Launcher/wrapper shell syntax passed.

## Other methods and references

- SpinQuant: <https://github.com/facebookresearch/SpinQuant>. Preserve the
  existing saved condition; do not rerun quantization as part of evaluation.
- FlatQuant: <https://github.com/ruikangliu/FlatQuant/blob/main/REALQUANT.md>.
  Exact-model checkpoint:
  <https://huggingface.co/Hyun9junn/Meta-Llama-3-8B-Instruct-W4A4KV4-FlatQuant>.
  Its custom runtime forces cache use and differs from the fake-quantization
  runtime; check protocol compatibility before reporting matched comparisons.
- OSTQuant: <https://github.com/BrotherHappy/OSTQuant>. Existing official
  Llama-3-8B transformation files were not verified as Instruct artifacts.
  We are preparing our own Instruct condition.
- QuaRot: <https://github.com/spcl/QuaRot>. Official research code has save/load
  and K/V precision options. Its model allowlist omits Instruct, so exact-model
  compatibility needs verification before producing an artifact.
- Driver compatibility:
  <https://docs.nvidia.com/deploy/cuda-compatibility/latest/why-cuda-compatibility.html>.

## Work log and next update

### 2026-10-15

- Final scope set to SpinQuant, FlatQuant, OSTQuant, and QuaRot W4A4KV4.
- Previous FlatQuant source and `flatquant-eval` environment were permanently
  removed at the user's request; checkpoints were preserved.
- Cloned and pinned OSTQuant, created its isolated environment and dependency
  lock, and completed CPU/import/parser checks.
- Added CPU and four-GPU environment checks, an optional Slurm batch wrapper,
  and setup documentation. No GPU jobs were submitted by the assistant.
- User initially preferred an interactive allocation, then switched to sbatch
  due to queue waiting. Documented submission, status, and log commands.
  Temporarily changed the environment-check wall time from 15 to 30 minutes,
  then restored it to 15 minutes at the user's request. No job has been
  submitted by the assistant.
- Next evidence required: environment-check job ID, logs, and outcome.
- While waiting, inspected QuaRot's official dependency, GPTQ, rotation, and
  save/load code and recorded the proposed single-GPU build plan above.
- At the user's request, consolidated setup documentation into this shared
  guide and deleted the duplicate OSTQuant local setup README. Upstream
  quantizer READMEs were retained.
- Cloned and pinned QuaRot; built an isolated environment and dependency lock;
  passed CPU/import/tokenizer/parser/interface compatibility checks. Fixed
  Instruct/local-model parsing and the immutable-input log-path bug.
- Prepared a 15-minute, single-A100 QuaRot environment check for user submission.
  No QuaRot GPU/model/quantization runs have occurred. Full-model BF16 parity,
  GPTQ, save/reload, and evaluation integration remain pending.
- At the user's request, started a fresh FlatQuant inference-only setup using
  the published exact-model W4A4KV4 artifact rather than re-quantizing weights.
  Downloaded/checksummed its pinned snapshot, installed/locked a separate
  environment, and compiled matching A100 kernels outside the model directory.
  Fixed mixed CUDA header paths and missing build headers without changing
  quantization numerics; retained the original failed build log.
- Prepared CPU artifact/tokenizer/extension checks and a 15-minute single-A100
  inference-only Slurm smoke with combined weight-loading audit. Full Triton
  imports require GPU availability and are deferred to Slurm. No FlatQuant GPU
  or model inference has run, no Slurm job was submitted by the assistant, and
  no safe-quant files or existing model artifacts were changed.
- FlatQuant CPU checks passed: checkpoint headers/metadata and file/code
  integrity, exact tokenizer vocabulary and tested chat IDs, compiled extension
  import, and official wheel imports. All 49 installed packages passed the
  dependency consistency check. GPU and full-model results remain pending.
- User supplied successful interactive QuaRot Hadamard and FlatQuant model
  smoke results from gpua056. FlatQuant's job 22913901 load audit was inspected;
  no unresolved required state or unrecognized checkpoint fields remained.
- At the user's request, prepared QuaRot's quantization-disabled BF16 model
  smoke and optional one-A100/15-minute batch wrapper. Fixed RMSNorm BF16
  accumulation with an exact nine-case CPU regression. Full-model GPU parity,
  cache/mask tests, GPTQ, and checkpoint production are still pending.
- The user then explicitly requested original upstream QuaRot behavior.
  Reverted only the RMSNorm BF16-upcast patch, preserving parser/path fixes.
  Updated CPU/GPU norm diagnostics to verify the upstream expression and
  report Llama differences, keeping full-model NRMSE/KL gates unchanged.
  The base model remains BF16; saved model inputs and safe-quant were untouched.
- The subsequent model test failed the custom rotated-vs-original gate on
  prompt 0; its report remains under `logs/model-22913901-20261015T065221321104Z`.
  The user explicitly stopped further parity investigation and approved moving
  to native quantization. This is not a pass or evidence of 4-bit failure.
- Prepared the official W4A4KV4 GPTQ launcher and two-hour single-A100 batch
  alternative, with upstream symmetric defaults and required K head grouping.
  Cached native WikiText-2 calibration/test inputs and recorded provenance.
  Fixed evaluator type detection for local paths and persisted native PPL/NLL
  results separately without changing numerical evaluation. Quantization jobs
  remain user-run; the assistant has not submitted or executed one.
- User-run native attempt 22914500 on gpua012 failed during calibration input
  capture because Transformers 4.44's shared model-level RoPE remained on CPU.
  No checkpoint exists for that run. Applied device-only moves in GPTQ/PPL
  capture and updated strict launcher source checks to allow exactly those
  blocks. CPU branch/source/syntax checks passed; restart and GPU confirmation
  are pending. No normalization, quantization, or calibration choices changed.
- The retry `official-22914500-20261015T072741204302Z` completed native GPTQ and
  PPL evaluation: 10.464049, 1166.34 seconds, qmodel.pt saved. Inspected native
  log, metrics JSON, and completed manifest. The code-default recipe differs
  from the paper's clipping/KV/online-Hadamard settings; this distinction was
  explained to the user rather than labeling the result a paper reproduction.
- The user approved paper-style settings. Updated only launcher flags/run
  labeling/manifest and batch job name, keeping the core algorithms, original
  normalization, BF16 base, cache, and existing artifacts unchanged. New runs
  use the `paperstyle-` prefix; CLI, manifest consistency, source integrity,
  and syntax are checked on CPU. No paper-style GPU job has run yet.

### 2026-10-15: paper-style artifact and inference integration

- Inspected completed paper-style job 22914989: native PPL 10.027104377746582,
  completed run manifest and 16,072,004,244-byte saved state. Kept it and the
  older code-default checkpoint immutable; no failed/incomplete run was deleted.
- Added inference-only QuaRot loading and an exact Llama-3 FlatQuant config
  using the already rebuilt runtime. Existing BF16, SpinQuant W4A4KV4 and
  Llama-3.1 exploratory configs remain unchanged. No numerical quantizer source
  or published checkpoint files were modified.
- Added the optional behavioral generation-default path so FlatQuant uses the
  same base-model native EOS configuration. Frozen prompt, generation length,
  sampling, batching and cache choices were not changed. FlatQuant requests
  for use_cache=False now fail clearly in the new exact-model adapter.
- Added the Slurm-only two-process saved-load smoke and user-submitted
  15-minute batch alternative. GPU forward/generation/reload checks are pending;
  the assistant has neither submitted a job nor run model inference.
- CPU validation: loader-contract/config/regression tests passed; exact FlatQuant
  config validation passed in flatquant-eval. Construction-only QuaRot check
  passed with 516 meta parameters, 225 wrappers, 32 online full and 32 partial
  Hadamards, real FP32 nonpersistent RoPE buffers, and strict assign=True alias
  handling on a tiny structural stub. It did not load checkpoint weights or
  execute any forward pass. Native GPTQ/data modules were not imported.
- Full benchmark dependencies/protocol compatibility (especially FlatQuant)
  remain a separate next step after the saved-load smoke passes.

### 2026-10-15: native FlatQuant benchmark smoke preparation

- User approved native implementations and FlatQuant-first evaluation, with
  seven scoring benchmarks grouped and three generation benchmarks separate
  for full runs; the initial smoke is grouped into exactly two GPU jobs.
- Installed benchmark-only dependencies in flatquant-eval. Before/after freeze
  comparison confirms fsspec was the only existing package replaced; numerical
  runtime packages and compiled extensions remain unchanged. CPU imports and
  uv pip check passed; snapshots are under safe-quant requirements/.
- Added explicit cache-enabled protocols and general-task batch size 1. Existing
  datasets, prompts, candidates, metric formulas and uncached configs remain
  unchanged. The native runtime still rejects False requests; new runners ask
  for True explicitly, starting a fresh cache per independent scoring forward.
- CPU preflight passed for all seven scoring datasets/tasks and all prepared
  generation inputs. Saved outputs/flatquant/setup/cpu-preflight-20261015.json.
  XSTest smoke uses verified interleaved safe/unsafe rows 0 and 25; the CPU
  subset guard caught and rejected the initial 0,250 selection before any GPU
  run. Full StrongREJECT uses the existing 313-row source, not StrongREJECT-small.
- CPU regression suite: 83 tests passed. Shell syntax passed. Per-stage logs,
  explicit partial-run selection/flags and source/runtime metadata are saved.
  No GPU benchmark has been executed or submitted by the assistant.
- Generation smoke saves two responses per benchmark and runs the existing
  XSTest local string-match judge. HarmBench classifier and StrongREJECT rubric
  judging remain separate downstream stages; no paid API calls are automatic.

### Record each subsequent test or run

For each attempt, append the date, source revision/local patches, environment
changes, Slurm job ID and GPU count/type, exact command, resolved scientific
settings, output/log paths, exit status, observed result, and next action.
Record changes to scientific settings with their rationale and user agreement.
Mark stages complete only after inspecting their outputs. Keep unsuccessful
attempts and tracebacks in the record rather than overwriting them.
