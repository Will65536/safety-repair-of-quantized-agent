#!/usr/bin/env bash
# AdvBench DirectRequest only; no attacks, calibration, or paid judges.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "AdvBench generation requires a Slurm GPU allocation." >&2
    exit 2
fi
if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: bash scripts/run_advbench.sh {bf16|flatquant|spinquant} {smoke|full} [RUN_TAG] [BATCH_SIZE]" >&2
    exit 2
fi
selection=$1
mode=$2
run_tag=${3:-"${SLURM_JOB_ID}"}
batch_size=${4:-1}
case "$selection" in bf16|flatquant|spinquant) ;; *) echo "Invalid model: $selection" >&2; exit 2 ;; esac
case "$mode" in smoke|full) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer." >&2
    exit 2
fi
if [[ ! "$run_tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "RUN_TAG must be a simple name, not a path." >&2
    exit 2
fi
repository_root=/work/nvme/bgxm/wliu12/safe-quant
cd "$repository_root"
model_config="configs/models/llama3_8b_instruct_${selection}_w4a4kv4.yaml"
if [[ "$selection" == bf16 ]]; then
    model_config=configs/models/llama3_8b_instruct_bf16.yaml
fi
if [[ "$selection" == flatquant ]]; then
    source /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/activate
    flatquant_setup=/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup
    export TRITON_CACHE_DIR="$flatquant_setup/triton-cache"
    export TORCH_EXTENSIONS_DIR="$flatquant_setup/torch-extensions"
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
else
    source /work/nvme/bgxm/wliu12/repos/SpinQuant/.venv/bin/activate
    if [[ "$selection" == spinquant ]]; then
        export SPINQUANT_ROOT=/work/nvme/bgxm/wliu12/repos/SpinQuant
    fi
fi
export SAFE_QUANT_MODEL_ROOT=/work/nvme/bgxm/wliu12/models/safe-quant
export PYTHONPATH="$repository_root/src"
if [[ "$selection" == spinquant ]]; then
    export PYTHONPATH="$PYTHONPATH:$SPINQUANT_ROOT"
fi
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
output_root="outputs/behavioral/advbench_directrequest/${selection}/${mode}-${run_tag}"
if [[ -e "$output_root" ]]; then
    echo "Refusing to reuse output directory: $output_root" >&2
    exit 2
fi
mkdir -p "$output_root/logs"
run_step() {
    local name=$1
    shift
    "$@" 2>&1 | tee "$output_root/logs/$name.log"
}
run_step preflight python -u -B - "$selection" <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys
expected = {
    "torch": "2.7.1+cu128",
    "transformers": "4.45.0" if sys.argv[1] == "flatquant" else "4.44.2",
    "accelerate": "0.34.2",
}
found = {name: version(name) for name in expected}
if found != expected:
    raise RuntimeError(f"Unexpected native environment: {found}; expected {expected}")
report = {"status": "passed", "versions": found, "model_inference": "not_run"}
if sys.argv[1] == "spinquant":
    root = Path(os.environ["SPINQUANT_ROOT"]).resolve()
    for package in ("eval_utils", "utils"):
        spec = find_spec(package)
        locations = [] if spec is None else list(spec.submodule_search_locations or [])
        if root / package not in [Path(path).resolve() for path in locations]:
            raise RuntimeError(f"SpinQuant package {package} is not resolved from {root}; check PYTHONPATH")
    report["spinquant_import_paths"] = "passed (module discovery only; no model or kernel execution)"
print(json.dumps(report))
PY
subset=()
if [[ "$mode" == smoke ]]; then
    subset=(--sample-indices 0,74,148,222,296,370,444,519)
fi
run_step generation python -u -B scripts/evaluate_behavioral.py \
    --model-config "$model_config" \
    --benchmark-config configs/benchmarks/advbench_directrequest_cached_256.yaml \
    --prompts outputs/behavioral/prompts/advbench_directrequest-cached-256-v1/prompts.jsonl \
    --batch-size "$batch_size" --output-dir "$output_root/generation" "${subset[@]}"
run_step judge python -u -B scripts/judge_behavioral.py \
    --responses "$output_root/generation/responses.jsonl" \
    --backend advbench_original_strmatch --output-dir "$output_root/judgments/strmatch"
echo "Completed $selection $mode -> $output_root (non-refusal proxy, not semantic harmful compliance)"
