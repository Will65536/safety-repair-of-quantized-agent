#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_behavioral_smoke.sh {bf16|spinquant|both} [RUN_TAG]

Run the representative 256-token HarmBench, StrongREJECT-small, and XSTest
smoke tests directly inside an existing interactive Slurm GPU allocation.
RUN_TAG defaults to cached-256-smoke-$SLURM_JOB_ID.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

selection=$1
run_tag=${2:-"cached-256-smoke-${SLURM_JOB_ID:-no-job}"}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Behavioral model evaluation must run inside a Slurm GPU allocation." >&2
    exit 1
fi

case "$selection" in
    bf16|spinquant|both) ;;
    *)
        usage
        exit 2
        ;;
esac

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export SAFE_QUANT_MODEL_ROOT=${SAFE_QUANT_MODEL_ROOT:-/work/nvme/bgxm/wliu12/models/safe-quant}
export SPINQUANT_ROOT=${SPINQUANT_ROOT:-/work/nvme/bgxm/wliu12/repos/SpinQuant}
export PYTHONPATH="${project_root}/src:${SPINQUANT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false

cd "$project_root"

python - "$selection" <<'PY'
import importlib.metadata
import sys

import torch

selection = sys.argv[1]
expected = {
    "accelerate": "0.34.2",
    "datasets": "2.20.0",
    "transformers": "4.44.2",
}
problems = []
for package, wanted in expected.items():
    try:
        found = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        found = None
    if found != wanted:
        problems.append(f"{package}: expected {wanted}, found {found}")

if not torch.cuda.is_available():
    problems.append("CUDA GPU is not visible in this Slurm allocation")

if selection in {"spinquant", "both"}:
    try:
        importlib.metadata.version("fast-hadamard-transform")
    except importlib.metadata.PackageNotFoundError:
        problems.append("fast-hadamard-transform: not installed")

if problems:
    raise SystemExit("Invalid evaluation environment:\n  " + "\n  ".join(problems))
PY

run_benchmark() {
    local model_dir=$1
    local model_config=$2
    local benchmark_dir=$3
    local benchmark_config=$4
    local prompts=$5
    local indices=$6
    local output_dir="outputs/behavioral/${benchmark_dir}/${model_dir}/${run_tag}"

    echo "Running ${model_dir}: ${benchmark_dir} -> ${output_dir}"
    python scripts/evaluate_behavioral.py \
        --model-config "$model_config" \
        --benchmark-config "$benchmark_config" \
        --prompts "$prompts" \
        --output-dir "$output_dir" \
        --sample-indices "$indices"
}

run_model() {
    local model_dir=$1
    local model_config=$2

    run_benchmark \
        "$model_dir" "$model_config" \
        harmbench_directrequest \
        configs/benchmarks/harmbench_directrequest_cached_256.yaml \
        outputs/behavioral/prompts/harmbench_directrequest-cached-256-v1/prompts.jsonl \
        0,1,2,3
    run_benchmark \
        "$model_dir" "$model_config" \
        strongreject_small \
        configs/benchmarks/strongreject_small_cached_256.yaml \
        outputs/behavioral/prompts/strongreject_small-cached-256-v1/prompts.jsonl \
        0,10,20,30,40,50
    run_benchmark \
        "$model_dir" "$model_config" \
        xstest \
        configs/benchmarks/xstest_cached_256.yaml \
        outputs/behavioral/prompts/xstest-cached-256-v1/prompts.jsonl \
        0,25,50,75,100,125,150,175
}

if [[ "$selection" == bf16 || "$selection" == both ]]; then
    run_model bf16 configs/models/llama3_8b_instruct_bf16.yaml
fi

if [[ "$selection" == spinquant || "$selection" == both ]]; then
    run_model spinquant configs/models/llama3_8b_instruct_spinquant_w4a4kv4.yaml
fi

if [[ "$selection" == both ]]; then
    python scripts/estimate_behavioral_runtime.py \
        "outputs/behavioral/harmbench_directrequest/bf16/${run_tag}" \
        "outputs/behavioral/harmbench_directrequest/spinquant/${run_tag}" \
        "outputs/behavioral/strongreject_small/bf16/${run_tag}" \
        "outputs/behavioral/strongreject_small/spinquant/${run_tag}" \
        "outputs/behavioral/xstest/bf16/${run_tag}" \
        "outputs/behavioral/xstest/spinquant/${run_tag}"
fi
