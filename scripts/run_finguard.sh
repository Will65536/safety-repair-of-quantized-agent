#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_finguard.sh {smoke|full} {bf16|spinquant|both} [RUN_TAG]

Run this only inside a Slurm GPU allocation. RUN_TAG defaults to
MODE-$SLURM_JOB_ID. Output directories must not already contain files.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage
    exit 2
fi

mode=$1
selection=$2
run_tag=${3:-"${mode}-${SLURM_JOB_ID:-no-job}"}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "FinGuard model evaluation must run inside a Slurm GPU allocation." >&2
    exit 1
fi

case "$mode" in
    smoke)
        evaluator_args=(--batch-size 1 --max-samples 8)
        ;;
    full)
        evaluator_args=(--batch-size 4)
        ;;
    *)
        usage
        exit 2
        ;;
esac

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

if selection in {"spinquant", "both"}:
    for package in ("fast-hadamard-transform", "torch"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{package}: not installed")

if problems:
    raise SystemExit(
        "Incompatible evaluation environment:\n  "
        + "\n  ".join(problems)
        + "\nRun the environment setup commands in README.md."
    )
PY

run_one() {
    local model_name=$1
    local model_config=$2
    local output_dir="outputs/finguard/${model_name}/${run_tag}"

    echo "Running ${model_name}: ${mode} -> ${output_dir}"
    python scripts/evaluate.py \
        --model-config "$model_config" \
        --benchmark-config configs/benchmarks/finguard.yaml \
        --output-dir "$output_dir" \
        "${evaluator_args[@]}"
}

if [[ "$selection" == bf16 || "$selection" == both ]]; then
    run_one \
        llama3_8b_instruct_bf16 \
        configs/models/llama3_8b_instruct_bf16.yaml
fi

if [[ "$selection" == spinquant || "$selection" == both ]]; then
    run_one \
        llama3_8b_instruct_spinquant_w4a4kv4 \
        configs/models/llama3_8b_instruct_spinquant_w4a4kv4.yaml
fi
