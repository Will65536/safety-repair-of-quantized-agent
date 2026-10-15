#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_wildguard.sh {bf16|spinquant|both} [RUN_TAG]

Run the full frozen WildGuardTest evaluation only inside a Slurm GPU
allocation. RUN_TAG defaults to full-$SLURM_JOB_ID. When both models run,
their results are compared automatically.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
    exit 2
fi

selection=$1
run_tag=${2:-"full-${SLURM_JOB_ID:-no-job}"}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "WildGuard model evaluation must run inside a Slurm GPU allocation." >&2
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
    raise SystemExit("Invalid evaluation environment:\n  " + "\n  ".join(problems))
PY

run_one() {
    local model_name=$1
    local model_config=$2
    local output_dir="outputs/wildguard/${model_name}/${run_tag}"

    echo "Running ${model_name}: full WildGuardTest -> ${output_dir}"
    python scripts/evaluate_wildguard.py \
        --model-config "$model_config" \
        --benchmark-config configs/benchmarks/wildguard_prompt.yaml \
        --output-dir "$output_dir" \
        --batch-size 2
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

if [[ "$selection" == both ]]; then
    python scripts/compare_wildguard.py \
        "outputs/wildguard/llama3_8b_instruct_bf16/${run_tag}" \
        "outputs/wildguard/llama3_8b_instruct_spinquant_w4a4kv4/${run_tag}" \
        --output "outputs/wildguard/comparisons/${run_tag}.json"
fi
