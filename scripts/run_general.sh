#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: scripts/run_general.sh {smoke|full} {bf16|spinquant|both} [RUN_TAG]

Run WikiText-2 perplexity and the frozen lm-eval four-task suite inside an
interactive Slurm GPU allocation. RUN_TAG defaults to MODE-$SLURM_JOB_ID.
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
    echo "General model evaluation must run inside a Slurm GPU allocation." >&2
    exit 1
fi

case "$mode" in
    smoke|full) ;;
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
    "lm-eval": "0.4.8",
    "peft": "0.13.2",
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

wikitext_args=()
harness_args=()
if [[ "$mode" == smoke ]]; then
    wikitext_args=(--max-blocks 2)
    harness_args=(--limit 8)
fi

run_one() {
    local model_name=$1
    local model_config=$2
    local output_root="outputs/general/${model_name}/${run_tag}"

    echo "Running ${model_name}: WikiText-2 ${mode} -> ${output_root}/wikitext2"
    python scripts/evaluate_wikitext2.py \
        --model-config "$model_config" \
        --output-dir "${output_root}/wikitext2" \
        "${wikitext_args[@]}"

    echo "Running ${model_name}: four general tasks ${mode} -> ${output_root}/lm_eval_4task"
    python scripts/evaluate_general.py \
        --model-config "$model_config" \
        --output-dir "${output_root}/lm_eval_4task" \
        "${harness_args[@]}"
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
    python scripts/compare_general.py \
        "outputs/general/llama3_8b_instruct_bf16/${run_tag}" \
        "outputs/general/llama3_8b_instruct_spinquant_w4a4kv4/${run_tag}" \
        --output "outputs/general/comparisons/${run_tag}.json"
fi
