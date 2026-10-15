#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: scripts/run_wildguard_train_validation.sh [RUN_TAG]" >&2
    exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "WildGuard model evaluation must run inside a Slurm GPU allocation." >&2
    exit 1
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run_tag=${1:-"train-256-seed0-${SLURM_JOB_ID}"}

export SAFE_QUANT_MODEL_ROOT=${SAFE_QUANT_MODEL_ROOT:-/work/nvme/bgxm/wliu12/models/safe-quant}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "$project_root"

python - <<'PY'
import importlib.metadata
import torch

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
    problems.append("PyTorch cannot see a CUDA GPU")
if problems:
    raise SystemExit("Invalid evaluation environment:\n  " + "\n  ".join(problems))
PY

output_dir="outputs/wildguard_validation/llama3_8b_instruct_bf16/${run_tag}"
echo "Running frozen BF16 WildGuardTrain validation -> ${output_dir}"
python scripts/evaluate_wildguard.py \
    --model-config configs/models/llama3_8b_instruct_bf16.yaml \
    --benchmark-config configs/benchmarks/wildguard_train_validation.yaml \
    --output-dir "$output_dir" \
    --batch-size 2

