#!/usr/bin/env bash
#SBATCH --job-name=safe-quant-finguard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch slurm/evaluate.sbatch MODEL_CONFIG OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
    exit 2
fi

model_config=$1
output_dir=$2
shift 2

if [[ -n "${SPINQUANT_ROOT:-}" ]]; then
    export PYTHONPATH="${SPINQUANT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

srun python scripts/evaluate.py \
    --model-config "${model_config}" \
    --benchmark-config configs/benchmarks/finguard.yaml \
    --output-dir "${output_dir}" \
    "$@"
