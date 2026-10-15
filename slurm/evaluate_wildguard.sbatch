#!/usr/bin/env bash
#SBATCH --job-name=safe-quant-wildguard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: sbatch slurm/evaluate_wildguard.sbatch MODEL_CONFIG BENCHMARK_CONFIG OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
    exit 2
fi

model_config=$1
benchmark_config=$2
output_dir=$3
shift 3

if [[ -n "${SPINQUANT_ROOT:-}" ]]; then
    export PYTHONPATH="${SPINQUANT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

srun python scripts/evaluate_wildguard.py \
    --model-config "${model_config}" \
    --benchmark-config "${benchmark_config}" \
    --output-dir "${output_dir}" \
    "$@"
