#!/usr/bin/env bash
#SBATCH --job-name=safe-quant-behavioral
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
repository_root=/work/nvme/bgxm/wliu12/safe-quant
spinquant_root=/work/nvme/bgxm/wliu12/repos/SpinQuant
environment_root=/work/nvme/bgxm/wliu12/repos/SpinQuant/.venv
source "$environment_root/bin/activate"
export SPINQUANT_ROOT="$spinquant_root"
export SAFE_QUANT_MODEL_ROOT=/work/nvme/bgxm/wliu12/models/safe-quant
export PYTHONPATH="$repository_root/src:$spinquant_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
cd "$repository_root"

if [[ $# -lt 4 ]]; then
    echo "Usage: sbatch slurm/evaluate_behavioral.sbatch MODEL_CONFIG BENCHMARK_CONFIG PROMPTS_JSONL OUTPUT_DIR [EVALUATOR_ARGS...]" >&2
    exit 2
fi
model_config=$1
benchmark_config=$2
prompts=$3
output_dir=$4
shift 4
srun python scripts/evaluate_behavioral.py \
    --model-config "$model_config" \
    --benchmark-config "$benchmark_config" \
    --prompts "$prompts" \
    --output-dir "$output_dir" "$@"
