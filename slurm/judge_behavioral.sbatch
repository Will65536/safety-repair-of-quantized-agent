#!/usr/bin/env bash
#SBATCH --job-name=safe-quant-judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
srun python scripts/judge_behavioral.py "$@"
