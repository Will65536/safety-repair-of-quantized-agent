#!/usr/bin/env bash
#SBATCH --job-name=quantized-load-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:15:00

set -euo pipefail
cd /work/nvme/bgxm/wliu12/safe-quant
srun bash scripts/check_quantized_inference.sh "$@"
