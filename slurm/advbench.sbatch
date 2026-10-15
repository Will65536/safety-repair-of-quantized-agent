#!/usr/bin/env bash
#SBATCH --job-name=advbench-direct
#SBATCH --account=bgxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=advbench-%j.out
#SBATCH --error=advbench-%j.err

set -euo pipefail
cd /work/nvme/bgxm/wliu12/safe-quant
srun bash scripts/run_advbench.sh "$@"
