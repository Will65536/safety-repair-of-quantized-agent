#!/usr/bin/env bash
#SBATCH --job-name=flatquant-bench
#SBATCH --account=bgxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=flatquant-%j.out
#SBATCH --error=flatquant-%j.err

set -euo pipefail
cd /work/nvme/bgxm/wliu12/safe-quant
srun bash scripts/run_flatquant_bench.sh "$@"
