#!/bin/bash
#SBATCH --job-name=spinquant-r1r2
#SBATCH --account=bgxm-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=logs/spinquant_rotation_%j.out
#SBATCH --error=logs/spinquant_rotation_%j.err

set -euo pipefail

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         $(hostname)"
echo "Start time:   $(date)"
echo "Working dir:  ${SLURM_SUBMIT_DIR}"

cd /work/nvme/bgxm/wliu12/repos/SpinQuant
mkdir -p logs

source /work/nvme/bgxm/wliu12/repos/SpinQuant/.venv/bin/activate

echo "Python: $(which python)"
python --version

nvidia-smi

# 如果此前执行过 hf auth login，且 home 目录共享，
# 通常不需要在这里重新设置 HF_TOKEN。
python - <<'PY'
from huggingface_hub import whoami
print("Hugging Face user:", whoami()["name"])
PY

bash scripts/10_optimize_rotation_single_gpu.sh

echo "End time: $(date)"