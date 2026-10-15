#!/usr/bin/env bash
# Run inside an interactive Slurm GPU allocation, or via the sbatch wrapper.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "A Slurm GPU allocation is required." >&2
    exit 2
fi
if [[ $# -ne 2 ]]; then
    echo "Usage: bash scripts/check_quantized_inference.sh {quarot|flatquant} NEW_OUTPUT_PARENT" >&2
    exit 2
fi
repository_root=/work/nvme/bgxm/wliu12/safe-quant
method=$1
output_parent=$2
case "$method" in
    quarot) environment_root=/work/nvme/bgxm/wliu12/envs/quarot ;;
    flatquant) environment_root=/work/nvme/bgxm/wliu12/envs/flatquant-eval ;;
    *) echo "Supported methods: quarot, flatquant" >&2; exit 2 ;;
esac
source "$environment_root/bin/activate"
export SAFE_QUANT_MODEL_ROOT=/work/nvme/bgxm/wliu12/models/safe-quant
export PYTHONPATH="$repository_root/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
if [[ "$method" == flatquant ]]; then
    unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
    flatquant_setup=/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup
    export TRITON_CACHE_DIR="$flatquant_setup/triton-cache"
    export TORCH_EXTENSIONS_DIR="$flatquant_setup/torch-extensions"
    export HF_MODULES_CACHE="$flatquant_setup/hf-modules"
fi
cd "$repository_root"
if [[ -e "$output_parent" ]]; then
    echo "Refusing to reuse output parent: $output_parent" >&2
    exit 2
fi
mkdir -p "$output_parent"
model_config="configs/models/llama3_8b_instruct_${method}_w4a4kv4.yaml"
# Two distinct Python processes, each loading the saved immutable checkpoint.
python -u -B scripts/check_quantized_inference.py --model-config "$model_config" --output-dir "$output_parent/capture"
python -u -B scripts/check_quantized_inference.py --model-config "$model_config" --output-dir "$output_parent/reload" --reference "$output_parent/capture"
