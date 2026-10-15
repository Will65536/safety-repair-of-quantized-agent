#!/usr/bin/env bash
set -euo pipefail
cd /work/nvme/bgxm/wliu12/repos/FlatQuant
source /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/activate
export CUDA_VISIBLE_DEVICES=''
export CUDA_HOME=/usr/local/cuda-12.9
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=2
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PATH="$VIRTUAL_ENV/bin:$CUDA_HOME/bin:/usr/bin:/bin"
unset PYTHONPATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
python -B local_setup/build_inference.py build_ext \
    --build-temp local_setup/build --build-lib local_setup/runtime
