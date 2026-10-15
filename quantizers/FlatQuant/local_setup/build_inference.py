"""Build only the checkpoint's inference extension; no calibration dependencies.

Run from any directory with build_ext --build-lib local_setup/runtime.
Only CPU compilation is performed. CUDA architecture is explicit: A100/sm80.
"""
import os
from pathlib import Path
import shutil
import sysconfig

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-12.9")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
os.environ.setdefault("MAX_JOBS", "2")

from setuptools import setup
from torch.utils import cpp_extension

SETUP = Path(__file__).resolve().parent
ROOT = SETUP.parent
MODEL = Path("/work/nvme/bgxm/wliu12/models/safe-quant/llama-3-8b-instruct/flatquant-w4a4kv4")
DEPLOY = MODEL / "deploy"
RUNTIME = SETUP / "runtime"
CUTLASS = ROOT / "third-party/cutlass"
NVIDIA = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
# Ignore module-provided CUDA 13.2 includes; use the environment's CUDA 12.8
# runtime/CCCL headers with the system 12.9 compiler (same CUDA major).
for variable in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"):
    os.environ.pop(variable, None)
CUDA_LIB = SETUP / "cuda-lib"
CUDA_LIB.mkdir(exist_ok=True)
runtime_link = CUDA_LIB / "libcudart.so"
if not runtime_link.exists():
    runtime_link.symlink_to(NVIDIA / "cuda_runtime/lib/libcudart.so.12")

if not (CUTLASS / "include/cutlass/cutlass.h").is_file():
    raise RuntimeError("Initialize the pinned third-party/cutlass submodule first")
for name in ("bindings.cpp", "gemm.cu", "quant.cu", "flashinfer.cu"):
    if not (DEPLOY / "kernels" / name).is_file():
        raise RuntimeError("Run download_checkpoint.py to obtain the checkpoint's kernel sources")

# Keep published code/weights unchanged; build a separate matching runtime.
shutil.copytree(DEPLOY, RUNTIME / "deploy", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("*.so", "*.pyc", "__pycache__"))
for flag in ("-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
             "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"):
    if flag in cpp_extension.COMMON_NVCC_FLAGS:
        cpp_extension.COMMON_NVCC_FLAGS.remove(flag)

setup(
    name="flatquant-inference-kernels",
    version="0.0.0",
    packages=[],
    ext_modules=[cpp_extension.CUDAExtension(
        name="deploy._CUDA",
        sources=[str(DEPLOY / "kernels" / name) for name in
                 ("bindings.cpp", "gemm.cu", "quant.cu", "flashinfer.cu")],
        include_dirs=[str(DEPLOY / "kernels/include"), str(CUTLASS / "include"),
                      str(CUTLASS / "tools/util/include"),
                      str(NVIDIA / "cuda_runtime/include"), str(NVIDIA / "cuda_cccl/include")],
        library_dirs=[str(CUDA_LIB)],
        extra_compile_args={"cxx": ["-O2"], "nvcc": ["-O2", "-gencode=arch=compute_80,code=sm_80"]},
    )],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
