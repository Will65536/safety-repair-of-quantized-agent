#!/usr/bin/env bash
# Native FlatQuant only. Smoke groups: scoring and generation (two GPU jobs).
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "FlatQuant benchmark execution requires a Slurm GPU allocation." >&2
    exit 2
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/run_flatquant_bench.sh {general|finguard|wildguard|scoring|generation|harmbench|xstest|strongreject} {smoke|full} [RUN_TAG]" >&2
    exit 2
fi
group=$1
mode=$2
run_tag=${3:-"${SLURM_JOB_ID}"}
case "$group" in general|finguard|wildguard|scoring|generation|harmbench|xstest|strongreject) ;; *) echo "Invalid group: $group" >&2; exit 2 ;; esac
case "$mode" in smoke|full) ;; *) echo "Invalid mode: $mode" >&2; exit 2 ;; esac
if [[ "$group" == generation && "$mode" == full ]]; then
    echo "Full generation runs must be separate: harmbench, xstest, strongreject." >&2
    exit 2
fi
if [[ ! "$run_tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
    echo "RUN_TAG must be a simple name, not a path." >&2
    exit 2
fi
repository_root=/work/nvme/bgxm/wliu12/safe-quant
flatquant_setup=/work/nvme/bgxm/wliu12/repos/FlatQuant/local_setup
source /work/nvme/bgxm/wliu12/envs/flatquant-eval/bin/activate
export SAFE_QUANT_MODEL_ROOT=/work/nvme/bgxm/wliu12/models/safe-quant
export PYTHONPATH="$repository_root/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/u/wliu12/.cache/huggingface/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/u/wliu12/.cache/huggingface/hub}"
# PIQA/Winogrande require the already-cached dataset Python modules. The
# published model is imported directly by runtime_support, not via this cache.
export HF_MODULES_CACHE=/u/wliu12/.cache/huggingface/modules
export TRITON_CACHE_DIR="$flatquant_setup/triton-cache"
export TORCH_EXTENSIONS_DIR="$flatquant_setup/torch-extensions"
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH LD_LIBRARY_PATH
cd "$repository_root"
output_root="outputs/flatquant/${group}/${mode}-${run_tag}"
if [[ -e "$output_root" ]]; then
    echo "Refusing to reuse output directory: $output_root" >&2
    exit 2
fi
mkdir -p "$output_root/logs"
model_config=configs/models/llama3_8b_instruct_flatquant_w4a4kv4.yaml
run_step() {
    local name=$1
    shift
    echo "Starting $name -> $output_root/$name"
    "$@" 2>&1 | tee "$output_root/logs/$name.log"
}
preflight_group=generation
case "$group" in scoring|general|finguard|wildguard) preflight_group=$group ;; esac
run_step preflight python -u -B scripts/check_flatquant_bench.py \
    --group "$preflight_group" --output-json "$output_root/preflight.json"
# Each smoke job is independently gated; no third GPU job is required.
if [[ "$mode" == smoke ]]; then
    run_step inference bash scripts/check_quantized_inference.sh flatquant "$output_root/inference"
fi
if [[ "$group" == scoring || "$group" == general || "$group" == finguard || "$group" == wildguard ]]; then
    wiki_args=()
    general_args=()
    classifier_args=()
    if [[ "$mode" == smoke ]]; then
        wiki_args=(--max-blocks 2)
        general_args=(--limit 2)
        classifier_args=(--max-samples 4)
    fi
    if [[ "$group" == scoring ]]; then
        run_step wikitext2 python -u -B scripts/evaluate_wikitext2.py \
            --model-config "$model_config" --benchmark-config configs/benchmarks/wikitext2_cached.yaml \
            --output-dir "$output_root/wikitext2" "${wiki_args[@]}"
    fi
    if [[ "$group" == scoring || "$group" == general ]]; then
        run_step general_4task python -u -B scripts/evaluate_general.py \
            --model-config "$model_config" --benchmark-config configs/benchmarks/general_4task_cached_bs1.yaml \
            --output-dir "$output_root/general_4task" "${general_args[@]}"
    fi
    if [[ "$group" == scoring || "$group" == finguard ]]; then
        run_step finguard python -u -B scripts/evaluate.py \
            --model-config "$model_config" --benchmark-config configs/benchmarks/finguard_cached.yaml \
            --batch-size 4 --output-dir "$output_root/finguard" "${classifier_args[@]}"
    fi
    if [[ "$group" == scoring || "$group" == wildguard ]]; then
        run_step wildguard python -u -B scripts/evaluate_wildguard.py \
            --model-config "$model_config" --benchmark-config configs/benchmarks/wildguard_prompt_cached.yaml \
            --batch-size 2 --output-dir "$output_root/wildguard" "${classifier_args[@]}"
    fi
else
    run_generation() {
        local name=$1
        local config=$2
        local prompts=$3
        local indices=$4
        local selection=()
        if [[ "$mode" == smoke ]]; then selection=(--sample-indices "$indices"); fi
        run_step "$name" python -u -B scripts/evaluate_behavioral.py \
            --model-config "$model_config" --benchmark-config "$config" --prompts "$prompts" \
            --output-dir "$output_root/$name" "${selection[@]}"
    }
    if [[ "$group" == generation || "$group" == harmbench ]]; then
        run_generation harmbench configs/benchmarks/harmbench_directrequest_cached_256.yaml \
            outputs/behavioral/prompts/harmbench_directrequest-cached-256-v1/prompts.jsonl 0,1
    fi
    if [[ "$group" == generation || "$group" == xstest ]]; then
        run_generation xstest configs/benchmarks/xstest_cached_256.yaml \
            outputs/behavioral/prompts/xstest-cached-256-v1/prompts.jsonl 0,25
        # This existing evaluator is CPU-only and requires no external API.
        run_step xstest_judge python -u -B scripts/judge_behavioral.py \
            --responses "$output_root/xstest/responses.jsonl" --backend xstest_official_strmatch \
            --output-dir "$output_root/xstest/judgments/strmatch"
    fi
    if [[ "$group" == generation || "$group" == strongreject ]]; then
        run_generation strongreject configs/benchmarks/strongreject_cached_256.yaml \
            outputs/behavioral/prompts/strongreject-cached-256-v1/prompts.jsonl 0,156
    fi
    echo "HarmBench and StrongREJECT responses are saved; their existing judge stages are separate. No paid API judge was called."
fi
echo "Completed $group $mode -> $output_root (smoke subsets are not full benchmark results)"
