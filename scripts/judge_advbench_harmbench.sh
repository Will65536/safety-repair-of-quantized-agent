#!/usr/bin/env bash
# Rejudge cached AdvBench responses only. Never regenerate or submit a job.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "HarmBench classifier judging requires a Slurm GPU allocation." >&2
    exit 2
fi
if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/judge_advbench_harmbench.sh {bf16|flatquant|spinquant} [GENERATION_RUN_TAG] [JUDGE_TAG]" >&2
    exit 2
fi
selection=$1
run_tag=${2:-full-bs4-run1}
judge_tag=${3:-harmbench-run1}
case "$selection" in bf16|flatquant|spinquant) ;; *) echo "Invalid model: $selection" >&2; exit 2 ;; esac
for tag in "$run_tag" "$judge_tag"; do
    if [[ ! "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
        echo "Run and judge tags must be simple names, not paths." >&2
        exit 2
    fi
done
repository_root=/work/nvme/bgxm/wliu12/safe-quant
cd "$repository_root"
run_root="$repository_root/outputs/behavioral/advbench_directrequest/$selection/$run_tag"
responses="$run_root/generation/responses.jsonl"
output_dir="$run_root/judgments/$judge_tag"
log_file="$run_root/logs/judge-$judge_tag.log"
if [[ ! -f "$responses" ]]; then
    echo "Saved responses missing: $responses" >&2
    exit 2
fi
if [[ -e "$output_dir" || -e "$log_file" ]]; then
    echo "Refusing to overwrite judge outputs; choose a fresh JUDGE_TAG." >&2
    exit 2
fi
source /work/nvme/bgxm/wliu12/venvs/harmbench-judge/bin/activate
export PYTHONPATH="$repository_root/src"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
classifier_revision=bda705349d1144fa618770bea64d99ce54e3835b
classifier_path="/work/nvme/bgxm/wliu12/hf-cache/hub/models--cais--HarmBench-Llama-2-13b-cls/snapshots/$classifier_revision"
mkdir -p "$run_root/logs"
python -u -B scripts/judge_behavioral.py \
    --responses "$responses" --backend advbench_harmbench_classifier \
    --output-dir "$output_dir" \
    --harmbench-root /work/nvme/bgxm/wliu12/repos/HarmBench \
    --classifier-path "$classifier_path" --classifier-revision "$classifier_revision" \
    2>&1 | tee "$log_file"
