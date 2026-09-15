#!/usr/bin/env bash
# CPU-only pass1 -> reduce-pass1 -> pass2 -> finalize. Reruns reuse compatible parts.
# --dry-run prints the four submissions without submitting jobs.
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning
WORKER="${ROOT}/slurm/compute_abc_yam_dreamzero_stats.bash"
PYTHON=/scratch/nl2752/miniconda3/envs/dreamzero/bin/python
SCRIPT="${ROOT}/dreamzero/scripts/data/compute_abc_yam_stats.py"
export ABC_DZ_STATS_DATASET_ROOT="${ABC_DZ_STATS_DATASET_ROOT:-/projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train}"
export ABC_DZ_STATS_OUTPUT_DIR="${ABC_DZ_STATS_OUTPUT_DIR:-${ROOT}/dreamzero/data/abc_yam_v3_norm_stats}"
export ABC_DZ_STATS_WORK_DIR="${ABC_DZ_STATS_WORK_DIR:-${ROOT}/policy-finetuning-logs/dreamzero-abc-stats/work}"
LOG_DIR="${ABC_DZ_STATS_LOG_DIR:-${ROOT}/policy-finetuning-logs/dreamzero-abc-stats}"
CONCURRENT="${ABC_DZ_STATS_MAX_CONCURRENT:-32}"
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if (( $# )); then echo 'Usage: submit_abc_yam_dreamzero_stats.bash [--dry-run]' >&2; exit 2; fi
FILES="$("${PYTHON}" "${SCRIPT}" --stage count-files --dataset-root "${ABC_DZ_STATS_DATASET_ROOT}" --max-files "${ABC_DZ_STATS_MAX_FILES:-0}")"
DEFAULT_SHARDS=32
if (( FILES < DEFAULT_SHARDS )); then DEFAULT_SHARDS="${FILES}"; fi
export ABC_DZ_STATS_NUM_SHARDS="${ABC_DZ_STATS_NUM_SHARDS:-${DEFAULT_SHARDS}}"
if [[ ! "${CONCURRENT}" =~ ^[1-9][0-9]*$ || ! "${ABC_DZ_STATS_NUM_SHARDS}" =~ ^[1-9][0-9]*$ ]] || (( ABC_DZ_STATS_NUM_SHARDS > FILES )); then
    echo "Invalid concurrency or shard count (files=${FILES})" >&2; exit 2
fi
mkdir -p "${LOG_DIR}"
COMMON=(--parsable --output="${LOG_DIR}/stats-%A_%a.out" --error="${LOG_DIR}/stats-%A_%a.err")
# Optional site-specific CPU partition; otherwise use the cluster default.
if [[ -n "${ABC_DZ_STATS_PARTITION:-}" ]]; then COMMON+=(--partition="${ABC_DZ_STATS_PARTITION}"); fi
ARRAY="0-$((ABC_DZ_STATS_NUM_SHARDS-1))%${CONCURRENT}"
PREVIOUS=""
for STAGE in pass1 reduce-pass1 pass2 finalize; do
    export ABC_DZ_STATS_STAGE="${STAGE}"
    CMD=(sbatch "${COMMON[@]}")
    if [[ -n "${PREVIOUS}" ]]; then CMD+=(--dependency="afterok:${PREVIOUS}" --kill-on-invalid-dep=yes); fi
    if [[ "${STAGE}" == pass1 || "${STAGE}" == pass2 ]]; then CMD+=(--array="${ARRAY}"); fi
    CMD+=("${WORKER}")
    if (( DRY_RUN )); then
        printf 'ABC_DZ_STATS_STAGE=%q ' "${STAGE}"
        printf '%q ' "${CMD[@]}"
        printf '\n'
        PREVIOUS="DRY_${STAGE}"
    else
        JOB="$("${CMD[@]}")"
        PREVIOUS="${JOB%%;*}"
        printf '%s=%s\n' "${STAGE}" "${PREVIOUS}"
        printf '%s %s\n' "${STAGE}" "${PREVIOUS}" >> "${LOG_DIR}/submissions.log"
    fi
done
printf 'files=%s shards=%s concurrency=%s\noutput=%s\n' "${FILES}" "${ABC_DZ_STATS_NUM_SHARDS}" "${CONCURRENT}" "${ABC_DZ_STATS_OUTPUT_DIR}"
