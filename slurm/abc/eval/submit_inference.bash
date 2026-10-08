#!/usr/bin/env bash
# Submit the same seeded validation samples for each supplied full checkpoint.
set -euo pipefail
ROOT="${DREAMZERO_REPO_ROOT:-/home/alex/policy-finetuning/dreamzero}"
cd "${ROOT}"
RUN="${ABC_CHECKPOINT_ROOT:-/projects/data/cloud_user/checkpoints/dreamzero_abc130k/full_h48_5491}"
OUTPUT="${ABC_INFERENCE_OUTPUT:-${ROOT}/outputs/abc/h48/inference-$(date +%Y%m%d-%H%M%S)}"
if [[ $# == 0 ]]; then set -- "${RUN}/checkpoint-15000" "${RUN}/checkpoint-16000" "${RUN}/checkpoint-17000"; fi
mkdir -p "${ROOT}/logs" "${OUTPUT}"
for checkpoint in "$@"; do
  job=$(sbatch --parsable "${ROOT}/slurm/abc/eval/infer_abc_yam_h100.sbatch" \
    "${checkpoint}" "${OUTPUT}/$(basename "${checkpoint}")")
  printf '%s\t%s\t%s\n' "${job}" "${checkpoint}" "${OUTPUT}/$(basename "${checkpoint}")" | tee -a "${OUTPUT}/jobs.tsv"
done
