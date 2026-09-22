#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-monitor
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=12:05:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/monitor-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/monitor-%j.err
set -euo pipefail
ROOT=/projects/work/yang-lab/projects/policy-finetuning/dreamzero
exec /usr/bin/python3 "${ROOT}/scripts/train/monitor_abc_yam_job.py" "${1:?GPU job ID required}" \
  --output-dir "${2:-${ROOT}/outputs/abc/h48/full_smoke_${1}}" \
  --status-file "${ROOT}/outputs/abc/h48/stats/monitor-${1}.json" \
  --hours "${MONITOR_HOURS:-12}"
