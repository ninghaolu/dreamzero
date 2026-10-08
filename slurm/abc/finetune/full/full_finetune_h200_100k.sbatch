#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-h48-full-100k
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=1024G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/home/alex/policy-finetuning/dreamzero
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/full-h100-%j.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/full-h100-%j.err
# Compatibility path; this cluster uses eight H100s.
set -euo pipefail
exec bash "${DREAMZERO_REPO_ROOT:-/home/alex/policy-finetuning/dreamzero}/slurm/abc/finetune/full/full_finetune_h100_100k.sbatch" "$@"
