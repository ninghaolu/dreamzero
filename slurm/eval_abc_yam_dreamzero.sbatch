#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-eval
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/eval-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/eval-%j.err
# Compatibility entrypoint; implementation lives in slurm/abc/eval/eval_abc_yam_dreamzero.sbatch.
set -euo pipefail
exec bash "/projects/work/yang-lab/projects/policy-finetuning/dreamzero/slurm/abc/eval/eval_abc_yam_dreamzero.sbatch" "$@"
