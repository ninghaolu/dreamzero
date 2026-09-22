#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-loader
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/distributed-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/policy-finetuning-logs/dreamzero-abc-loader/distributed-%j.err
# Compatibility entrypoint; implementation lives in slurm/abc/benchmarks/validate_abc_yam_loader.bash.
set -euo pipefail
exec bash "/projects/work/yang-lab/projects/policy-finetuning/dreamzero/slurm/abc/benchmarks/validate_abc_yam_loader.bash" "$@"
