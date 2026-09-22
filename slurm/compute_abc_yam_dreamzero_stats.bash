#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-stats
#SBATCH --chdir=/projects/work/yang-lab/projects/policy-finetuning
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
# Compatibility entrypoint; implementation lives in slurm/abc/stats/compute_abc_yam_dreamzero_stats.bash.
set -euo pipefail
exec bash "/projects/work/yang-lab/projects/policy-finetuning/dreamzero/slurm/abc/stats/compute_abc_yam_dreamzero_stats.bash" "$@"
