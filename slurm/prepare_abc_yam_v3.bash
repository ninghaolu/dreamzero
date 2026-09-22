#!/usr/bin/env bash
#SBATCH --job-name=dz-abc-index
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
# Compatibility entrypoint; implementation lives in slurm/abc/stats/prepare_abc_yam_v3.bash.
set -euo pipefail
exec bash "/projects/work/yang-lab/projects/policy-finetuning/dreamzero/slurm/abc/stats/prepare_abc_yam_v3.bash" "$@"
