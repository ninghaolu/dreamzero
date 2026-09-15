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
set -euo pipefail
source /projects/work/yang-lab/projects/policy-finetuning/envs/dreamzero/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec torchrun --standalone --nproc_per_node=2 scripts/data/inspect_abc_yam_sample.py \
  --metadata-root data/abc_yam_v3_train \
  --output-dir "../policy-finetuning-logs/dreamzero-abc-loader/distributed-${SLURM_JOB_ID}" \
  --episode-ids 0 1 --workers 2 --samples 8
