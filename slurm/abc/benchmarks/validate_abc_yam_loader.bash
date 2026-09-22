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
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
source /projects/work/yang-lab/projects/policy-finetuning/envs/dreamzero/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
ABC_VALIDATE_ONLY=1 bash scripts/train/abc_yam_v3_training.sh
exec torchrun --standalone --nproc_per_node=2 scripts/data/inspect_abc_yam_sample.py \
  --metadata-root "${ABC_METADATA_ROOT:-data/abc/${ABC_TEMPORAL}/train}" --temporal "${ABC_TEMPORAL}" \
  --output-dir "outputs/abc/${ABC_TEMPORAL}/inspection-${SLURM_JOB_ID}" \
  --episode-ids 0 1 --workers 2 --samples 8
