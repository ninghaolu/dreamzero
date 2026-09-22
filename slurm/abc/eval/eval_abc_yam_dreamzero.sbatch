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
set -euo pipefail
export ABC_TEMPORAL="${ABC_TEMPORAL:-h48}"
source /projects/work/yang-lab/projects/policy-finetuning/envs/dreamzero/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec python scripts/eval_abc_yam.py \
  --metadata-root "${ABC_METADATA_ROOT:-data/abc/${ABC_TEMPORAL}/val}" --temporal "${ABC_TEMPORAL}" \
  --anchors "${ABC_EVAL_ANCHORS:-outputs/abc/${ABC_TEMPORAL}/validation/anchors.json}" \
  --checkpoint "${ABC_CHECKPOINT_PATH:?Set ABC_CHECKPOINT_PATH to the saved model}" \
  --num-samples "${ABC_EVAL_SAMPLES:-2}" \
  --output-dir "outputs/abc/${ABC_TEMPORAL}/evaluation-${SLURM_JOB_ID}"
