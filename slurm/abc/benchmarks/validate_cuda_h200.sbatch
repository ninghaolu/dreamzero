#!/usr/bin/env bash
#SBATCH --job-name=dz-cpuadam-check
#SBATCH --account=torch_pr_147_courant
#SBATCH --partition=h200_public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/cuda-check-%j.out
#SBATCH --error=/projects/work/yang-lab/projects/policy-finetuning/dreamzero/outputs/abc/h48/stats/cuda-check-%j.err
set -euo pipefail
source /projects/work/yang-lab/projects/policy-finetuning/envs/dreamzero/activate.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
source scripts/train/abc_cuda_env.sh
python scripts/train/validate_abc_cuda.py
