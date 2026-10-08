#!/usr/bin/env bash
#SBATCH --job-name=dz-h100-env
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/alex/policy-finetuning/dreamzero
#SBATCH --output=/home/alex/policy-finetuning/dreamzero/logs/environment-h100-%j.out
#SBATCH --error=/home/alex/policy-finetuning/dreamzero/logs/environment-h100-%j.err
set -euo pipefail
source env.sh
source scripts/train/abc_cuda_env.sh
python scripts/verify_environment.py
