# Local H100 setup

Validated on this cluster on 2026-09-15.

## Activate

```bash
cd /home/alex/policy-finetuning/dreamzero
source env.sh
```

The isolated `.venv` uses Python 3.11.16, PyTorch 2.8.0+cu129,
FlashAttention 2.8.3.post1, and DeepSpeed 0.16.9. `env.sh` sets writable
caches, the local CUDA 12.9 compiler, and these checkpoint variables:

| Variable | Directory |
| --- | --- |
| `AGIBOT_CKPT_DIR` | `/projects/data/cloud_user/checkpoints/DreamZero-AgiBot` |
| `WAN_CKPT_DIR` | `/projects/data/cloud_user/checkpoints/Wan2.1-I2V-14B-480P` |
| `TOKENIZER_DIR` | `/projects/data/cloud_user/checkpoints/umt5-xxl` |

The downloads total 128,142,750,388 bytes. UMT5 contains tokenizer files;
its model weights are unnecessary. Each directory includes
`dreamzero_download_manifest.json` recording its Hugging Face revision.
File counts, byte totals, and checkpoint shard references were checked.

DeepSpeed is pinned because 0.19.6 caused a circular import with the
repository's Transformers 4.51.3 pin. The cluster has no system `nvcc`,
so CUDA compiler packages live in `.cache/cuda`. Preserve `.cache/python`
and `.cache/cuda`: these contain installed runtimes, not just download caches.

## Validation

- All 11 existing repository tests passed.
- `uv pip check` reported compatible dependencies.
- `scripts/verify_environment.py` passed model/training imports and a
  FlashAttention forward/backward check against PyTorch attention on an H100.
- Full model training has not been launched.

```bash
python -m unittest discover -s tests
srun --partition=defq --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  --time=00:10:00 bash -lc \
  'cd /home/alex/policy-finetuning/dreamzero && source env.sh && python scripts/verify_environment.py'
```

Validation logs are `logs/environment-tests.log` and
`logs/environment-gpu-check.log`.

## Normalization statistics

The CPU pipeline processed the full training dataset at
`/projects/data/datasets/abc_130k_v3_train` with 32 shards/workers.
All 66 tasks completed successfully:

| Stage | Slurm job |
| --- | --- |
| Pass 1, 32 shards | 1742 |
| Reduce pass 1 | 1743 |
| Pass 2, 32 shards | 1744 |
| Finalize | 1776 |

Outputs are in `data/abc_yam_v3_norm_stats/`:

- `meta/stats.json`: source training state statistics.
- `meta/relative_stats_dreamzero.json`: DreamZero relative action statistics.
- `norm_stats_manifest.json`: dataset fingerprint, coverage, and output hashes.

Coverage: 742 files, 382,468,339 rows, 370,062,739 valid four-block windows.
Quantiles use 5,000 histogram bins; their error bounds are in the manifest.
Output hashes and the source fingerprint were verified. Logs and resumable
intermediate parts are in `logs/abc-stats/`.

The completed run used `.cache/setup-venv/bin/python`, a small NumPy/PyArrow
environment, so it could run while GPU dependencies installed. Future runs
default to `.venv/bin/python`:

```bash
ABC_DZ_STATS_PARTITION=defq bash slurm/submit_abc_yam_dreamzero_stats.bash --dry-run
# Remove --dry-run to submit. Compatible intermediate parts are reused.
```

## Recreate the environment

`requirements-h100.txt` records the installed Python packages and official
FlashAttention wheel URL. `cuda-h100-explicit.txt` records the compiler packages.
From the repository root, on Linux x86_64 with `uv` and `conda` installed:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export TMPDIR="$PWD/.cache/tmp"
mkdir -p "$TMPDIR"
uv venv --python 3.11.16 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  -r requirements-h100.txt
CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs" conda create --yes \
  --prefix "$PWD/.cache/cuda" --file cuda-h100-explicit.txt
source env.sh
```

To resume or repeat the checkpoint download:

```bash
python scripts/download_training_checkpoints.py \
  --destination /projects/data/cloud_user/checkpoints
```
