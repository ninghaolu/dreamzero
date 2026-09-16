# Task-specific ABC/YAM LoRA finetuning

These scripts wrap the existing `scripts/train/abc_yam_v3_training.sh` and
`slurm/finetune_abc_yam_dreamzero.sbatch`. The model, action representation,
checkpoint/resume behavior, and W&B setup remain those of the ABC launcher.
Commands below run from the DreamZero repository root.

## Prepare a task

```bash
source env.sh
python scripts/train/task_specific/prepare.py \
  --task 'put the plastic bottles in the bin' \
  --output-root data/task_specific/put_the_plastic_bottles_in_the_bin
```

The input defaults are `data/abc_yam_v3_train` and `data/abc_yam_v3_val`;
override them with `--train-metadata-root` and `--validation-metadata-root`.
The output must be a new directory. It contains `train/`, `validation/`,
and `summary.json`. Exact task text is resolved independently in each split:
the bottle task is ID **54** in training and **188** in validation.
`throw the plastic bottles in the bin` is a different task and is excluded.

Only matching window ranges and their episodes are kept. Episode/task IDs and
source file offsets are preserved. Videos and numeric payloads are read in place;
they are not copied. Source-wide `meta/info.json`, the task dictionary, and the
row-group index stay intact; subset counts are in `manifest.json` and
`summary.json`, and sampling uses the filtered Parquet metadata. Both splits
are verified before the output directory is published. Existing datasets are
never edited.

## Normalization

No new statistics pass is required. Both subsets retain the same verified
**full training-source** state and pooled four-block relative-action statistics,
byte for byte, including the original normalization manifest and hashes. A task
filter changes the sampling population, not the robot dimensions or relative
action definition. This preserves the normalization used by the existing ABC
training run. Validation never contributes to statistics.

Subset manifests explicitly record `reuse-source-training-statistics` and the
parent metadata hash. `norm_stats_manifest.json` intentionally describes the
full statistics population, not the smaller task subset. Re-estimating task-only
quantiles would be a separate experiment that changes action scaling; these
scripts do not claim to do that.

## Validate without training

```bash
export TASK_DATA_ROOT="$PWD/data/task_specific/put_the_plastic_bottles_in_the_bin"
python scripts/train/task_specific/validate.py \
  --metadata-root "$TASK_DATA_ROOT/train" --expected-split train \
  --samples 3 --decode-video
python scripts/train/task_specific/validate.py \
  --metadata-root "$TASK_DATA_ROOT/validation" --expected-split validation \
  --samples 3 --decode-video
bash scripts/train/task_specific/finetune.sh --preflight
bash scripts/train/task_specific/finetune.sh --dry-run
```

Validation checks hashes, task membership, subset counts, and normalization
provenance. With `--samples`, it reads sampled numeric windows and checks task
boundaries and the 96-action/four-state temporal plan. `--decode-video` also
decodes 33 frames from each of the three cameras. Sampling is a smoke check,
not an exhaustive payload scan. Preflight checks the real training config and
checkpoint paths; dry-run prints the resolved config. Neither starts training.

## Submit a new run

```bash
TASK_DATA_ROOT="$PWD/data/task_specific/put_the_plastic_bottles_in_the_bin" \
EXP_NAME=dreamzero_bottles_lora_bs8_run1 \
NUM_GPUS=8 BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1 \
MAX_STEPS=10000 SAVE_STEPS=2000 \
  bash scripts/train/task_specific/finetune.sh --submit
```

`10000` is an example budget, not a tuned stopping criterion. If omitted,
`MAX_STEPS` inherits the existing launcher's 100,000-step default. GPU, batch,
and accumulation defaults are 8, 1, and 1: global batch 8. Configure batch
sizes through these environment variables; the wrapper derives
`global_batch_size`. It requests four CPUs per GPU, 1024G RAM, and 24 hours by
default. `TASK_CPUS`, `TASK_MEMORY`, and `TASK_TIME` override those requests;
for example `TASK_CPUS=64 TASK_TIME=72:00:00 NUM_WORKERS=8` requests 64 CPUs,
72 hours, and eight data-loader workers per GPU. More workers do not guarantee
faster GPU computation. `TASK_NODELIST` optionally pins a node; otherwise
Slurm selects one. `TASK_DEPENDENCY=afterany:JOB_ID` optionally delays startup
until another job exits, and cancels the submission if the dependency is invalid.

Other launcher variables such as `LEARNING_RATE`, `OUTPUT_DIR`, `REPORT_TO`,
and `WANDB_MODE` are inherited. Extra arguments are Hydra overrides; dataset
selection must use `TASK_DATA_ROOT`. Without `EXP_NAME`, the default is
`dreamzero_<subset-directory-name>_lora_run1`. Use a new experiment name for
each new run. `--submit` performs preflight and then submits; the batch job
retains the existing output-directory and W&B checks.

To resume **this task run**, use `RUN_MODE=resume` with the same `EXP_NAME`,
`OUTPUT_DIR` (if overridden), subset, and batch settings. This is not a command
for converting the ongoing all-task job into a task-specific job. Fresh task
runs load the configured base pretrained model as usual.

## Held-out evaluation

The existing evaluator accepts the filtered validation metadata:

```bash
python scripts/eval_abc_yam.py \
  --metadata-root "$TASK_DATA_ROOT/validation" \
  --output-dir "logs/task_specific/bottles_baseline" --num-samples 32
```

That command computes the hold-state baseline on CPU. For model evaluation,
add `--checkpoint /path/to/task-run/checkpoint-N` and run on an allocated GPU.
Use separate output/anchor files for this subset; all-task anchors have a
different metadata hash. Training does not automatically run validation.

## Data volume and speed

The prepared bottle subset has 3,793 training episodes / 6,472,912 eligible
windows and 54 held-out episodes / 104,124 eligible windows. Sampling is with
replacement. At global batch 8, 10k steps draw 80k windows (1.24% of the training
window count), and 100k draw 800k (12.36%). These are sample-volume ratios,
not unique coverage. Filtering focuses updates on the chosen task; it does
not reduce the model's per-sample compute or guarantee faster steps.
