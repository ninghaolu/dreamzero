# ABC-130K full finetuning

The ABC launcher defaults to `ABC_TEMPORAL=h48` and `train_architecture=full`.
The diffusion/action model, including state/action encoders and action decoder,
is trainable. The text encoder, image encoder, and VAE remain frozen. The previous
ABC launcher and saved smoke runs used LoRA. This launcher saves full checkpoints
(`save_lora_only=false`) and uses ZeRO-2 with CPU optimizer offload. It does not
change the model's existing parameter-freezing implementation.

## Experiment tracking

ABC training defaults to `report_to=wandb` at
<https://wandb.ai/oneworld-ai/policy-finetuning>, matching the pi0.5 ABC launcher.
`scripts/train/abc_wandb_env.sh` uses the same credential file at
`../yam-abc-reproduce/.secrets/wandb_alex_api_key`; an exported `WANDB_API_KEY`
takes precedence, and `WANDB_API_KEY_FILE` can select another credential file.
The Trainer creates one run from rank zero, named after the output directory, and
logs losses, learning rate, existing timing metrics, configuration, and W&B
system metrics. Model/checkpoint uploads and gradient watching are off by default.
This enables tracking but does not add an MFU calculation or rolling throughput
metrics. The per-rank resource JSON files and GPU utilization CSV remain local.

Set `WANDB_PROJECT` / `WANDB_ENTITY` before submission to choose a different
destination. Use `REPORT_TO=none` or a trailing `report_to=none` argument to
disable tracking, or `WANDB_MODE=offline` to record locally for later syncing.
Queued jobs that call the shared `scripts/train/abc_yam_v3_training.sh` read its
defaults when they start; already-running training processes keep their settings.

## Temporal presets

Native Hydra presets live in `groot/vla/configs/abc_temporal/`. Select the same
preset for statistics, preparation, inspection, and training. Shell entrypoints
use `ABC_TEMPORAL`; Python preparation/statistics commands use `--temporal`.
The low-level Python commands retain H24 as their compatibility default, so pass
`--temporal h48` explicitly when calling them directly.

| Setting | `h24` | `h48` (default launcher) |
|---|---:|---:|
| Action rate | 30 Hz | 30 Hz |
| Video rate / source stride | 10 FPS / 3 | 5 FPS / 6 |
| `action_horizon` / `num_action_per_block` | 24 / 24 | 48 / 48 |
| `num_frame_per_block` (K) | 2 | 2 |
| `num_state_per_block` | 1 | 1 |
| `max_chunk_size` (M) | 4 | 4 |
| `num_frames` | 33 | 33 |
| Actions per sample | 96 | 192 |
| Required consecutive same-task episode rows | 97 | 193 |
| Chunk duration / first-to-last video span | 0.8 s / 3.2 s | 1.6 s / 6.4 s |

H48 samples source video indices `0,6,...,192`, actions `0,...,191`, and
states `0,48,96,144`, relative to sample start. Video requests use the numeric
rows' timestamps plus each camera's own episode offset. Original MP4 files are
read in place, with no re-encoding. The final video endpoint must still belong
to the same task and episode; neither actions nor video are boundary-padded.

Targets, including both grippers, are
`action[start + chunk*H + offset] - state[start + chunk*H]`.
Statistics pool all such targets from **every eligible training window**, uniformly
by window start. State statistics are copied from the full training source.
Validation preparation always copies the training normalizer; it never fits one
on validation data. Histogram quantiles retain the existing approximation and
record their error bounds in the normalizer manifest.

Changing H requires new statistics. Changing M also requires new statistics under
this policy: it changes eligible windows and target multiplicities even though
individual chunk-relative targets are unchanged. These presets support M=4;
setting only `max_chunk_size=2` is rejected. A future M=2 preset would also need
consistent frame counts, metadata, and statistics. A per-chunk statistics
population could permit reuse across M, but would be a different normalization
policy.

## Prepare full-data statistics and window metadata

Run from the repository root. These Slurm scripts use the existing cluster
account and DreamZero environment. Implementations are grouped by purpose under
`slurm/abc/`; see the [Slurm directory guide](../../slurm/README.md). The old
`slurm/*abc*` paths are compatibility wrappers.

```bash
source /projects/work/yang-lab/projects/policy-finetuning/envs/dreamzero/activate.sh
export ABC_TEMPORAL=h48
export ABC_DZ_STATS_NUM_SHARDS=128
export ABC_DZ_STATS_MAX_CONCURRENT=128
bash slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash --dry-run
bash slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash
```

The submission prints IDs for `pass1`, `reduce-pass1`, `pass2`, and `finalize`.
The reducers depend on successful completion of the preceding stage. All 742
training numeric files are included; do not use `ABC_DZ_STATS_MAX_FILES` for a
training normalizer. Restart the same command to reuse completed compatible
parts. Changing the temporal preset or shard/bin settings requires a separate
work/output directory. The shared identity manifest is published once, avoiding
concurrent replacement on NFS.

```bash
# Replace with the finalize ID printed above; preparation starts only on success.
FINALIZE_JOB_ID=123456
sbatch --array=0-1 --dependency=afterok:${FINALIZE_JOB_ID} --kill-on-invalid-dep=yes \
  --output="outputs/abc/${ABC_TEMPORAL}/stats/prepare-%A_%a.out" \
  --error="outputs/abc/${ABC_TEMPORAL}/stats/prepare-%A_%a.err" \
  slurm/abc/stats/prepare_abc_yam_v3.bash
```

Array index 0 prepares train; index 1 prepares validation. The fresh outputs are:

```text
data/abc/h48/norm_stats/  # full training normalizer and provenance
data/abc/h48/train/       # H48 windows + a copy of training statistics
data/abc/h48/val/         # H48 validation windows + the same training statistics
outputs/abc/h48/stats/    # logs and resumable work/pass{1,2} parts
```

Preparation refuses to overwrite an existing metadata directory. H24 outputs go
under `data/abc/h24/` when prepared using `ABC_TEMPORAL=h24`. Existing legacy
`data/abc_yam_v3_train`, `data/abc_yam_v3_val`, and `data/abc_yam_v3_norm_stats`
remain usable with H24. Manifests without temporal fields are interpreted as
legacy H24 only; H48 cannot use them.

Direct preparation, including a reproducible pilot, is also available. A pilot
still requires the **full-training-data** normalizer:

```bash
python scripts/data/prepare_abc_yam_v3.py --temporal h48 \
  --source-root /projects/work/yang-lab/projects/pretrain_world_model/abc_130k_v3_train \
  --normalizer-root data/abc/h48/norm_stats \
  --output-root data/abc/h48/pilot --pilot-episodes 128 --seed 42
```

## Validate and inspect

Compute nodes use headless OpenCV (`opencv-python-headless==4.11.0.86`), pinned
in `pyproject.toml`. Do not install the GUI `opencv-python` package alongside it;
both provide `cv2`, and the GUI build requires system `libGL`.

```bash
python -m unittest discover -s tests

# Compose and validate configuration without requiring prepared files.
ABC_DRY_RUN=1 bash scripts/train/abc_yam_v3_training.sh

# Check real metadata, normalizer identity/hashes and checkpoint files; no model/GPU allocation.
ABC_VALIDATE_ONLY=1 bash scripts/train/abc_yam_v3_training.sh

# Run on a CPU allocation; inspect raw targets, timestamps, decoded frames,
# transformed shapes, mixed batches and worker iteration.
python scripts/data/inspect_abc_yam_sample.py --temporal h48 \
  --metadata-root data/abc/h48/train --output-dir outputs/abc/h48/inspection \
  --episode-ids 0 1 --workers 2 --samples 8

# Two distributed ranks with two workers each, plus launch preflight:
sbatch slurm/abc/benchmarks/validate_abc_yam_loader.bash
```

Expected H48 transformed shapes are images `(33,352,640,3)`, state `(4,64)`,
and action `(192,32)`, with 14 active state/action dimensions. Inspection writes
PNG samples and a JSON report with all requested indices and video timestamps.
Launch validation rejects mismatched H, K, M, frame counts, dataset stride,
metadata, normalizer, and LoRA overrides.

## Launch full finetuning

Current long-run submission: **17988437**, submitted September 18 at 22:52:05
cluster local time, initially pending. The user requested direct launch without
further test jobs. The dedicated script requests eight H200s, 32 CPUs, 1000G
host memory, 48 hours, global batch 16 (two samples per GPU, accumulation 1),
100,000 target steps, and checkpoints every
1,000 steps. H48/K2/M4 and full-data/full-parameter training remain unchanged.

Previous attempt: jobs 17940081 and 17940082 were cancelled by UID 0 (root/admin) before starting.
Training `17940081` was cancelled September 18 at 07:25:32 cluster local time;
monitor `17940082` at 07:24:25. Neither had a node allocation or runtime. No
training logs or new checkpoint were created. Slurm records no cancellation
reason; this is an administrative cancellation, not a model/runtime failure.
The user explicitly requested resubmission. Replacement 17947778 was submitted
September 18 at 10:57:48 cluster local time. It ran on gh134 from 11:58:28 to
16:00:34 (4h02m06s), then Slurm recorded `CANCELLED by 0`. The error log confirms
external SIGTERM; no CUDA, OOM, or Python runtime error was found. All eight ranks
reached step 998, effective global batch 8, and all 99 aggregated loss records
were finite (last logged loss 0.0645 at step 990). No checkpoint was saved:
cancellation occurred two steps before the first scheduled save at step 1000.
This attempt cannot resume its 998 updates. Monitor 17947783 exited FAILED/1
because its watched training job was cancelled. Slurm's AdminComment and
SystemComment are empty; the cancellation's underlying reason needs cluster
administrator confirmation. The accounting Reason field `QOSMaxGRESPerUser`
is a pending-job reason, not evidence of why this running job was cancelled.
No node pin or reservation flag was requested; Slurm associated the permitted
`my3149-h200` reservation with the gh134 allocation. gh101–gh104 were not used.

```bash
sbatch slurm/abc/finetune/full/full_finetune_h200_100k.sbatch
```

The user supplied an HPC utilization warning for 17947778: per-GPU averages
60.01–62.66%, warning threshold 75%, cancellation threshold 60%. That email
explicitly had `warning=1 cancel_job=0`; low-utilization enforcement is a
plausible explanation for the later root cancellation, not confirmed by this
warning alone. The replacement increases the per-GPU batch from one to two,
leaving CPU/offload/data-loader settings unchanged, and records GPU utilization
every five seconds in `gpu_utilization.csv`. Throughput/memory at batch two
remain unmeasured. Test job 17988326 was cancelled before allocation at the
user's request, and its unused launch script was removed. No additional test
or monitor job was submitted with 17988437.

Output: `outputs/abc/h48/full_100k_17988437/`. No node pin or reservation flag is
requested; `gh101`–`gh104` remain protected by their active non-magnetic
reservation. The full job records per-rank training progress and GPU utilization directly.
At measured throughput, 100,000 steps exceed 48 hours. A later allocation can
resume using the same `OUTPUT_DIR` after checking that its latest checkpoint
finished writing; no automatic extra 48-hour allocation is submitted.

### CUDA compiler for CPU optimizer offload

The September 14 environment snapshot (`../envs/dreamzero/requirements.freeze.txt`)
uses PyTorch 2.8.0+cu129, DeepSpeed 0.16.9, and FlashAttention 2.8.3.post1.
These versions were unchanged when job 17912007 failed. The switch from LoRA's
`zero2.json` to full training's `zero2_offload.json` introduced CPUAdam's extension
build, which found the H200 node's CUDA 13.3 toolkit instead of CUDA 12.9.
The prior setup installed CUDA runtime wheels, but no pinned compiler toolkit.
The separate GUI-to-headless OpenCV repair did not cause this CUDA mismatch.

Install a separate build toolkit once, leaving the Python environment intact:

```bash
CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs" CONDA_OVERRIDE_CUDA=12.9 \
  conda create --solver libmamba \
  --prefix "$PWD/.cache/cuda-12.9" --override-channels \
  -c nvidia/label/cuda-12.9.1 -c conda-forge -y \
  cuda-nvcc=12.9.86 cuda-cudart-dev=12.9.79 \
  libcublas-dev=12.9.1.4 libcurand-dev=10.3.10.19

# Small allocated-GPU test: compile CPUAdam, compare updates with PyTorch AdamW,
# and run a BF16 CUDA matrix multiplication.
sbatch slurm/abc/benchmarks/validate_cuda_h200.sbatch
```

The ABC training launcher sources `scripts/train/abc_cuda_env.sh`, selects this
toolkit and its host compiler, and runs `validate_abc_cuda.py` before loading the
model. `ABC_CUDA_HOME` can select another toolkit with the same CUDA major/minor
as PyTorch. Extension builds are cached locally per job and reused by all ranks.
Do not suppress version checks with `DS_SKIP_CUDA_CHECK=1`. Config-only and
metadata-only validation remain usable on CPU nodes without a toolkit.

After preparation and validation succeed:

```bash
export ABC_TEMPORAL=h48
export ABC_METADATA_ROOT="$PWD/data/abc/h48/train"
export MAX_STEPS=10000 SAVE_STEPS=1000 NUM_WORKERS=4
export BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=1
sbatch slurm/abc/finetune/full/finetune_abc_yam_dreamzero.sbatch

# Or, inside an existing GPU allocation:
NUM_GPUS=2 bash scripts/train/abc_yam_v3_training.sh
```

The ABC GPU scripts have no node pin and request no reservation. The user's
clarified policy permits preemption on other nodes but forbids using or preempting
jobs on `gh101`–`gh104`. Do not request `--reservation=my3149-h200-2`.

The administrator removed `MAGNETIC` from `my3149-h200-2` on `gh101`–`gh104`;
its flags are now `FLEX,IGNORE_JOBS,SPEC_NODES`. Jobs no longer automatically use
that reservation. `my3149-h200` on `gh105`–`gh106` still has `MAGNETIC` and includes
this user. The cluster rejects `--exclude` submissions for these ranges.

Correction to the earlier scheduling note: the cluster uses
`PreemptType=preempt/partition_prio`. `h200_public` has PriorityTier 10000, `h200`
has 5000, and `cpu_prem` has 800. `PreemptMode=OFF` on a partition protects its
jobs; it does not establish that its jobs cannot preempt lower-tier jobs.
The empty `gpu48` QoS preemption list does not disable partition-based preemption.
Preemption outside `gh101`–`gh104` is explicitly authorized. The active non-magnetic
reservation protects those four nodes from this unreserved submission; recheck
this protection if the reservation expires or changes. Reservation auto-assignment
and partition-priority preemption are separate issues. See the
[Slurm preemption documentation](https://slurm.schedmd.com/preempt.html).

The launcher defaults
to only 50 steps when `MAX_STEPS` is unset. Full checkpoints are much larger than
LoRA checkpoints; `SAVE_STEPS` above avoids the old ten-step smoke cadence. CPU
optimizer offload is the default for full training; memory/throughput still need
a GPU smoke run on the intended allocation.

To select the original temporal setup with existing metadata:

```bash
ABC_TEMPORAL=h24 ABC_METADATA_ROOT="$PWD/data/abc_yam_v3_train" \
  ABC_VALIDATE_ONLY=1 bash scripts/train/abc_yam_v3_training.sh
# Remove ABC_VALIDATE_ONLY to train this H24 recipe in a GPU allocation.
```

## Held-out evaluation

```bash
python scripts/eval_abc_yam.py --temporal h48 \
  --metadata-root data/abc/h48/val --num-samples 32 \
  --output-dir outputs/abc/h48/validation

# On a GPU allocation, evaluate a saved full checkpoint at those same anchors:
python scripts/eval_abc_yam.py --temporal h48 \
  --metadata-root data/abc/h48/val --num-samples 32 \
  --anchors outputs/abc/h48/validation/anchors.json \
  --checkpoint outputs/abc/h48/YOUR_RUN/checkpoint-1000 \
  --output-dir outputs/abc/h48/evaluation
```

Evaluation validates the entire eligible window, then compares the first causal
H-step prediction to absolute source actions and the hold-state baseline.
H48 uses 48 steps; H24 uses 24. Saved anchors are tied to the metadata hash, and
checkpoint temporal settings must match evaluation data. Model outputs are
inverse-normalized and converted to absolute actions by the native policy once.

## One-node H200 smoke test

```bash
# 2 H200 GPUs on one node, 30-minute limit, 3 optimizer steps, full checkpoint save.
# No node pin or reservation; gh101–gh104 remain in the non-magnetic reservation.
sbatch slurm/abc/benchmarks/full_finetune_h200_smoke.sbatch
```

This uses the same H48 full-finetuning launcher with CPU optimizer offload,
16 CPU cores and 512 GiB host memory. Per-GPU batch size is 1, gradient
accumulation is 1, and effective global batch size is 2. The successful eight-GPU
smoke remains evidence for that allocation; the new two-GPU allocation must pass
its own memory, optimizer-step, and checkpoint checks.

After training, the smoke script runs `scripts/train/validate_abc_yam_run.py`.
It requires three optimizer steps on every rank, finite logged losses, the
expected frozen/trainable parameter inventory, and full-checkpoint tensor-key
and shape coverage. Results go to `smoke_validation.json` in the run directory.
This validates checkpoint structure; it does not perform a GPU inference reload.

After the smoke passes, run the 100-step stability check (same two H200s,
H48/full dataset, effective batch 2, checkpoints every 50 steps, 2-hour limit):

```bash
sbatch slurm/abc/benchmarks/full_finetune_h200_stability.sbatch
# The monitor accepts the stability run directory as an optional second argument.
sbatch slurm/abc/monitor/monitor_full_smoke.sbatch JOB_ID \
  "$PWD/outputs/abc/h48/full_stability_JOB_ID"
```

To monitor a queued smoke job without an interactive session:

```bash
# Replace with the GPU job ID printed by sbatch.
sbatch slurm/abc/monitor/monitor_full_smoke.sbatch GPU_JOB_ID
```

The CPU monitor checks once per minute for up to 12 hours and records Slurm state,
per-rank optimizer progress, and the final validation result in
`outputs/abc/h48/stats/monitor-GPU_JOB_ID.json`. It exits on completion/failure,
and does not cancel or automatically retry the GPU job. Failed runs require
inspection of `full-smoke-GPU_JOB_ID.{out,err}` before resubmission.

## Verified preparation and CPU results (2026-09-17)

- Full H48 normalizer: 742 training files, 382,468,339 source rows,
  357,681,851 valid windows, 68,674,915,392 chunk-relative action targets.
- H48 metadata: 129,225 training episodes and 1,592 validation episodes;
  validation has 4,764,046 eligible windows. Both splits contain identical
  training-normalizer manifests.
- Launch preflight passed for H48 and the existing legacy H24 metadata.
- Distributed H48 inspection passed on two ranks, each with two workers and
  eight samples: `(33,352,640,3)` images, `(4,64)` state, `(192,32)` action,
  timestamps, per-camera endpoint decoding, and chunk-relative targets.
  Reports: `outputs/abc/h48/inspection-17909035/rank-{0,1}/inspection.json`.
- Held-out H48 hold-state baseline: 32 anchors, saved under
  `outputs/abc/h48/validation/`.
- 21 repository tests passed before adding four smoke-artifact validator tests;
  all four additional tests passed as well.
- GPU smoke job `17912007` eventually ran without a reservation on `gh101`,
  then failed before the first optimizer step after 8 minutes 9 seconds.
  DeepSpeed CPUAdam compilation detected CUDA toolkit 13.3 versus PyTorch CUDA
  12.9. Monitor `17912015` recorded the failure and exited.
- September 17 repair: isolated CUDA 12.9 toolkit installed; original Python
  package versions preserved. H200 preflight job `17935806` passed CPUAdam
  compilation and numerical update comparison, plus BF16 GPU matmul. All 25
  repository tests passed together.
- Repaired eight-H200 smoke `17935824` passed on `gh101`: three optimizer steps
  on all ranks, losses 0.1197 / 0.0861 / 0.123, all 16,484,292,448 diffusion-model
  parameters trainable, only text/image encoders and VAE frozen. Full checkpoint
  validation passed for 2,146 parameter tensors across ten safetensors shards.
  Peak allocated GPU memory was 61.5 GiB per rank. Report:
  `outputs/abc/h48/full_smoke_17935824/smoke_validation.json`.
  Sampled diffusion attention, state encoder, and action decoder checkpoint
  weights changed relative to the base; sampled frozen text-encoder weights
  were identical (`outputs/abc/h48/environment_debug/smoke_weight_comparison.json`).
- September 18: user selected **two H200 GPUs total**, per-GPU batch 1,
  accumulation 1, effective batch 2. Three-step test **`17938445` completed and
  passed** on `gh110` in 23m13s (monitor `17938451` also succeeded). The 100-step
  run **`17938459` completed and passed** on `gh108` in 1h22m05s (monitor `17938463`
  also succeeded). Both ranks completed all 100 optimizer steps with finite losses;
  full-checkpoint validation passed for 2,146 parameter tensors in ten shards.
  Peak allocated GPU memory was about 73.8 GiB per rank; mean training loss was
  0.10117. Report: `outputs/abc/h48/full_stability_17938459/smoke_validation.json`.
  Both requested 16 CPUs and 512 GiB host memory, with no node pin or reservation
  flag. Slurm automatically associated them with the other magnetic reservation,
  `my3149-h200`; neither used `gh101`–`gh104` or `my3149-h200-2`. The old eight-GPU
  job `17937600` and its monitor were cancelled. Two-GPU full-finetuning stability
  and checkpoint saving are now validated; held-out trained-policy quality is not
  established by this 100-step test.
- Previous eight-GPU 100-step job: **`17937600`**, no node pin,
  no reservation assigned, eight H200s, 2-hour wall limit, global batch 8, full
  H48 data, checkpoint cadence 50. Output: `outputs/abc/h48/full_stability_17937600/`.
  Monitor `17937613` starts with training. The user explicitly permits preemption
  outside `gh101`–`gh104`; the non-magnetic reservation protects that node range.
  Held attempt `17937241` was cancelled after Slurm refused to release it;
  `17937600` is a fresh unheld submission.
  The earlier gh108-pinned job `17936730` and monitor `17936773` were cancelled
  after discovering partition-priority preemption remained enabled independently
  of the reservation change.
  Previous job `17936356` was cancelled during initialization at the user's
  request to avoid `gh101`–`gh104`; Slurm had automatically assigned the magnetic
  reservation despite no reservation flag being submitted. The 100-step result
  is not yet validated.
  First attempt `17936252` stopped at launch preflight because its retention
  override was below DreamZero's required minimum; the replacement preserves
  the default `save_total_limit=5` and passed the exact metadata/config preflight.
