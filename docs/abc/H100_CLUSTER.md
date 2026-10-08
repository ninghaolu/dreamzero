# ABC full fine-tuning on the H100 cluster

Use `slurm/abc/finetune/full/full_finetune_h100_100k.sbatch` from this checkout.
The old `full_finetune_h200_100k.sbatch` path forwards to it and has the same
H100 resource directives. Other historical H200 benchmark scripts are not
entrypoints for this cluster.

## Configuration

| Setting | Value |
| --- | --- |
| Partition / account | `defq` / inherited cluster account (`oneworld`) |
| Allocation | One node, 8 H100s, 32 CPUs, 1024 GiB RAM, 48 hours |
| GPU request | `--gres=gpu:8`; Slurm advertises generic GPU resources |
| Dataset | `/projects/data/datasets/abc_130k_v3_train` |
| Metadata | `data/abc/h48/train` |
| Temporal preset | H48, K2, M4, 192 actions and 33 video frames |
| Training | Full diffusion/action model; text/image encoders and VAE frozen |
| Precision / optimizer | BF16, TF32, ZeRO-2 with CPU optimizer offload |
| Learning rate | `1e-5` |
| Batch | 1 per GPU, accumulation 1, global batch 8 |
| Steps / save interval | 100,000 / 1,000 |
| Checkpoint retention | Most recent 1,000 periodic checkpoints |
| Data workers | 2 per rank |
| Checkpoints | `/projects/data/cloud_user/checkpoints/dreamzero_abc130k/full_h48_JOBID` |
| Slurm logs | `logs/full-h100-JOBID.out` and `.err` |
| Tracking | W&B `oneworld-ai/policy-finetuning` |

The launcher checks that all eight visible GPUs are H100s. It calculates global
batch size from GPU count, microbatch and accumulation. `MAX_STEPS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`,
`BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`, `NUM_WORKERS`, `LOGGING_STEPS`,
`OUTPUT_DIR`, and `REPORT_TO` can be exported before submission.

`env.sh` activates the existing `.venv`, sets local caches and locates the
pretrained DreamZero-AgiBot, Wan2.1 and tokenizer assets under
`/projects/data/cloud_user/checkpoints`. The training CUDA setup uses the
installed CUDA 12.9 compiler and makes the PyTorch wheel's cuRAND library
available to the CPUAdam linker through a job-local library directory.

## Preparation and submission

Legacy `data/abc_yam_v3_train` and its normalizer are H24 and cannot be used for
H48. Compute the full H48 normalizer, then index the training data; raw videos
and numeric files are read in place. The statistics arrays default to 128
shards and 128 concurrent one-CPU tasks, with 8 GiB requested per task.

```bash
cd /home/alex/policy-finetuning/dreamzero
source env.sh
mkdir -p logs
bash slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash
# Set FINALIZE_ID to the printed finalize job ID.
sbatch --dependency=afterok:${FINALIZE_ID} --kill-on-invalid-dep=yes \
  slurm/abc/stats/prepare_abc_yam_v3.bash
sbatch slurm/abc/benchmarks/validate_cuda_h100.sbatch
# Set PREP_ID and ENV_ID to those two job IDs.
sbatch --job-name=dz-abc-h48-full-smoke --time=02:00:00 \
  --export=ALL,MAX_STEPS=3,SAVE_STEPS=3,LOGGING_STEPS=1 \
  --dependency=afterok:${PREP_ID}:${ENV_ID} --kill-on-invalid-dep=yes \
  slurm/abc/finetune/full/full_finetune_h100_100k.sbatch
# Set SMOKE_ID to the returned job ID.
sbatch --dependency=afterok:${SMOKE_ID} --kill-on-invalid-dep=yes \
  slurm/abc/finetune/full/full_finetune_h100_100k.sbatch
```

The smoke uses the same eight GPUs and full dataset as training. It requires
three optimizer steps, finite losses, the expected trainable/frozen parameter
inventory, and a structurally complete full checkpoint on all eight ranks.
Its final checkpoint is not used to initialize the separate long run. An
unsuccessful prerequisite prevents its dependent jobs from starting.

To resume an interrupted long run, export its existing `OUTPUT_DIR` and submit
the full launcher again. The training code selects its latest checkpoint;
verify that checkpoint finished saving before resuming. The 48-hour allocation
is a walltime limit, not an estimate that all 100,000 steps will fit.

## Submission record: September 22, 2026

| Job | Purpose |
| --- | --- |
| `5227` | H48 statistics pass 1, 128 shards; concurrency raised to 128 |
| `5228` | Reduce pass 1 |
| `5229` | H48 statistics pass 2, 128 shards; concurrency raised to 128 |
| `5230` | Finalize H48 normalizer |
| `5263` | Prepare H48 training metadata after `5230` |
| `5359` | H100 environment validation, **passed** |
| `5360` | Three-step full-training smoke after `5263` and `5359` |
| `5361` | Initial full run; cancelled at user request before any checkpoint was saved |
| `5491` | Replacement full run, retaining the most recent 1,000 checkpoints |

The user requested the retention change and relaunch after the smoke passed.
The replacement starts fresh in `full_h48_5491`, saves every 1,000 steps, and
keeps up to 1,000 periodic checkpoints. At 100,000 target steps, this allows
all 100 scheduled periodic checkpoints to remain. The original run's outputs
are preserved in `full_h48_5361`.

```bash
sbatch --parsable --export=ALL,SAVE_TOTAL_LIMIT=1000 \
  slurm/abc/finetune/full/full_finetune_h100_100k.sbatch
```

Environment job `5226` exposed a missing unversioned `libcurand` linker target;
`5359` passed after the library-path fix. No dependency version changes were
needed. The GPU check exercised FlashAttention forward/backward, torchvision
CUDA NMS, BF16 matrix multiplication, and CPUAdam against PyTorch AdamW.
All 22 `test_abc_yam*.py` tests passed. The H48 config composes as full training
with full checkpoint saves; pretrained file validation also passed using the
existing H24 metadata while the new H48 metadata was being prepared.

Job submission does not establish H100 training memory fit; the smoke is the
end-to-end check. Inspect current state with:

```bash
squeue -u alex
sacct -j 5359,5360,5491,5263 --format=JobID,State,ExitCode,Elapsed,NodeList
tail -f logs/full-h100-5491.out logs/full-h100-5491.err
```
