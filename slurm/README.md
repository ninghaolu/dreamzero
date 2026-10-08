# Slurm jobs

Run submission commands from the repository root. ABC jobs are grouped by purpose:

| Directory | Jobs |
| --- | --- |
| `abc/stats/` | Sharded normalization statistics, reduction, and train/validation metadata preparation |
| `abc/finetune/full/` | General full-finetuning launcher and the 100k-step H200 run |
| `abc/finetune/lora/` | Reserved for ABC LoRA jobs; no ABC LoRA Slurm recipe exists yet |
| `abc/benchmarks/` | CUDA and loader checks, smoke/stability runs, and memory/throughput experiments |
| `abc/eval/` | Held-out policy evaluation and [validation video inference](abc/eval/README.md) |
| `abc/monitor/` | CPU-only job progress and validation monitoring |

## Common commands

```bash
# Preview the statistics pipeline without submitting jobs.
bash slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash --dry-run

# Long full-finetuning run (eight H200s, CPU optimizer offload).
sbatch slurm/abc/finetune/full/full_finetune_h200_100k.sbatch

# No-offload comparison: eight H200s, batch 1/GPU, 100 updates, one-hour limit.
sbatch slurm/abc/benchmarks/full_finetune_h200_no_offload.sbatch

# Other short checks.
sbatch slurm/abc/benchmarks/full_finetune_h200_smoke.sbatch
sbatch slurm/abc/benchmarks/full_finetune_h200_stability.sbatch
sbatch slurm/abc/benchmarks/validate_cuda_h200.sbatch
sbatch slurm/abc/benchmarks/validate_abc_yam_loader.bash

# Monitor an existing job; supply its actual output directory.
sbatch slurm/abc/monitor/monitor_full_smoke.sbatch JOB_ID OUTPUT_DIR
```

See [the ABC training guide](../docs/abc/FULL_FINETUNING.md) for preparation,
configuration, evaluation, and checkpoint details. Resource requests and output
locations are defined in each script; moving scripts does not relocate run data.

The six older entrypoints directly under `slurm/` remain compatibility wrappers.
Use the categorized paths for new submissions. The previous flat `slurm/abc/`
paths have moved to the directories above. Slurm keeps a submitted copy of each
batch script, so already submitted training and monitor jobs keep their code.
