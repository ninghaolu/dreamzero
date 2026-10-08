# ABC validation video inference

From the repository root, prepare validation metadata once if absent:

```bash
sbatch --array=1 slurm/abc/stats/prepare_abc_yam_v3.bash
```

After preparation completes, submit checkpoints 15000, 16000 and 17000 from
`/projects/data/cloud_user/checkpoints/dreamzero_abc130k/full_h48_5491`:

```bash
bash slurm/abc/eval/submit_inference.bash
```

Or pass any full checkpoint directories as positional arguments. Each job uses
one GPU, four validation anchors and seed 42, so checkpoints see identical
samples. `ABC_EVAL_SAMPLES`, `ABC_EVAL_SEED`, `ABC_EVAL_ANCHORS`,
`ABC_METADATA_ROOT`, and `ABC_INFERENCE_OUTPUT` can override these defaults.
Use a fresh output directory per comparison. Submission records are in
`jobs.tsv`; job logs are in `logs/infer-JOBID.{out,err}`.

To run just one checkpoint with an explicit output directory:

```bash
sbatch slurm/abc/eval/infer_abc_yam_h100.sbatch /path/to/checkpoint /path/to/output
```

Each sample saves the first causal prediction chunk as an H.264 MP4 at the
training video rate (5 fps for H48), including the conditioning frame. These are
short model predictions, not environment rollouts. Outputs also include input
camera mosaics, action predictions, metrics and anchors identifying the episode,
frame, prompt and seed. The evaluator uses the checkpoint's native inference
defaults and VAE decoder. Datasets and checkpoints are read in place.

For 12-chunk autoregressive predictions, also saving the matching 8-chunk prefix:

```bash
ABC_INFERENCE_CHUNKS=12 ABC_SAVE_CHUNK_PREFIX=8 \
  bash slurm/abc/eval/submit_inference.bash
```

Only the initial images/state come from the dataset. Subsequent calls consume
generated video latents and use the previous chunk's final absolute action as
the next estimated joint/gripper state. This is a model rollout, not a physics
simulation or closed-loop robot execution. The native four-chunk attention
limit is preserved: chunks 5 and 9 restart context from the last generated image.
Later clips can drift as visual and state errors accumulate.

Eight/twelve H48 chunks span 12.8/19.2 seconds; the MP4s contain 65/97 frames
(13.0/19.4 seconds of playback including the initial frame). The 8-chunk video
is the exact prefix of the same 12-chunk rollout before video compression.
`actions.npz` includes `rollout_prediction` for all chunks; action error metrics
remain explicitly limited to the first chunk. Per-chunk seeds, timings and
context resets are recorded in `metrics.json`.

For action errors and validation loss on a larger fixed sample (no MP4 decoding):

```bash
ABC_EVAL_SAMPLES=256 sbatch slurm/abc/eval/metrics_abc_yam_h100.sbatch \
  /path/to/checkpoint /path/to/fresh-output
```

The default is 256 windows, seed 42, with two fixed noise draws per window.
`ABC_EVAL_ANCHORS` can point to an existing shared anchor file.
`metrics.json` reports action errors in dataset units for the first causal H48 chunk
and the hold-current-state baseline; `actions.npz` retains individual predictions
and targets. Joint and gripper dimensions should be summarized separately.
`validation-loss.json` reports the native scheduler-weighted action and video
flow-matching losses on the full four-block window. Model weights stay frozen;
image transforms use evaluation mode and language dropout is disabled.
`validation-loss-samples.jsonl` retains each noise draw for paired comparisons.
This denoising loss differs from inference action MAE/RMSE and from robot success.
Both metrics use the same episode/start anchors: action error scores the first
48 actions, while validation loss scores the full 192-action / 33-frame window.
The evaluation batch size is one window; noise draws run sequentially.
