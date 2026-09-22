#!/usr/bin/env bash
# Compatibility entrypoint; implementation lives in slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash.
set -euo pipefail
exec bash "/projects/work/yang-lab/projects/policy-finetuning/dreamzero/slurm/abc/stats/submit_abc_yam_dreamzero_stats.bash" "$@"
