"""Small per-rank training resource record for the ABC smoke/pilot runs."""
import json
from pathlib import Path
import resource
import time
import torch
from transformers import TrainerCallback


class ABCTrainingMetrics(TrainerCallback):
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.started=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        inventory={key:dict(shape=list(value.shape),trainable=value.requires_grad)
                   for key,value in model.named_parameters()}
        if state.is_world_process_zero:
            (Path(args.output_dir)/"parameter_inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")

    def on_step_end(self, args, state, control, **kwargs):
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        record=dict(global_step=state.global_step,seconds=time.monotonic()-self.started,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                    peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    effective_global_batch=args.world_size*args.per_device_train_batch_size*args.gradient_accumulation_steps)
        (Path(args.output_dir)/f"resources-rank-{rank}.json").write_text(json.dumps(record,indent=2)+"\n")
