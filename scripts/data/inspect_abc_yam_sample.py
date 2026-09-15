#!/usr/bin/env python3
"""Real ABC alignment, transformed-shape and worker smoke checks."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE","1")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
from abc_yam_config import compose_abc


def main():
    import faulthandler
    faulthandler.dump_traceback_later(120, repeat=True)
    import av
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from PIL import Image
    from torch.utils.data import DataLoader
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--episode-ids",type=int,nargs="+",default=[0,1])
    parser.add_argument("--workers",type=int,default=0)
    parser.add_argument("--samples",type=int,default=4)
    args=parser.parse_args()
    distributed=int(os.environ.get("WORLD_SIZE","1"))>1
    if distributed:
        torch.distributed.init_process_group("gloo")
        args.output_dir=args.output_dir/f"rank-{torch.distributed.get_rank()}"
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(1)
    cfg=compose_abc(args.metadata_root,["++train_dataset.mixture_kwargs.num_shards_to_sample=16",
                                       "train_dataset.mixture_kwargs.shard_sampling_rate=0.0001"])
    begun=time.monotonic()
    mixture=instantiate(cfg.train_dataset)
    dataset=mixture.datasets[0]
    collator=instantiate(cfg.data_collator)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={"initialization_seconds":time.monotonic()-begun,"episodes":[],"valid_windows":len(dataset)}
    samples=[]
    for ep in args.episode_ids:
        ranges=dataset.step_filter[ep]
        if not len(ranges):raise ValueError(f"No valid windows for episode {ep}")
        start=ranges[0]
        begun=time.monotonic()
        raw=dataset.raw_sample(ep,start)
        item=dataset.transforms(copy.deepcopy(raw))
        expected={"images":(33,352,640,3),"state":(4,64),"action":(96,32)}
        for key,shape in expected.items():
            if tuple(item[key].shape)!=shape:raise ValueError(f"{key}: {item[key].shape} != {shape}")
        for key in ["state_mask","action_mask"]:
            if not np.all(item[key].sum(axis=-1)==14):raise ValueError(f"Invalid active dimensions: {key}")
        samples.append(item)
        detail=dict(episode=ep,start=start,task=raw["annotation.task"][0],seconds=time.monotonic()-begun,
                    shapes={key:list(item[key].shape) for key in expected},cameras={})
        rows=dataset.storage.read_episode(ep)
        # Compare the first/last requested frame with an independent seek/decode.
        from groot.vla.data.dataset.abc_v3_storage import CAMERAS
        for camera in CAMERAS:
            requested_indices=np.array([start,start+96])
            path,timestamps,interval=dataset.storage.camera_request(ep,camera,rows["timestamp"][requested_indices])
            errors=[]
            for sample_pos,requested in zip([0,-1],timestamps):
                with av.open(str(path)) as container:
                    stream=container.streams.video[0]
                    container.seek(max(0,int((requested-.1)/float(stream.time_base))),stream=stream,backward=True)
                    candidates=[]
                    for frame in container.decode(stream):
                        timestamp=float(frame.pts*stream.time_base)
                        candidates.append((abs(timestamp-requested),frame.to_ndarray(format="rgb24")))
                        if timestamp >= requested+1/30:break
                    _,reference=min(candidates,key=lambda v:v[0])
                actual=raw[f"video.{camera}"][sample_pos]
                error=int(np.abs(actual.astype(np.int16)-reference.astype(np.int16)).max())
                if error>2:raise ValueError(f"Independent decoder mismatch: {ep}/{camera}: {error}")
                errors.append(error)
            detail["cameras"][camera]=dict(path=str(path),interval=list(interval),max_pixel_errors=errors)
        for frame in [0,16,32]:
            Image.fromarray(item["images"][frame]).save(args.output_dir/f"episode-{ep}-frame-{frame}.png")
        report["episodes"].append(detail)
    for batch_size in [1,4]:
        batch=collator([samples[i%len(samples)] for i in range(batch_size)])
        if tuple(batch["action"].shape)!=(batch_size,96,32):raise ValueError("Invalid batch action shape")
    positions=[set(mixture.schedule_positions(rank,2,worker,2)) for rank in range(2) for worker in range(2)]
    if set.union(*positions)!=set(range(16)) or sum(map(len,positions))!=16:
        raise ValueError("Schedule partition mismatch")
    begun=time.monotonic()
    dataset.video_reader.close()
    dataset.storage.reset_cache()
    worker_options={"multiprocessing_context":"spawn"} if args.workers else {}
    loader=DataLoader(mixture,batch_size=1,num_workers=args.workers,collate_fn=collator,**worker_options)
    iterator=iter(loader)
    for _ in range(args.samples):
        batch=next(iterator)
        if tuple(batch["action"].shape)!=(1,96,32):raise ValueError("Worker batch shape mismatch")
    # Drain this deliberately tiny schedule so workers finish decoding before teardown.
    # Abruptly stopping a long prefetch stream can abort native decoder threads on exit.
    if args.workers:
        for batch in iterator:
            if tuple(batch["action"].shape)!=(1,96,32):raise ValueError("Worker batch shape mismatch")
    del iterator,loader
    report.update(worker_count=args.workers,iterated_samples=args.samples,
                  iteration_seconds=time.monotonic()-begun,numeric_cache_bytes=dataset.storage.cached_bytes,
                  video_cache_bytes=dataset.video_reader.cached_bytes,
                  rank=mixture.rank,world_size=mixture.world_size,
                  schedule_partition_verified=True,mixed_batch_verified=True)
    if distributed:
        assigned=list(mixture.schedule_positions(mixture.rank,mixture.world_size))
        gathered=[None]*mixture.world_size
        torch.distributed.all_gather_object(gathered,assigned)
        if sorted(sum(gathered,[]))!=list(range(16)):
            raise ValueError("Actual distributed schedule partition mismatch")
    (args.output_dir/"inspection.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    faulthandler.cancel_dump_traceback_later()
    if distributed:torch.distributed.destroy_process_group()


if __name__=="__main__":main()
