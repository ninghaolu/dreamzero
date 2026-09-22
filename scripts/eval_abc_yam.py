#!/usr/bin/env python3
"""Fixed held-out ABC anchors: causal configured-horizon action errors and hold-state baseline."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"groot/vla/data/dataset"))
from abc_v3_storage import ABCV3Storage, CAMERAS, SLICES, sha256, verify_normalizer
from yam_temporal import TemporalConfig, YAMTemporalPlan
from lerobot_v3_video import V3VideoReader


def metrics(predicted,target):
    error=np.asarray(predicted,dtype=np.float64)-np.asarray(target,dtype=np.float64)
    return dict(per_dimension_mae=np.abs(error).mean(axis=(0,1)).tolist(),
                per_dimension_rmse=np.sqrt(np.square(error).mean(axis=(0,1))).tolist(),
                left_gripper_mae=float(np.abs(error[:,:,6]).mean()),
                right_gripper_mae=float(np.abs(error[:,:,13]).mean()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", choices=["h24", "h48"], help="Default: infer from prepared metadata")
    parser.add_argument("--metadata-root",required=True)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--anchors",type=Path)
    parser.add_argument("--num-samples",type=int,default=32)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--checkpoint",help="Omit to evaluate only the hold-current-state baseline")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    if args.num_samples<1:parser.error("num-samples must be positive")
    storage=ABCV3Storage(args.metadata_root)
    temporal = storage.temporal
    if args.temporal and temporal != TemporalConfig.from_name(args.temporal):
        raise ValueError("Evaluation temporal configuration differs from metadata")
    verify_normalizer(args.metadata_root, temporal)
    horizon = temporal.block_length
    if storage.manifest["split"]!="validation":raise ValueError("Use separate validation-source metadata")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    identity=sha256(Path(args.metadata_root)/"manifest.json")
    anchors_path=args.anchors or args.output_dir/"anchors.json"
    if anchors_path.exists():
        anchor_doc=json.loads(anchors_path.read_text())
        if anchor_doc["metadata_sha256"]!=identity:raise ValueError("Anchor dataset identity mismatch")
        anchors=anchor_doc["anchors"][:args.num_samples]
        if not anchors:raise ValueError("Anchor file is empty")
    else:
        windows=storage.windows
        lengths=(windows.stop-windows.start).to_numpy()
        ends=np.cumsum(lengths)
        rng=np.random.default_rng(args.seed)
        anchors=[]
        for draw in rng.integers(int(ends[-1]),size=args.num_samples):
            pos=int(np.searchsorted(ends,draw,side="right"));row=windows.iloc[pos]
            start=int(row.start+draw-(ends[pos-1] if pos else 0))
            anchors.append(dict(episode=int(row.episode_index),start=start,seed=args.seed+len(anchors)))
        anchors_path.parent.mkdir(parents=True,exist_ok=True)
        anchors_path.write_text(json.dumps(dict(metadata_sha256=identity,anchors=anchors),indent=2)+"\n")
    policy=None
    if args.checkpoint:
        saved_config = json.loads((Path(args.checkpoint)/"config.json").read_text())
        head_config = saved_config["action_head_cfg"]["config"]
        if (saved_config["action_horizon"] != horizon or
            head_config["diffusion_model_cfg"]["num_action_per_block"] != horizon or
            head_config["diffusion_model_cfg"]["num_frame_per_block"] != temporal.num_frame_per_block or
            head_config["num_frames"] != temporal.num_frames):
            raise ValueError("Checkpoint temporal configuration differs from evaluation data")
        import torch
        import torch.distributed as dist
        from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
        from groot.vla.data.schema import EmbodimentTag
        from tianshou.data import Batch
        if not dist.is_initialized():
            # File rendezvous avoids a fixed TCP port collision for independent evaluations.
            rendezvous=args.output_dir.resolve()/f"rendezvous-{os.getpid()}"
            dist.init_process_group("gloo",init_method=f"file://{rendezvous}",rank=0,world_size=1)
        policy=GrootSimPolicy(embodiment_tag=EmbodimentTag.YAM,model_path=args.checkpoint,device=args.device)
    decoder=V3VideoReader()
    targets=[];baseline=[];predictions=[];tasks=[];latencies=[];records=[]
    begin=time.perf_counter()
    for anchor in anchors:
        ep,start=anchor["episode"],anchor["start"]
        rows=storage.read_episode(ep)
        YAMTemporalPlan(start, temporal).validate(len(rows["index"]), rows["task_index"])
        state=rows["observation.state"][start]
        target=rows["action"][start:start+horizon]
        hold=np.repeat(state[None,:],horizon,axis=0)
        task=storage.tasks[int(rows["task_index"][start])]
        record=dict(anchor,task=task)
        if policy is not None:
            obs={f"state.{key}":state[a:b][None,:] for key,(a,b) in SLICES.items()}
            frames=[]
            for camera in CAMERAS:
                path,timestamps,interval=storage.camera_request(ep,camera,[rows["timestamp"][start]])
                frame,_=decoder.read(path,timestamps,interval)
                obs[f"video.{camera}"]=frame
                frames.append(frame[0])
            obs["annotation.task"]=task
            head=policy.trained_model.action_head
            for attr in ["kv_cache1","kv_cache_neg","crossattn_cache","crossattn_cache_neg","clip_feas","ys","language"]:
                setattr(head,attr,None)
            head.current_start_frame=0;head.seed=anchor["seed"]
            random.seed(anchor["seed"]);np.random.seed(anchor["seed"]);torch.manual_seed(anchor["seed"])
            torch.cuda.synchronize();t0=time.perf_counter()
            result,_=policy.lazy_joint_forward_causal(Batch(obs=obs))
            torch.cuda.synchronize();latencies.append(time.perf_counter()-t0)
            values=[]
            for key,(a,b) in SLICES.items():
                value=result.act[f"action.{key}"]
                if torch.is_tensor(value):value=value.detach().cpu().numpy()
                values.append(np.asarray(value).reshape(-1,b-a)[:horizon])
            prediction=np.concatenate(values,axis=1)
            if prediction.shape!=(horizon,14) or not np.isfinite(prediction).all():raise ValueError("Invalid predicted action chunk")
            # Native policy.unapply already inverse-normalizes and adds the anchor state once.
            predictions.append(prediction)
            if len(predictions)<=8:
                from PIL import Image
                mosaic=np.concatenate([np.concatenate([frames[0],frames[2]],axis=1),
                                       np.concatenate([frames[1],np.zeros_like(frames[1])],axis=1)],axis=0)
                Image.fromarray(mosaic).save(args.output_dir/f"sample-{len(predictions)-1}.png")
        targets.append(target);baseline.append(hold);tasks.append(task);records.append(record)
    target=np.asarray(targets);hold=np.asarray(baseline)
    report=dict(metadata_sha256=identity,checkpoint=args.checkpoint,num_anchors=len(anchors),
                horizon=horizon,temporal=temporal.to_dict(),baseline=metrics(hold,target),per_task={},seconds=time.perf_counter()-begin,
                inference_latency_seconds=latencies,anchors=records)
    for task in sorted(set(tasks)):
        selected=np.asarray(tasks)==task
        report["per_task"][task]=dict(anchors=int(selected.sum()),baseline=metrics(hold[selected],target[selected]))
    arrays=dict(target=target,hold_current_state=hold)
    if predictions:
        prediction=np.asarray(predictions);arrays["prediction"]=prediction
        report["model"]=metrics(prediction,target)
        for task in report["per_task"]:
            selected=np.asarray(tasks)==task
            report["per_task"][task]["model"]=metrics(prediction[selected],target[selected])
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(7,2,figsize=(12,16))
        for dim,axis in enumerate(axes.flat):
            axis.plot(target[0,:,dim],label="target");axis.plot(prediction[0,:,dim],label="prediction")
            axis.plot(hold[0,:,dim],label="hold state",linestyle="--");axis.set_title(f"Dimension {dim}")
        axes.flat[0].legend();fig.tight_layout();fig.savefig(args.output_dir/"first-anchor-actions.png");plt.close(fig)
    np.savez_compressed(args.output_dir/"actions.npz",**arrays)
    (args.output_dir/"metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(f"Evaluated {len(anchors)} validation anchors; results: {args.output_dir}")
    if policy is not None:dist.destroy_process_group()


if __name__=="__main__":main()
