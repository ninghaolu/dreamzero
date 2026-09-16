"""Shared official-YAM overrides for ABC training, inspection and validation."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def overrides(metadata_root):
    wan = os.environ.get("WAN_CKPT_DIR", str(ROOT/"checkpoints/Wan2.1-I2V-14B-480P"))
    tokenizer = os.environ.get("TOKENIZER_DIR", str(ROOT/"checkpoints/umt5-xxl"))
    base = os.environ.get("AGIBOT_CKPT_DIR", str(ROOT/"checkpoints/DreamZero-AgiBot"))
    return [
        "data=dreamzero/abc_yam_v3_relative", "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf",
        "model/dreamzero/transform=dreamzero_cotrain",
        "train_architecture=lora", "num_frames=33", "action_horizon=24", "num_views=3",
        "num_frame_per_block=2", "num_action_per_block=24", "num_state_per_block=1",
        "max_chunk_size=4", "frame_seqlen=880", "seed=42",
        "image_resolution_width=320", "image_resolution_height=176",
        "learning_rate=1e-5", "weight_decay=1e-5",
        "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json",
        "warmup_ratio=0.05", "per_device_train_batch_size=1",
        "gradient_accumulation_steps=1", "max_steps=50", "save_steps=10", "save_total_limit=5",
        "save_strategy=steps", "save_lora_only=true", "upload_checkpoints=false",
        "bf16=true", "tf32=true", "eval_bf16=true", "dataloader_pin_memory=false",
        "dataloader_num_workers=0", "report_to=none", "wandb_project=dreamzero-abc",
        f"abc_metadata_root={Path(metadata_root).resolve()}",
        f"output_dir={os.environ.get('OUTPUT_DIR', str(ROOT/'outputs/abc_yam_v3_smoke'))}",
        f"dit_version={wan}", f"text_encoder_pretrained_path={wan}/models_t5_umt5-xxl-enc-bf16.pth",
        f"image_encoder_pretrained_path={wan}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"vae_pretrained_path={wan}/Wan2.1_VAE.pth", f"tokenizer_path={tokenizer}",
        f"pretrained_model_path={base}", "++action_head_cfg.config.skip_component_loading=true",
        "++action_head_cfg.config.defer_lora_injection=true",
    ]


def compose_abc(metadata_root, extra=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT/"groot/vla/configs")):
        return compose(config_name="conf", overrides=overrides(metadata_root)+list(extra))


if __name__ == "__main__":
    # NUL-separated arguments preserve Bash array boundaries.
    import sys
    sys.stdout.write("\0".join(overrides(sys.argv[1]))+"\0")
