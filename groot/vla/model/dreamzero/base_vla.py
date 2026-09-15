from dataclasses import dataclass, field
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


@dataclass
class VLAConfig(PretrainedConfig):
    model_type = "vla"
    backbone_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Backbone configuration."}
    )

    action_head_cfg: PretrainedConfig = field(
        default=None, metadata={"help": "Action head configuration."}
    )

    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})

    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class VLA(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = VLAConfig
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    see discussion at https://nvidia.slack.com/archives/C07T1V7L886/p1732550624654139
    """

    def __init__(
        self,
        config: VLAConfig,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)
        super().__init__(config)
        self.backbone = instantiate(config.backbone_cfg)
        self.action_head = instantiate(config.action_head_cfg)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def validate_inputs(self, inputs):
        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] % self.action_horizon == 0
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):

        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def lazy_joint_video_action_causal_gt_cond(
        self,
        inputs: dict,
        latent_video: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        action_head_outputs = self.action_head.lazy_joint_video_action_causal_gt_cond(backbone_outputs, action_inputs, latent_video=latent_video)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def lazy_joint_video_action_efficient(
        self,
        inputs: dict,
        prompt_embs: torch.Tensor | None = None,
        prompt_emb_nega: torch.Tensor | None = None,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.lazy_joint_video_action_efficient(backbone_outputs, action_inputs, prompt_embs=prompt_embs, prompt_emb_nega=prompt_emb_nega)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def gt_video_action_pred(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs = self.action_head.gt_video_action_pred(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs
    
    def get_language(
        self,
        inputs: dict,
    ) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone.generate(backbone_inputs)
        return backbone_outputs

    def get_video(
        self,
        inputs: dict,
    ) -> BatchFeature:
        _, video_inputs = self.prepare_input(inputs)
        video_outputs = self.action_head.get_video(video_inputs)
        return video_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs


    @classmethod
    def from_pretrained_for_tuning(
        cls, pretrained_model_name_or_path, config=None, device_map="auto",
        dtype=torch.bfloat16, offload_state_dict=True, lora_weights_path=None,
    ):
        from .checkpoint_utils import load_base_and_inject, load_weights
        if config is None:
            raise ValueError("A configuration is required for tuning")
        inner = config.action_head_cfg.get("config", config.action_head_cfg)
        inner.update(defer_lora_injection=True, skip_component_loading=True)
        model = load_base_and_inject(cls(config), pretrained_model_name_or_path)
        if lora_weights_path is not None:
            load_weights(model, lora_weights_path, trainable_only=True)
        return model

    @classmethod
    def load_lora(cls, pretrained_model_name_or_path, config=None, base_path=None):
        import json
        from pathlib import Path
        from .checkpoint_utils import reconstruct_adapter
        if config is None:
            config = VLAConfig(**json.loads((Path(pretrained_model_name_or_path)/"config.json").read_text()))
        return reconstruct_adapter(cls, config, pretrained_model_name_or_path, base_path)

    def load_lora_weight(self, pretrained_model_name_or_path):
        from .checkpoint_utils import load_weights
        return load_weights(self, pretrained_model_name_or_path, trainable_only=True)

    @classmethod
    def from_config_with_lora_weights(cls, config, pretrained_model_path):
        return cls.load_lora(pretrained_model_path, config=config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None):
        import json
        from pathlib import Path
        from .checkpoint_utils import load_base_and_inject
        root = Path(pretrained_model_name_or_path)
        if (root / "checkpoint_manifest.json").exists():
            return cls.load_lora(root, config=config)
        if config is None:
            config = VLAConfig(**json.loads((root/"config.json").read_text()))
        inner = config.action_head_cfg.get("config", config.action_head_cfg)
        inner.update(defer_lora_injection=True, skip_component_loading=True)
        return load_base_and_inject(cls(config), root)

    def post_initialize(self):
        self.action_head.post_initialize()

    def parallelize(self, device_mesh: DeviceMesh):
        self.action_head.parallelize(device_mesh=device_mesh)


class CotrainVLA(VLA):

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:
        if "cotrain" in inputs and inputs["cotrain"]:
            return self.backbone.cotrain(inputs)
        return super().forward(inputs)


def create_vla_with_pretrained_action_head(pretrained_vla_path: str, config: VLAConfig):
    # 1. Instantiate a new VLAModel
    vla = VLA(config)

    # 2. Load the pretrained VLAModel
    pretrained_vla = VLA.from_pretrained(pretrained_vla_path)

    # 3. Replace the action head in the new VLAModel with the pretrained action head
    vla.action_head = pretrained_vla.action_head

    # 4. Replace the action head config in the new VLAModel with the pretrained action head config
    vla.config.action_head_cfg = pretrained_vla.config.action_head_cfg

    # 5. Return the new VLAModel
    return vla


# register
AutoConfig.register("vla", VLAConfig)
AutoModel.register(VLAConfig, VLA)
