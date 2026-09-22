"""Compose the real ABC launcher config without allocating model weights."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/train'))
from abc_yam_config import compose_abc
from validate_abc_yam_launch import validate_config


class ConfigTests(unittest.TestCase):
    def test_both_full_finetuning_presets(self):
        for name, horizon, stride in [('h24', 24, 3), ('h48', 48, 6)]:
            cfg = compose_abc('/unused', temporal=name)
            temporal = validate_config(cfg)
            self.assertEqual(temporal.block_length, horizon)
            self.assertEqual(temporal.video_stride, stride)
            self.assertEqual(cfg.model.config.action_horizon, horizon)
            self.assertEqual(cfg.model_specific_transform.action_horizon, horizon)
            self.assertEqual(list(cfg.modality_config_yam.state.eval_delta_indices), [0])
            self.assertEqual(cfg.train_architecture, 'full')
            self.assertFalse(cfg.save_lora_only)

    def test_inconsistent_overrides_rejected(self):
        for override in [
            'action_horizon=24', 'num_action_per_block=24', 'abc_video_stride=3',
            'num_frame_per_block=1', 'num_state_per_block=2', 'max_chunk_size=2', 'num_frames=17',
            'train_architecture=lora', 'save_lora_only=true',
            'train_dataset.dataset_kwargs.video_stride=3',
            'action_head_cfg.config.diffusion_model_cfg.num_action_per_block=24',
            'action_head_cfg.config.tune_diffusion_model=false',
            'modality_config_yam.action.delta_indices=[0,1]',
        ]:
            with self.subTest(override=override):
                cfg = compose_abc('/unused', [override], temporal='h48')
                with self.assertRaises(ValueError):
                    validate_config(cfg)


if __name__ == '__main__':
    unittest.main()
