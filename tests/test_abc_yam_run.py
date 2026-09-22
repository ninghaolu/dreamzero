import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/train'))
from validate_abc_yam_run import validate_run


class RunValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inventory = {
            'action_head.model.weight': {'shape': [1], 'trainable': True},
            **{f'action_head.{name}.weight': {'shape': [1], 'trainable': False}
               for name in ('text_encoder', 'image_encoder', 'vae')},
        }
        self.write('parameter_inventory.json', self.inventory)
        self.write('config.json', {'action_head_cfg': {'config': {'train_architecture': 'full'}}})
        self.write('trainer_state.json', {'global_step': 3})
        self.write('resources-rank-0.json', {'global_step': 3, 'peak_gpu_allocated_bytes': 100})
        (self.root / 'loss_log.jsonl').write_text('{"step": 1, "loss": 0.2}\n')
        self.save_weights(self.inventory)

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def save_weights(self, inventory):
        header = json.dumps({key: {'shape': value['shape']} for key, value in inventory.items()}).encode()
        (self.root / 'model.safetensors').write_bytes(struct.pack('<Q', len(header))+header)

    def test_completed_full_run(self):
        self.assertTrue(validate_run(self.root, ranks=1)['passed'])

    def test_frozen_diffusion_model_is_rejected(self):
        self.inventory['action_head.model.weight']['trainable'] = False
        self.write('parameter_inventory.json', self.inventory)
        with self.assertRaisesRegex(ValueError, 'parameter status'):
            validate_run(self.root, ranks=1)

    def test_trainable_only_checkpoint_is_rejected(self):
        self.save_weights({'action_head.model.weight': self.inventory['action_head.model.weight']})
        with self.assertRaisesRegex(ValueError, 'Full checkpoint missing'):
            validate_run(self.root, ranks=1)

    def test_incomplete_or_nonfinite_run_is_rejected(self):
        self.write('resources-rank-0.json', {'global_step': 2, 'peak_gpu_allocated_bytes': 100})
        with self.assertRaisesRegex(ValueError, 'optimizer steps'):
            validate_run(self.root, ranks=1)
        self.write('resources-rank-0.json', {'global_step': 3, 'peak_gpu_allocated_bytes': 100})
        (self.root / 'loss_log.jsonl').write_text('{"loss": NaN}\n')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            validate_run(self.root, ranks=1)
