import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from safetensors.torch import save_file
from peft import LoraConfig, get_peft_model

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("checkpoint_utils",ROOT/"groot/vla/model/dreamzero/checkpoint_utils.py")
cp=importlib.util.module_from_spec(spec);spec.loader.exec_module(cp)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(defer_lora_injection=True)
        self.model=torch.nn.ModuleDict({k:torch.nn.Linear(4,4) for k in ["q","state_encoder","action_encoder","action_decoder"]})

    def inject_lora_after_loading(self):
        self.requires_grad_(False)
        self.model=get_peft_model(self.model,LoraConfig(r=2,lora_alpha=2,target_modules=["q"]))
        for key in ["state_encoder","action_encoder","action_decoder"]:
            self.model.base_model.model[key].requires_grad_(True)


class TinyModel(torch.nn.Module):
    def __init__(self,config=None):
        super().__init__();self.action_head=TinyHead()

    def forward(self,x):
        model=self.action_head.model
        if hasattr(model,"base_model"):model=model.base_model.model
        return sum(layer(x) for layer in model.values())


class CheckpointTests(unittest.TestCase):
    def test_real_peft_roundtrip_and_optimizer_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/"base";adapter=root/"adapter"
            base.mkdir();adapter.mkdir()
            torch.manual_seed(31)
            original=TinyModel();state=original.state_dict();keys=sorted(state)
            mapping={}
            for i in range(2):
                part={k:state[k].contiguous() for k in keys[i::2]}
                filename=f"model-{i}.safetensors";save_file(part,str(base/filename))
                mapping.update({k:filename for k in part})
            (base/"model.safetensors.index.json").write_text(json.dumps({"weight_map":mapping}))
            trained=cp.load_base_and_inject(TinyModel(),base)
            opt=torch.optim.AdamW([p for p in trained.parameters() if p.requires_grad],lr=.01)
            x=torch.randn(3,4)
            trained(x).square().mean().backward();opt.step();opt.zero_grad()
            trainable={k:v.detach().contiguous() for k,v in trained.named_parameters() if v.requires_grad}
            save_file(trainable,str(adapter/"model.safetensors"))
            (adapter/"config.json").write_text("{}")
            cp.write_checkpoint_manifest(trained,adapter)
            config=SimpleNamespace(action_head_cfg={"config":{}})
            reloaded=cp.reconstruct_adapter(TinyModel,config,adapter)
            for key,value in trained.state_dict().items():
                torch.testing.assert_close(value,reloaded.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(trained(x),reloaded(x),rtol=0,atol=0)
            opt2=torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad],lr=.01)
            opt2.load_state_dict(opt.state_dict())
            # Clone optimizer tensors to avoid sharing mutable state between the two optimizers.
            import copy
            opt2.load_state_dict(copy.deepcopy(opt.state_dict()))
            for model,optimizer in [(trained,opt),(reloaded,opt2)]:
                model(x).square().mean().backward();optimizer.step()
            for a,b in zip(trained.parameters(),reloaded.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            # A missing robot module must fail; frozen base omissions are not the adapter criterion.
            trainable.pop(next(k for k in trainable if "state_encoder" in k))
            save_file(trainable,str(adapter/"model.safetensors"))
            with self.assertRaisesRegex(ValueError,"coverage"):
                cp.load_weights(reloaded,adapter,trainable_only=True)

    def test_shape_and_unexpected_keys_rejected_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=TinyModel();before={k:v.clone() for k,v in model.state_dict().items()}
            broken=dict(before);key=next(iter(broken));broken[key]=torch.zeros(1)
            save_file(broken,str(root/"model.safetensors"))
            with self.assertRaisesRegex(ValueError,"shape mismatch"):cp.load_weights(model,root)
            for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k])
            broken=dict(before);broken["unexpected"]=torch.zeros(1)
            save_file(broken,str(root/"model.safetensors"))
            with self.assertRaisesRegex(ValueError,"unexpected"):cp.load_weights(model,root)


if __name__=="__main__":unittest.main()
