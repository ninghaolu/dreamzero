import importlib.util
from pathlib import Path
import unittest
import torch

spec=importlib.util.spec_from_file_location("action_loss",Path(__file__).resolve().parents[1]/"groot/vla/model/dreamzero/action_loss.py")
loss=importlib.util.module_from_spec(spec);spec.loader.exec_module(loss)


class MaskTests(unittest.TestCase):
    def test_mixed_batch_masks_gradients_and_keeps_original_scaling(self):
        for batch in [1,4]:
            pred=torch.ones(batch,96,32,requires_grad=True)
            flags=torch.tensor([i%2==0 for i in range(batch)])
            mask=torch.zeros_like(pred);mask[:,:,:14]=1
            mse=loss.masked_action_mse(pred,torch.zeros_like(pred),mask,flags)
            mse.mean().backward()
            expected=mask*flags[:,None,None]*2/pred.numel()
            torch.testing.assert_close(pred.grad,expected)
            self.assertEqual(int(torch.count_nonzero(pred.grad)),int(flags.sum())*96*14)


if __name__=="__main__":unittest.main()
