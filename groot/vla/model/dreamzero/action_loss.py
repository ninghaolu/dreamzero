import torch


def masked_action_mse(prediction, target, action_mask, has_real_action):
    """Unreduced MSE preserving the official padded-dimension loss scaling."""
    if prediction.ndim != 3 or has_real_action.shape != prediction.shape[:1]:
        raise ValueError("Expected actions [B,T,D] and real-action flags [B]")
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float(), reduction="none")
    return loss * action_mask * has_real_action[:, None, None].to(loss.dtype)
