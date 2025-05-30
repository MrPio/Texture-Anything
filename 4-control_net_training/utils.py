import torch.nn.functional as F

def masked_mse_loss(pred, targ, mask):
    """Compute the masked MSE loss

    Args:
        pred: shape=(N, C, W, H)
        targ: shape=(N, C, W, H)
        mask: shape=(N, W, H)
    """
    _, channels, _, _ = pred.shape
    # Repeat the channels
    mask = mask.unsqueeze(1).repeat(1, channels, 1, 1)
    # Resize the mask to args.resolution
    mask = F.interpolate(mask.float(), size=pred.shape[2:], mode="nearest")
    loss_elems = (F.mse_loss(pred.float(), targ.float(), reduction="none")) * mask
    return loss_elems.sum() / (mask.sum() + 1e-6)
