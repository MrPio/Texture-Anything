import torch.nn.functional as F

def masked_mse_loss(pred, targ, mask):
    """Compute the masked MSE loss

    Args:
        pred (torch.tensor): CNet output, shape=(N, C, W, H)
        targ (torch.tensor): Ground truth, shape=(N, C, W, H)
        mask (torch.tensor): UV mask, shape=(N, W, H)
    """
    _, channels, _, _ = pred.shape
    # Repeat the channels (the mask is B/W)
    mask = mask.float().unsqueeze(1).repeat(1, channels, 1, 1)
    # Resize the mask to args.resolution
    mask = F.interpolate(mask, size=pred.shape[2:], mode="nearest")
    loss_elems = (F.mse_loss(pred.float(), targ.float(), reduction="none")) * mask
    return loss_elems.sum() / (mask.sum() + 1e-6)
