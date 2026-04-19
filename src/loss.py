import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits, gt, eps=1e-6) -> torch.Tensor:
    """
    This function computes the dice loss, with an eps term, between logits and gt.

    Args:
        logits (torch.Tensor): logits tensor
        gt (torch.Tensor): gt tensor
        eps (float): eps term
    Returns:
        torch.Tensor: Dice loss, defined as 1 - Dice Similarity Coefficient (DSC)
    """
    # convert logits to probabilities
    probs = torch.sigmoid(logits) # (B, 1, H, W)

    # flattening
    probs = probs.flatten(start_dim=1) # (B, H*W)
    gt = gt.flatten(start_dim=1) # (B, H*W)

    # compute intersection per sample, for DSC
    intersection = (probs * gt).sum(dim=1) # (B,), TP

    # compute sums per sample, for DSC
    probs_sum = probs.sum(dim=1) # (B,)
    gt_sum = gt.sum(dim=1) # (B,)

    # DSC per sample
    dsc = (2.0 * intersection + eps) / (probs_sum + gt_sum + eps) # (B,)

    # Dice loss per sample
    return 1.0 - dsc # (B,)


class FocalBCEDiceLoss(nn.Module):
    """
    This function defines focal bce dice loss, whose computation needs dice_loss().

    Args:
        dice_weight (float): dice weight parameter.
        gamma (float): gamma parameter.
        alpha (float): alpha parameter.
        neg_ohem_weight (float): neg ohem weight parameter.
        neg_topk (int): top k negative samples.
    Returns:

    """
    def __init__(self, dice_weight=1.0, alpha=0.25, gamma=2.0, neg_ohem_weight=0.05, neg_topk=1024):
        super().__init__()
        self.dice_weight = dice_weight
        self.alpha = alpha
        self.gamma = gamma
        self.neg_ohem_weight = neg_ohem_weight
        self.neg_topk = neg_topk

    def forward(self, logits, gt):
        gt = gt.float()

        # ----------------- FOCAL BCE WITH LOGITS, ON POSITIVES AND NEGATIVES

        bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none") # (B, 1, H, W)

        # convert logits to probabilities
        probs = torch.sigmoid(logits) # (B, 1, H, W)

        # pt = probability of the true class
        pt = probs * gt + (1.0 - probs) * (1.0 - gt) # (B, 1, H, W)

        # defin alpha weighting
        alpha_t = self.alpha * gt + (1.0 - self.alpha) * (1.0 - gt) # (B, 1, H, W)

        # focal BCE formula
        focal_bce = alpha_t * (1.0 - pt).pow(self.gamma) * bce # (B, 1, H, W)

        # convert to scalar
        focal_bce = focal_bce.mean() # scalar
        assert focal_bce.ndim == 0, "focal_bce should be a scalar"

        # ----------------- DICE LOSS, ON POSITIVES ONLY

        # Boolean vector the same shape as a batch to restrict Dice computation to positives only
        is_pos = gt.sum(dim=(1, 2, 3)) > 0 # (B,) bool

        if is_pos.any():
            # compute Dice loss on pos samples only, create tensor with the same length of # pos in the batch
            dice_per_sample = dice_loss(logits[is_pos], gt[is_pos]) # (Bpos,)

            # define scalar that averages dice_per_sample losses
            dice = dice_per_sample.mean() # scalar
        else:
            # if there is no positive in the batch, loss is 0.0
            dice = logits.new_tensor(0.0)

        # ----------------- OHEM TOP-K BCE, ON NEGATIVES ONLY

        # to restrict Dice computation to negatives only
        is_neg = ~is_pos # (B,) bool

        if is_neg.any() and self.neg_ohem_weight > 0:
            # select only BCE loss values for neg samples in the batch
            bce_neg = bce[is_neg] # (Bneg, 1, H, W)

            # flatten pixels, to later select the ones with the highest BCE loss
            flat = bce_neg.flatten(start_dim=1) # (Bneg, H*W)

            # define k, make sure it's never above H*W
            k = min(self.neg_topk, flat.shape[1])

            # for each negative, pick the k pixels with the highest BCE loss
            topk_vals, _ = torch.topk(flat, k, dim=1) # (Bneg, k)

            # average per batch
            neg_ohem = topk_vals.mean()
        else:
            # if there is no negative in the batch, loss is 0.0
            neg_ohem = logits.new_tensor(0.0)

        # total loss
        return focal_bce + self.dice_weight * dice + self.neg_ohem_weight * neg_ohem