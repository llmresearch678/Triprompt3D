"""
losses/dice_loss.py
Multi-label Dice loss for volumetric segmentation.
Also includes combined Dice + Cross-Entropy loss (L_SEG + L_CE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelDiceLoss(nn.Module):
    """
    Soft Dice loss averaged over K classes.

    DSC = 2 * |A ∩ B| / (|A| + |B| + ε)

    Args:
        smooth     : smoothing constant ε for numerical stability
        sigmoid    : apply sigmoid to logits before computing loss
        reduction  : 'mean' or 'none'
    """

    def __init__(self, smooth=1e-5, sigmoid=True, reduction="mean"):
        super().__init__()
        self.smooth    = smooth
        self.sigmoid   = sigmoid
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits : (B, K, H, W, D)  — raw model output
            targets: (B, K, H, W, D)  — binary ground-truth per class

        Returns:
            loss: scalar (mean) or (B, K) tensor
        """
        if self.sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        # Flatten spatial dimensions
        probs   = probs.flatten(2)    # (B, K, N)
        targets = targets.flatten(2)  # (B, K, N)

        intersection = (probs * targets).sum(dim=2)       # (B, K)
        union        = probs.sum(dim=2) + targets.sum(dim=2)  # (B, K)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice    # (B, K)

        if self.reduction == "mean":
            return loss.mean()
        return loss


class CombinedSegLoss(nn.Module):
    """
    L = L_SEG (Dice) + L_CE (Binary Cross-Entropy)

    Used as the primary segmentation objective in TRIPROMPT-3D.
    """

    def __init__(self, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice    = MultiLabelDiceLoss(sigmoid=True)
        self.dice_w  = dice_weight
        self.ce_w    = ce_weight

    def forward(self, logits, targets):
        """
        Args:
            logits : (B, K, H, W, D)
            targets: (B, K, H, W, D)  float binary

        Returns:
            loss_total, loss_dice, loss_ce
        """
        loss_dice = self.dice(logits, targets)
        loss_ce   = F.binary_cross_entropy_with_logits(logits, targets.float())

        loss_total = self.dice_w * loss_dice + self.ce_w * loss_ce
        return loss_total, loss_dice, loss_ce
