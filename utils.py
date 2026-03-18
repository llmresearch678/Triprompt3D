"""
utils.py
Utility functions for TRIPROMPT-3D.

Includes:
  - set_seed()          : deterministic reproducibility
  - save_checkpoint()   : save model + optimizer state
  - load_checkpoint()   : resume training exactly from saved epoch
  - compute_dice()      : per-class DSC computation
  - compute_hd95()      : 95th-percentile Hausdorff Distance
  - compute_assd()      : Average Symmetric Surface Distance
  - anneal_temperature(): Gumbel-Softmax temperature schedule
  - AverageMeter        : running mean tracker
  - MetricLogger        : structured per-epoch logging
"""

import os
import math
import random
import logging
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed=42):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"[Seed] Fixed to {seed}")


# ─────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────

def save_checkpoint(epoch, model, optimizer, scheduler, metrics, path):
    """
    Save full training state.

    Args:
        epoch    : current epoch number
        model    : nn.Module
        optimizer: torch optimizer
        scheduler: LR scheduler
        metrics  : dict of current metrics (e.g., {'val_dsc': 0.92})
        path     : str or Path to save file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch"    : epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics"  : metrics,
    }
    torch.save(state, path)
    logger.info(f"[Checkpoint] Saved epoch {epoch} → {path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Resume training exactly from a saved checkpoint.

    Args:
        checkpoint_path: str or Path
        model          : nn.Module (modified in-place)
        optimizer      : optional optimizer (modified in-place)
        device         : target device

    Returns:
        start_epoch: int   — epoch to resume from
        metrics    : dict  — metrics at checkpoint
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing:
        logger.warning(f"[Checkpoint] Missing keys: {missing}")
    if unexpected:
        logger.warning(f"[Checkpoint] Unexpected keys: {unexpected}")

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    epoch   = ckpt.get("epoch", 0) + 1
    metrics = ckpt.get("metrics", {})
    logger.info(f"[Checkpoint] Resumed from epoch {epoch - 1} at {checkpoint_path}")
    return epoch, metrics


# ─────────────────────────────────────────────
# Temperature Annealing
# ─────────────────────────────────────────────

def anneal_temperature(epoch, total_epochs, tau0=1.0, lambda_tau=3.0, tau_min=0.07):
    """
    Gumbel-Softmax temperature annealing schedule (Eq. 19 in paper):
        τ(t) = max(τ0 · exp(−λ_τ · t/T), τ_min)

    Args:
        epoch       : current epoch t
        total_epochs: T
        tau0        : initial temperature (1.0)
        lambda_tau  : decay rate (3.0)
        tau_min     : minimum temperature (0.07)

    Returns:
        tau: float
    """
    tau = tau0 * math.exp(-lambda_tau * epoch / total_epochs)
    return max(tau, tau_min)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_dice(pred, target, smooth=1e-5):
    """
    Compute DSC per class.

    Args:
        pred  : (K, H, W, D) binary predictions
        target: (K, H, W, D) binary ground truth

    Returns:
        dsc: (K,) numpy array
    """
    pred   = pred.astype(bool)
    target = target.astype(bool)
    K      = pred.shape[0]
    dsc    = np.zeros(K)

    for c in range(K):
        inter = (pred[c] & target[c]).sum()
        union = pred[c].sum() + target[c].sum()
        if union == 0:
            dsc[c] = 1.0   # both empty → perfect
        else:
            dsc[c] = (2 * inter + smooth) / (union + smooth)

    return dsc


def compute_hd95(pred, target, voxel_spacing=(1.5, 1.5, 1.5)):
    """
    Compute 95th-percentile Hausdorff Distance per class.

    Requires scipy.

    Args:
        pred          : (K, H, W, D) binary
        target        : (K, H, W, D) binary
        voxel_spacing : physical spacing in mm

    Returns:
        hd95: (K,) numpy array (mm)
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scipy is required for HD95 computation.")

    K    = pred.shape[0]
    hd95 = np.full(K, np.nan)

    for c in range(K):
        p  = pred[c].astype(bool)
        t  = target[c].astype(bool)

        if p.sum() == 0 or t.sum() == 0:
            hd95[c] = np.nan
            continue

        # Surface points via erosion
        def surface(mask):
            from scipy.ndimage import binary_erosion
            return mask ^ binary_erosion(mask)

        surf_p = surface(p)
        surf_t = surface(t)

        dist_t = distance_transform_edt(~t, sampling=voxel_spacing)
        dist_p = distance_transform_edt(~p, sampling=voxel_spacing)

        forward  = dist_t[surf_p]
        backward = dist_p[surf_t]

        all_dists = np.concatenate([forward, backward])
        hd95[c]   = np.percentile(all_dists, 95)

    return hd95


def compute_assd(pred, target, voxel_spacing=(1.5, 1.5, 1.5)):
    """
    Compute Average Symmetric Surface Distance per class.

    Args:
        pred          : (K, H, W, D) binary
        target        : (K, H, W, D) binary
        voxel_spacing : physical spacing in mm

    Returns:
        assd: (K,) numpy array (mm)
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        raise ImportError("scipy is required for ASSD computation.")

    K    = pred.shape[0]
    assd = np.full(K, np.nan)

    for c in range(K):
        p = pred[c].astype(bool)
        t = target[c].astype(bool)

        if p.sum() == 0 or t.sum() == 0:
            assd[c] = np.nan
            continue

        def surface(mask):
            return mask ^ binary_erosion(mask)

        surf_p = surface(p)
        surf_t = surface(t)
        n_p    = surf_p.sum()
        n_t    = surf_t.sum()

        dist_t  = distance_transform_edt(~t, sampling=voxel_spacing)
        dist_p  = distance_transform_edt(~p, sampling=voxel_spacing)

        d_p2t   = dist_t[surf_p].sum()
        d_t2p   = dist_p[surf_t].sum()

        assd[c] = (d_p2t + d_t2p) / (n_p + n_t)

    return assd


# ─────────────────────────────────────────────
# AverageMeter & MetricLogger
# ─────────────────────────────────────────────

class AverageMeter:
    """Tracks running mean and sum of a scalar value."""

    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


class MetricLogger:
    """Logs and prints metrics per epoch."""

    def __init__(self):
        self.meters = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k)
            val = v.item() if torch.is_tensor(v) else float(v)
            self.meters[k].update(val)

    def log(self, epoch, prefix=""):
        parts = [f"Epoch {epoch:03d}"]
        for k, m in self.meters.items():
            parts.append(f"{k}={m.avg:.4f}")
        msg = f"[{prefix}] " + " | ".join(parts)
        logger.info(msg)
        print(msg)
        return {k: m.avg for k, m in self.meters.items()}

    def reset(self):
        for m in self.meters.values():
            m.reset()
