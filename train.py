"""
train.py
Training script for TRIPROMPT-3D.

Full training objective (Eq. 14 in paper):
    L = L_SEG + L_CE + λ1·L_ALIGN(1) + λ2·L_ALIGN(2)
        + λ_3d·L_3D + λ_kl·L_KL

With:
  - Gradient-norm balanced adaptive loss weights (λ1, λ2)
  - Gumbel-Softmax temperature annealing
  - Checkpoint saving every 10 epochs
  - 4× NVIDIA A100 (80GB) — DataParallel or single GPU

Usage:
    python train.py
    python train.py --resume checkpoints/epoch_50.pth
    python train.py --config config.json
"""

import os
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from models           import TriPrompt3D
from datasets.ct_dataset import build_dataloader
from losses           import CombinedSegLoss, SegPromptAlignmentLoss, \
                             PromptPromptAlignmentLoss, GradNormBalancer
from utils            import (
    set_seed, save_checkpoint, load_checkpoint,
    anneal_temperature, MetricLogger
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("train.log")],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Default config (matches paper hyperparameters)
# ─────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data_dir"      : "./data",
    "output_dir"    : "./checkpoints",
    "num_classes"   : 13,
    "img_size"      : [96, 96, 96],
    "sub_vol_size"  : [48, 48, 48],
    "feature_dim"   : 256,
    "latent_dim"    : 128,
    "batch_size"    : 2,
    "num_workers"   : 4,
    "epochs"        : 150,
    "lr"            : 1e-4,
    "weight_decay"  : 1e-5,
    "tau_init"      : 1.0,
    "tau_min"       : 0.07,
    "lambda_tau"    : 3.0,
    "kl_weight"     : 1.0,
    "lambda1_init"  : 1.0,
    "lambda2_init"  : 1.0,
    "save_every"    : 10,
    "seed"          : 42,
    "amp"           : True,
    "pretrained_bb" : None,
    "resume"        : None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train TRIPROMPT-3D")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_config(args):
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    # CLI overrides
    for key in ["data_dir", "epochs", "batch_size", "lr", "seed", "resume"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    return cfg


# ─────────────────────────────────────────────
# Contrastive Loss Helpers
# ─────────────────────────────────────────────

def compute_alignment_losses(loss_dict, align1_fn, align2_fn):
    """Compute both alignment losses from model output dict."""
    loss_a1 = align1_fn(
        loss_dict["Qs_norm"],
        loss_dict["Qa_proj"],
        loss_dict["Qt_proj"],
        loss_dict["Qd_proj"],
    )
    loss_a2 = align2_fn(
        loss_dict["Qa_proj"],
        loss_dict["Qt_proj"],
    )
    return loss_a1, loss_a2


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scaler,
    seg_loss_fn, align1_fn, align2_fn, grad_balancer,
    epoch, cfg, device,
):
    model.train()
    metric_log = MetricLogger()

    for step, batch in enumerate(loader):
        volume      = batch["image"].to(device)         # (B,1,H,W,D)
        sub_vols    = batch["sub_volumes"].to(device)   # (B,K,1,Ha,Wa,Da)
        cross_masks = batch["cross_masks"].to(device)   # (B,K,1,H,W,D)
        gt_multi    = batch["gt_multilabel"].to(device) # (B,K,H,W,D)
        gt_3d       = batch["gt_3d"].to(device)         # (B,K,1,H,W,D)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg["amp"]):
            seg_logits, loss_dict = model(
                volume, sub_vols, cross_masks,
                gt_volume=gt_3d, training=True,
            )

            # Segmentation loss
            loss_total, loss_dice, loss_ce = seg_loss_fn(seg_logits, gt_multi)

            # PDP losses
            loss_3d = loss_dict["loss_3d"]
            loss_kl = loss_dict["loss_kl"]

            # Alignment losses
            loss_a1, loss_a2 = compute_alignment_losses(
                loss_dict, align1_fn, align2_fn
            )

            # Adaptive weights
            lam1, lam2 = grad_balancer.weights

            # Full objective
            loss = (
                loss_total
                + cfg["kl_weight"] * loss_kl
                + loss_3d
                + lam1 * loss_a1
                + lam2 * loss_a2
            )

        if cfg["amp"]:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        metric_log.update(
            loss=loss,
            loss_dice=loss_dice,
            loss_ce=loss_ce,
            loss_kl=loss_kl,
            loss_3d=loss_3d,
            loss_a1=loss_a1,
            loss_a2=loss_a2,
        )

        if step % 20 == 0:
            logger.info(
                f"  Step {step:4d}/{len(loader)} | "
                f"loss={loss.item():.4f} | "
                f"dice={loss_dice.item():.4f} | "
                f"kl={loss_kl.item():.4f}"
            )

    return metric_log.log(epoch, prefix="TRAIN")


@torch.no_grad()
def validate(model, loader, seg_loss_fn, epoch, device, cfg):
    import numpy as np
    from utils import compute_dice

    model.eval()
    all_dsc = []
    total_loss = 0.0

    for batch in loader:
        volume      = batch["image"].to(device)
        sub_vols    = batch["sub_volumes"].to(device)
        cross_masks = batch["cross_masks"].to(device)
        gt_multi    = batch["gt_multilabel"].to(device)
        gt_3d       = batch["gt_3d"].to(device)

        seg_logits, alpha = model(
            volume, sub_vols, cross_masks,
            gt_volume=None, training=False,
        )

        loss, _, _ = seg_loss_fn(seg_logits, gt_multi)
        total_loss += loss.item()

        # DSC
        preds = (torch.sigmoid(seg_logits) > 0.5).cpu().numpy()
        gts   = gt_multi.cpu().numpy()
        for b in range(preds.shape[0]):
            dsc = compute_dice(preds[b], gts[b])
            all_dsc.append(dsc)

    mean_dsc = np.nanmean(all_dsc)
    logger.info(f"[VAL] Epoch {epoch:03d} | DSC={mean_dsc:.4f} | "
                f"loss={total_loss/len(loader):.4f}")
    return {"val_dsc": mean_dsc, "val_loss": total_loss / len(loader)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args)

    set_seed(cfg["seed"])

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────
    train_loader = build_dataloader(
        data_dir=cfg["data_dir"], split="train",
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        num_classes=cfg["num_classes"],
        img_size=tuple(cfg["img_size"]),
        seed=cfg["seed"],
    )
    val_loader = build_dataloader(
        data_dir=cfg["data_dir"], split="test",
        batch_size=1,
        num_workers=cfg["num_workers"],
        num_classes=cfg["num_classes"],
        img_size=tuple(cfg["img_size"]),
        seed=cfg["seed"],
    )

    # ── Model ─────────────────────────────────────────────
    model = TriPrompt3D(
        num_classes=cfg["num_classes"],
        img_size=tuple(cfg["img_size"]),
        feature_dim=cfg["feature_dim"],
        latent_dim=cfg["latent_dim"],
        tau_init=cfg["tau_init"],
        kl_weight=cfg["kl_weight"],
        pretrained_bb=cfg["pretrained_bb"],
    ).to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Optimizer & Scheduler ─────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=cfg["amp"])

    # ── Loss Functions ─────────────────────────────────────
    seg_loss_fn = CombinedSegLoss(dice_weight=1.0, ce_weight=1.0).to(device)
    align1_fn   = SegPromptAlignmentLoss(temperature=cfg["tau_min"]).to(device)
    align2_fn   = PromptPromptAlignmentLoss(temperature=cfg["tau_min"]).to(device)
    grad_bal    = GradNormBalancer(cfg["lambda1_init"], cfg["lambda2_init"])

    # ── Resume ────────────────────────────────────────────
    start_epoch = 0
    best_dsc    = 0.0

    resume_path = cfg.get("resume") or args.resume
    if resume_path and os.path.isfile(resume_path):
        start_epoch, metrics = load_checkpoint(
            resume_path, model, optimizer, device
        )
        best_dsc = metrics.get("val_dsc", 0.0)

    # ── Training Loop ─────────────────────────────────────
    logger.info(f"Starting training: {start_epoch} → {cfg['epochs']} epochs")

    for epoch in range(start_epoch, cfg["epochs"]):

        # Anneal Gumbel-Softmax temperature
        tau = anneal_temperature(
            epoch, cfg["epochs"],
            cfg["tau_init"], cfg["lambda_tau"], cfg["tau_min"]
        )
        m = model.module if hasattr(model, "module") else model
        m.set_tau(tau)

        # Update gradient-norm balancing weights every epoch
        # (simplified: update every 5 epochs to save compute)
        if epoch > 0 and epoch % 5 == 0:
            logger.info(f"[GradNorm] λ1={grad_bal.lambda1:.3f} λ2={grad_bal.lambda2:.3f}")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler,
            seg_loss_fn, align1_fn, align2_fn, grad_bal,
            epoch, cfg, device,
        )

        # Validate
        val_metrics = validate(model, val_loader, seg_loss_fn, epoch, device, cfg)

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg["save_every"] == 0:
            ckpt_path = os.path.join(cfg["output_dir"], f"epoch_{epoch+1:03d}.pth")
            save_checkpoint(
                epoch + 1, model, optimizer, scheduler,
                {**train_metrics, **val_metrics}, ckpt_path
            )

        # Save best
        if val_metrics["val_dsc"] > best_dsc:
            best_dsc = val_metrics["val_dsc"]
            best_path = os.path.join(cfg["output_dir"], "best_model.pth")
            save_checkpoint(
                epoch + 1, model, optimizer, scheduler,
                {**train_metrics, **val_metrics}, best_path
            )
            logger.info(f"[Best] DSC={best_dsc:.4f} saved → {best_path}")

    logger.info(f"Training complete. Best DSC: {best_dsc:.4f}")


if __name__ == "__main__":
    main()
