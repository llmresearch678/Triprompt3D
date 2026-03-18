"""
inference.py
Inference script for TRIPROMPT-3D.

Loads a trained checkpoint and runs voxel-wise multi-label
segmentation on all volumes in the input directory.
Saves predictions as NIfTI (.nii.gz) files.

Usage:
    python inference.py
    python inference.py --checkpoint checkpoints/best_model.pth
    python inference.py --checkpoint checkpoints/best_model.pth \\
                        --input_dir ./data/test/images \\
                        --output_dir ./predictions
"""

import os
import argparse
import logging
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from pathlib import Path

from models          import TriPrompt3D
from utils           import set_seed, load_checkpoint, compute_dice, compute_hd95

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="TRIPROMPT-3D Inference")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--input_dir",   type=str, default="./data/test/images")
    parser.add_argument("--mask_dir",    type=str, default="./data/test/masks",
                        help="Optional: ground truth masks for evaluation")
    parser.add_argument("--output_dir",  type=str, default="./predictions")
    parser.add_argument("--num_classes", type=int, default=13)
    parser.add_argument("--img_size",    type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--latent_dim",  type=int, default=128)
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="Sigmoid threshold for binary prediction")
    parser.add_argument("--save_probs",  action="store_true",
                        help="Also save probability maps")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def preprocess_volume(nib_img, img_size, hu_min=-1000, hu_max=1000):
    """
    Preprocess a single NIfTI volume for inference.

    Returns:
        tensor      : (1, 1, H, W, D) float32
        affine      : (4, 4) original affine for saving predictions
        orig_shape  : original spatial shape
    """
    data   = nib_img.get_fdata().astype(np.float32)
    affine = nib_img.affine

    # HU clipping + normalization
    data = np.clip(data, hu_min, hu_max)
    data = (data - hu_min) / (hu_max - hu_min)

    orig_shape = data.shape

    # Resize to model input size
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)   # (1,1,H,W,D)
    tensor = F.interpolate(
        tensor,
        size=img_size,
        mode="trilinear",
        align_corners=False,
    )
    return tensor, affine, orig_shape


def postprocess_prediction(pred_tensor, orig_shape, threshold=0.5):
    """
    Upsample prediction back to original volume shape and threshold.

    Args:
        pred_tensor: (1, K, H, W, D) sigmoid probabilities
        orig_shape : (H0, W0, D0) original spatial shape
        threshold  : binary threshold

    Returns:
        pred_binary: (K, H0, W0, D0) uint8 numpy array
        pred_probs : (K, H0, W0, D0) float32 numpy array
    """
    pred_up = F.interpolate(
        pred_tensor,
        size=orig_shape,
        mode="trilinear",
        align_corners=False,
    )   # (1, K, H0, W0, D0)

    probs  = pred_up.squeeze(0).cpu().numpy()            # (K, H0, W0, D0)
    binary = (probs > threshold).astype(np.uint8)

    return binary, probs


def build_dummy_prompts(num_classes, img_size, sub_vol_size=(48, 48, 48), device="cpu"):
    """
    Build placeholder prompt inputs for inference.
    In production, sub_volumes come from the actual image crops
    and cross_masks from a held-out population set.
    """
    B = 1
    K = num_classes
    sub_vols    = torch.zeros(B, K, 1, *sub_vol_size, device=device)
    cross_masks = torch.zeros(B, K, 1, *img_size, device=device)
    return sub_vols, cross_masks


@torch.no_grad()
def run_inference(model, volume_tensor, sub_vols, cross_masks, device):
    """
    Run model forward pass for a single volume.

    Returns:
        seg_logits: (1, K, H, W, D)
        alpha     : (1, K) reliability scores
    """
    volume_tensor = volume_tensor.to(device)
    sub_vols      = sub_vols.to(device)
    cross_masks   = cross_masks.to(device)

    seg_logits, alpha = model(
        volume_tensor, sub_vols, cross_masks,
        training=False,
    )
    probs = torch.sigmoid(seg_logits)
    return probs, alpha


def save_prediction(binary, probs, affine, output_path, save_probs=False):
    """Save binary prediction (and optionally probabilities) as NIfTI."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Multi-label: argmax over classes (background = 0)
    # Each channel c corresponds to class c+1
    multilabel = np.zeros(binary.shape[1:], dtype=np.int16)
    for c in range(binary.shape[0]):
        multilabel[binary[c] == 1] = c + 1

    nib.save(
        nib.Nifti1Image(multilabel, affine),
        str(output_path),
    )
    logger.info(f"  Saved → {output_path}")

    if save_probs:
        prob_path = str(output_path).replace(".nii.gz", "_probs.nii.gz")
        # Save channel 0 (first class) as example; adjust as needed
        nib.save(
            nib.Nifti1Image(probs[0], affine),
            prob_path,
        )


def main():
    args   = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    img_size = tuple(args.img_size)

    # ── Load Model ────────────────────────────────────────
    model = TriPrompt3D(
        num_classes=args.num_classes,
        img_size=img_size,
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
    ).to(device)
    model.eval()

    if os.path.isfile(args.checkpoint):
        load_checkpoint(args.checkpoint, model, device=device)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}. Running with random weights.")

    # ── Discover input files ──────────────────────────────
    input_dir = Path(args.input_dir)
    vol_files = sorted(
        list(input_dir.glob("*.nii.gz")) + list(input_dir.glob("*.nii"))
    )
    logger.info(f"Found {len(vol_files)} volumes in {input_dir}")

    # ── Per-volume inference ──────────────────────────────
    all_dsc = []
    has_gt  = os.path.isdir(args.mask_dir)

    for vol_path in vol_files:
        case_id = vol_path.name.replace(".nii.gz", "").replace(".nii", "")
        logger.info(f"Processing: {case_id}")

        # Load and preprocess
        nib_img = nib.load(str(vol_path))
        volume_tensor, affine, orig_shape = preprocess_volume(nib_img, img_size)

        # Build dummy prompts (replace with real sub-volumes for best results)
        sub_vols, cross_masks = build_dummy_prompts(
            args.num_classes, img_size, device=device
        )

        # Inference
        probs, alpha = run_inference(model, volume_tensor, sub_vols, cross_masks, device)

        # Log reliability scores
        alpha_np = alpha.squeeze(0).cpu().numpy()
        logger.info(f"  Reliability scores (mean): {alpha_np.mean():.3f}")

        # Post-process
        binary, probs_np = postprocess_prediction(probs, orig_shape, args.threshold)

        # Save prediction
        out_path = Path(args.output_dir) / f"{case_id}.nii.gz"
        save_prediction(binary, probs_np, affine, out_path, args.save_probs)

        # Evaluate if GT available
        if has_gt:
            mask_path = Path(args.mask_dir) / vol_path.name
            if mask_path.exists():
                gt_nib  = nib.load(str(mask_path))
                gt_data = gt_nib.get_fdata().astype(np.int16)

                gt_binary = np.stack([
                    (gt_data == c + 1).astype(np.uint8)
                    for c in range(args.num_classes)
                ], axis=0)

                dsc = compute_dice(binary, gt_binary)
                all_dsc.append(dsc)
                logger.info(f"  DSC: {dsc.mean():.4f} (mean over {args.num_classes} classes)")

    # ── Summary ───────────────────────────────────────────
    logger.info(f"\nInference complete. {len(vol_files)} volumes processed.")
    logger.info(f"Predictions saved to: {args.output_dir}")

    if all_dsc:
        import numpy as np
        all_dsc = np.array(all_dsc)
        logger.info(f"\nEvaluation Summary:")
        logger.info(f"  Mean DSC (all classes): {np.nanmean(all_dsc):.4f} ± {np.nanstd(all_dsc):.4f}")
        for c in range(args.num_classes):
            logger.info(f"  Class {c+1:2d}: DSC = {np.nanmean(all_dsc[:, c]):.4f}")


if __name__ == "__main__":
    main()
