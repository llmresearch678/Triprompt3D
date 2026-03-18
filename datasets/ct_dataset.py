"""
datasets/ct_dataset.py
Unified 3D CT dataset loader for TRIPROMPT-3D.

Handles:
  - NIfTI loading (images + masks)
  - Resampling to 1.5×1.5×1.5 mm³ isotropic
  - HU clipping [-1000, 1000] + normalization [0, 1]
  - Per-class sub-volume extraction (Qa input)
  - Cross-subject mask sampling (Qd input — different subject s ≠ i)
  - Training augmentation
  - Reproducible random seeding
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAffined,
    ToTensord,
    Resized,
)


class CTDataset(Dataset):
    """
    Unified 3D CT dataset for multi-organ segmentation.

    Directory structure expected:
        data/
          train/
            images/  case_0001.nii.gz  ...
            masks/   case_0001.nii.gz  ...
          test/
            images/
            masks/

    Args:
        data_dir      : root data directory
        split         : "train" or "test"
        num_classes   : K — number of segmentation classes
        img_size      : (H, W, D) after resampling crop
        sub_vol_size  : (Ha, Wa, Da) for structural sub-volumes
        mask_size     : (H, W, D) for cross-subject masks
        augment       : apply training augmentation (train only)
        seed          : random seed for reproducibility
        class_labels  : list of integer label values (length K)
    """

    VOXEL_SPACING = (1.5, 1.5, 1.5)   # mm³ isotropic
    HU_MIN        = -1000
    HU_MAX        =  1000

    def __init__(
        self,
        data_dir,
        split="train",
        num_classes=13,
        img_size=(96, 96, 96),
        sub_vol_size=(48, 48, 48),
        mask_size=(96, 96, 96),
        augment=True,
        seed=42,
        class_labels=None,
    ):
        super().__init__()
        self.data_dir    = Path(data_dir)
        self.split       = split
        self.num_classes = num_classes
        self.img_size    = img_size
        self.sub_vol_size = sub_vol_size
        self.mask_size   = mask_size
        self.augment     = augment and (split == "train")
        self.seed        = seed

        # Default class labels: 1-indexed, background = 0
        self.class_labels = class_labels or list(range(1, num_classes + 1))
        assert len(self.class_labels) == num_classes

        # Discover cases
        img_dir  = self.data_dir / split / "images"
        mask_dir = self.data_dir / split / "masks"
        self.cases = sorted([
            (str(img_dir / f), str(mask_dir / f))
            for f in os.listdir(img_dir)
            if f.endswith(".nii.gz") or f.endswith(".nii")
        ])
        assert len(self.cases) > 0, f"No cases found in {img_dir}"

        self._build_transforms()

    def _build_transforms(self):
        """Build MONAI transform pipeline."""
        base = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(
                keys=["image", "label"],
                pixdim=self.VOXEL_SPACING,
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=self.HU_MIN, a_max=self.HU_MAX,
                b_min=0.0, b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(
                keys=["image", "label"],
                spatial_size=self.img_size,
                mode=("trilinear", "nearest"),
            ),
        ]

        aug = []
        if self.augment:
            aug = [
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
                RandAffined(
                    keys=["image", "label"],
                    prob=0.3,
                    rotate_range=(0.26, 0.26, 0.26),   # ±15°
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    padding_mode="border",
                ),
                RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
                RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
            ]

        final = [ToTensord(keys=["image", "label"])]
        self.transform = Compose(base + aug + final)

    def _extract_sub_volume(self, image, label, class_label):
        """
        Extract a tight bounding-box crop around a specific class label.
        Returns a sub-volume of shape sub_vol_size.

        If the class is absent, returns a zero volume.
        """
        mask = (label == class_label)
        if mask.sum() == 0:
            return torch.zeros(1, *self.sub_vol_size)

        # Find bounding box
        coords = torch.nonzero(mask.squeeze(), as_tuple=False)
        lo     = coords.min(0).values
        hi     = coords.max(0).values + 1

        # Expand bbox by 20% on each side
        size   = hi - lo
        pad    = (size * 0.2).long().clamp(min=1)
        lo     = (lo - pad).clamp(min=0)
        hi     = torch.minimum(hi + pad, torch.tensor(image.shape[1:]))

        sub = image[
            :,
            lo[0]:hi[0],
            lo[1]:hi[1],
            lo[2]:hi[2],
        ]

        # Resize to standard sub_vol_size
        sub = torch.nn.functional.interpolate(
            sub.unsqueeze(0).float(),
            size=self.sub_vol_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return sub

    def _sample_cross_subject_mask(self, class_label, current_idx):
        """
        Sample a binary mask from a DIFFERENT subject (s ≠ i).
        This is the PDP's population prior input — prevents data leakage.
        """
        other_indices = list(range(len(self.cases)))
        other_indices.remove(current_idx)
        other_idx = random.choice(other_indices)

        _, mask_path = self.cases[other_idx]
        mask_nib = nib.load(mask_path)
        mask_np  = mask_nib.get_fdata().astype(np.float32)

        binary_mask = (mask_np == class_label).astype(np.float32)

        # Resize to mask_size
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=self.mask_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        return mask_tensor.clamp(0, 1)   # (1, H, W, D)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        img_path, mask_path = self.cases[idx]

        # Load and transform
        data = self.transform({"image": img_path, "label": mask_path})
        image = data["image"].float()    # (1, H, W, D)
        label = data["label"].long()     # (1, H, W, D)

        # ── Per-class sub-volumes for structural prompt (Qa) ────
        sub_volumes = []
        for cl in self.class_labels:
            sub = self._extract_sub_volume(image, label, cl)
            sub_volumes.append(sub)
        sub_volumes = torch.stack(sub_volumes, dim=0)   # (K, 1, Ha, Wa, Da)

        # ── Cross-subject masks for deformation prompt (Qd) ─────
        # Sample from a different subject to prevent data leakage
        cross_masks = []
        for cl in self.class_labels:
            cm = self._sample_cross_subject_mask(cl, idx)
            cross_masks.append(cm)
        cross_masks = torch.stack(cross_masks, dim=0)   # (K, 1, H, W, D)

        # ── Multi-label GT: one binary channel per class ─────────
        gt_multilabel = torch.stack([
            (label == cl).float().squeeze(0)
            for cl in self.class_labels
        ], dim=0)                                        # (K, H, W, D)

        # ── 3D occupancy volume for PDP decoder training ─────────
        gt_3d = torch.stack([
            torch.nn.functional.interpolate(
                (label == cl).float().unsqueeze(0),
                size=self.mask_size,
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)
            for cl in self.class_labels
        ], dim=0)                                        # (K, 1, H, W, D)

        return {
            "image":        image,          # (1, H, W, D)
            "label":        label,          # (1, H, W, D) raw
            "gt_multilabel": gt_multilabel, # (K, H, W, D) binary per class
            "sub_volumes":  sub_volumes,    # (K, 1, Ha, Wa, Da)
            "cross_masks":  cross_masks,    # (K, 1, H, W, D)
            "gt_3d":        gt_3d,          # (K, 1, H, W, D)
            "case_id":      Path(img_path).stem,
        }


def build_dataloader(
    data_dir,
    split,
    batch_size=2,
    num_workers=4,
    num_classes=13,
    img_size=(96, 96, 96),
    seed=42,
    **kwargs,
):
    """Convenience function to build DataLoader."""
    from torch.utils.data import DataLoader

    dataset = CTDataset(
        data_dir=data_dir,
        split=split,
        num_classes=num_classes,
        img_size=img_size,
        augment=(split == "train"),
        seed=seed,
        **kwargs,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(num_workers > 0),
    )

    print(f"[Dataset] {split}: {len(dataset)} cases, {len(loader)} batches")
    return loader
