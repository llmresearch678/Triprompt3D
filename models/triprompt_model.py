"""
triprompt_model.py
Full TRIPROMPT-3D model assembly.

Integrates:
  - SwinUNETRBackbone
  - StructuralPromptEncoder  (Qa)
  - TextPromptEncoder        (Qt)
  - PopulationDeformationPrompt (Qd / PDP)
  - TriQueryIntegrator
  - PromptContextAligner
  - Voxel-wise multi-label segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone            import SwinUNETRBackbone
from .structural_prompt   import StructuralPromptEncoder
from .text_prompt         import TextPromptEncoder, DEFAULT_ORGAN_DESCRIPTIONS
from .deformation_prompt  import PopulationDeformationPrompt
from .triprompt_aligner   import TriQueryIntegrator, PromptContextAligner


class TriPrompt3D(nn.Module):
    """
    TRIPROMPT-3D: Deformation-Aware Multimodal Prompting Framework.

    Args:
        num_classes    : K — number of segmentation classes
        img_size       : (H, W, D) input volume size
        feature_dim    : C — shared embedding dimension (256)
        latent_dim     : B — deformation latent dimension (128)
        tau_init       : initial Gumbel-Softmax temperature
        kl_weight      : weight for KL loss term
        class_descs    : dict {class_name: text_description}
        pretrained_bb  : path to pretrained Swin-UNETR weights
    """

    def __init__(
        self,
        num_classes=13,
        img_size=(96, 96, 96),
        feature_dim=256,
        latent_dim=128,
        tau_init=1.0,
        kl_weight=1.0,
        class_descs=None,
        pretrained_bb=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.latent_dim  = latent_dim

        # ── Backbone ────────────────────────────────────────────
        self.backbone = SwinUNETRBackbone(
            img_size=img_size,
            in_channels=1,
            feature_dim=feature_dim,
            pretrained=pretrained_bb,
        )

        # ── Prompt Encoders ─────────────────────────────────────
        self.struct_enc = StructuralPromptEncoder(
            in_channels=1,
            embed_dim=latent_dim,
            feature_dim=feature_dim,
        )

        self.text_enc = TextPromptEncoder(
            feature_dim=feature_dim,
            freeze_bert=True,
        )

        self.pdp = PopulationDeformationPrompt(
            backbone_dim=feature_dim,
            text_dim=feature_dim,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
        )

        # Project deformation latent → C for alignment
        self.deform_proj = nn.Linear(latent_dim, feature_dim)

        # ── Learnable Segmentation Queries ──────────────────────
        # Qs ∈ R^{K×C}: one query per class
        self.seg_queries = nn.Parameter(
            torch.randn(1, num_classes, feature_dim) * 0.02
        )

        # ── Aligner ─────────────────────────────────────────────
        self.tq_integrator = TriQueryIntegrator(
            dim=feature_dim,
            num_heads=8,
            tau_init=tau_init,
        )

        self.context_aligner = PromptContextAligner(
            dim=feature_dim,
            num_heads=8,
        )

        # ── Segmentation Head ────────────────────────────────────
        # Dot-product classifier: M_c(x,y,z) = σ(<U_s^(c), φ_{x,y,z}>)
        # No additional parameters — handled in forward()

        # ── Class descriptions ───────────────────────────────────
        self.class_descs = class_descs or DEFAULT_ORGAN_DESCRIPTIONS

    def set_tau(self, tau):
        """Anneal Gumbel-Softmax temperature."""
        self.tq_integrator.set_tau(tau)

    def _extract_cls_features(self, Z, num_classes):
        """
        Extract per-class features from dense map Z by global pooling.
        Returns (B, K, C) class feature vectors.
        """
        B, C, H, W, D = Z.shape
        # Simple global average pool — can be replaced with RoI pooling
        cls_feat = Z.flatten(2).mean(-1)                      # (B, C)
        cls_feat = cls_feat.unsqueeze(1).expand(-1, num_classes, -1)  # (B, K, C)
        return cls_feat

    def forward(
        self,
        volume,
        sub_volumes,
        cross_subject_masks,
        gt_volume=None,
        class_descs=None,
        training=True,
    ):
        """
        Args:
            volume              : (B, 1, H, W, D)      — input CT volume
            sub_volumes         : (B, K, 1, Ha, Wa, Da) — class sub-volumes
            cross_subject_masks : (B, K, 1, H, W, D)   — masks from other subjects
            gt_volume           : (B, K, 1, H, W, D)   — 3D GT (train only)
            class_descs         : dict or None (use defaults)
            training            : bool

        Returns (train):
            seg_logits : (B, K, H, W, D)   — per-class segmentation logits
            losses     : dict with keys [loss_seg, loss_ce, loss_align1,
                                         loss_align2, loss_3d, loss_kl]

        Returns (inference):
            seg_logits : (B, K, H, W, D)
            alpha      : (B, K)  reliability scores
        """
        B = volume.shape[0]
        K = self.num_classes
        device = volume.device

        # ── 1. Backbone ─────────────────────────────────────────
        Z, multi_scale = self.backbone(volume)       # Z: (B,C,H,W,D)

        # ── 2. Structural Prompt Qa ──────────────────────────────
        Qa = self.struct_enc(sub_volumes)            # (B, K, C)

        # ── 3. Text Prompt Qt ────────────────────────────────────
        desc = class_descs or self.class_descs
        Qt   = self.text_enc(desc, device=device)    # (1, K, C)
        Qt   = Qt.expand(B, -1, -1)                  # (B, K, C)

        # ── 4. Deformation Prompt Qd (PDP) ──────────────────────
        cls_feat = self._extract_cls_features(Z, K)  # (B, K, C)
        qt_mean  = Qt.mean(dim=1, keepdim=True).expand(-1, K, -1)

        pdp_out  = self.pdp(
            cross_subject_masks, cls_feat, qt_mean,
            gt_volume=gt_volume, training=training,
        )
        q_d_tilde, alpha, loss_3d, loss_kl = pdp_out

        # Project deformation latent B → C
        Qd = self.deform_proj(q_d_tilde)             # (B, K, C)

        # ── 5. Segmentation Queries ──────────────────────────────
        Qs = self.seg_queries.expand(B, -1, -1)      # (B, K, C)

        # ── 6. TriQuery Integrator ───────────────────────────────
        Qs_hat, Qt_hat, Qa_hat, Qd_hat = self.tq_integrator(
            Qs, Qt, Qa, Qd, multi_scale
        )

        # ── 7. Prompt–Query Context Aligner ─────────────────────
        Os = self.context_aligner(Qs_hat, Qt_hat, Qa_hat, Qd_hat)
        # Os: (B, K, C)  — refined segmentation queries

        # ── 8. Voxel-wise Segmentation ───────────────────────────
        # M_c(x,y,z) = σ(<U_s^(c), φ_{x,y,z}>)
        phi = Z.flatten(2).transpose(1, 2)           # (B, H*W*D, C)
        seg_logits = torch.bmm(phi, Os.transpose(1, 2))   # (B, H*W*D, K)
        seg_logits = seg_logits.transpose(1, 2)           # (B, K, H*W*D)

        H, W, D = Z.shape[-3:]
        seg_logits = seg_logits.view(B, K, H, W, D)       # (B, K, H, W, D)

        if not training:
            return seg_logits, alpha

        # ── 9. Compute Losses ────────────────────────────────────
        losses = {}

        # Will be filled by train.py with actual GT masks
        losses["loss_3d"] = loss_3d if loss_3d is not None else torch.tensor(0.0, device=device)
        losses["loss_kl"] = loss_kl if loss_kl is not None else torch.tensor(0.0, device=device)

        # Contrastive alignment losses (computed in train.py with GT)
        losses["Qs_norm"]  = F.normalize(Os, dim=-1)         # for L_ALIGN(1)
        losses["Qa_proj"]  = F.normalize(Qa_hat, dim=-1)     # for L_ALIGN(2)
        losses["Qt_proj"]  = F.normalize(Qt_hat, dim=-1)     # for L_ALIGN(2)
        losses["Qd_proj"]  = F.normalize(Qd_hat, dim=-1)     # for L_ALIGN(1)

        return seg_logits, losses
