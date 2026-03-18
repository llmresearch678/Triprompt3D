"""
losses/contrastive_alignment.py
Contrastive alignment losses for TRIPROMPT-3D.

Implements:
  L_ALIGN(1): Segmentation–Prompt Alignment
      Aligns segmentation queries U_s^(c) with fused prompt p_c = Qa+Qt+Qd

  L_ALIGN(2): Prompt–Prompt Alignment
      Aligns anatomical (Qa) and textual (Qt) prompts for the same class

Both use InfoNCE (NT-Xent) contrastive objective with temperature τ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegPromptAlignmentLoss(nn.Module):
    """
    L_ALIGN(1): Segmentation–Prompt Alignment (Eq. 15 in paper)

    For each class c, the normalized segmentation query Û_s^(c)
    should align with the normalized fused prompt p̂_c = (Qa+Qt+Qd)/‖·‖

    L = -1/K · Σ_c log [ exp(Û_s^(c)·p̂_c / τ) / Σ_j exp(Û_s^(c)·p̂_j / τ) ]

    Args:
        temperature: τ (default 0.07 as in paper)
        eps        : numerical stability constant ξ
    """

    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, Qs_norm, Qa_proj, Qt_proj, Qd_proj):
        """
        Args:
            Qs_norm : (B, K, C) — L2-normalized segmentation queries
            Qa_proj : (B, K, C) — L2-normalized structural prompts
            Qt_proj : (B, K, C) — L2-normalized text prompts
            Qd_proj : (B, K, C) — L2-normalized deformation prompts

        Returns:
            loss: scalar
        """
        B, K, C = Qs_norm.shape

        # Fused prompt: p_c = Qa + Qt + Qd, then normalize
        p = Qa_proj + Qt_proj + Qd_proj                      # (B, K, C)
        p_norm = F.normalize(p, dim=-1, eps=self.eps)        # (B, K, C)

        # Compute similarity matrix (B, K, K)
        sim = torch.bmm(Qs_norm, p_norm.transpose(1, 2)) / self.tau  # (B, K, K)

        # InfoNCE: diagonal = positive pairs
        labels = torch.arange(K, device=sim.device).unsqueeze(0).expand(B, -1)
        loss   = F.cross_entropy(sim.view(B * K, K), labels.view(B * K))

        return loss


class PromptPromptAlignmentLoss(nn.Module):
    """
    L_ALIGN(2): Prompt–Prompt Alignment (Eq. 16 in paper)

    Encourages anatomical (Qa) and textual (Qt) prompts for the same class
    to occupy nearby locations in the shared embedding space.

    L = -1/K · Σ_c log [ exp(Q̂a^(c)·Q̂t^(c) / τ) / Σ_j exp(Q̂a^(c)·Q̂t^(j) / τ) ]

    Args:
        temperature: τ
        eps        : numerical stability
    """

    def __init__(self, temperature=0.07, eps=1e-8):
        super().__init__()
        self.tau = temperature
        self.eps = eps

    def forward(self, Qa_norm, Qt_norm):
        """
        Args:
            Qa_norm: (B, K, C) — L2-normalized structural prompts
            Qt_norm: (B, K, C) — L2-normalized text prompts

        Returns:
            loss: scalar
        """
        B, K, C = Qa_norm.shape

        # Similarity: Q̂a^(c) · Q̂t^(j)
        sim = torch.bmm(Qa_norm, Qt_norm.transpose(1, 2)) / self.tau  # (B, K, K)

        labels = torch.arange(K, device=sim.device).unsqueeze(0).expand(B, -1)
        loss   = F.cross_entropy(sim.view(B * K, K), labels.view(B * K))

        return loss


class GradNormBalancer:
    """
    Adaptive loss weight balancing via gradient-norm matching.

    λ_i ← λ_i · ‖∇_θ L_SEG‖ / ‖∇_θ L_i‖

    Updates λ1, λ2 every epoch to keep alignment loss gradients
    commensurate with the segmentation gradient.

    Args:
        lambda1_init: initial weight for L_ALIGN(1)
        lambda2_init: initial weight for L_ALIGN(2)
        clip_val    : gradient norm clip value (10.0)
    """

    def __init__(self, lambda1_init=1.0, lambda2_init=1.0, clip_val=10.0):
        self.lambda1  = lambda1_init
        self.lambda2  = lambda2_init
        self.clip_val = clip_val

    def update(self, loss_seg, loss_align1, loss_align2, shared_params):
        """
        Compute gradient norms and update weights.

        Args:
            loss_seg    : segmentation loss (scalar)
            loss_align1 : alignment loss 1 (scalar)
            loss_align2 : alignment loss 2 (scalar)
            shared_params: list of model parameters (decoder + query modules)
        """
        def grad_norm(loss, params):
            grads = torch.autograd.grad(
                loss, params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            total = sum(
                g.norm().item() ** 2
                for g in grads if g is not None
            )
            return total ** 0.5 + 1e-8

        norm_seg = grad_norm(loss_seg,    shared_params)
        norm_a1  = grad_norm(loss_align1, shared_params)
        norm_a2  = grad_norm(loss_align2, shared_params)

        self.lambda1 = min(self.lambda1 * (norm_seg / norm_a1), self.clip_val)
        self.lambda2 = min(self.lambda2 * (norm_seg / norm_a2), self.clip_val)

    @property
    def weights(self):
        return self.lambda1, self.lambda2
