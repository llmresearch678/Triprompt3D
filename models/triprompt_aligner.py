"""
triprompt_aligner.py
Query-Centric TriPrompt Aligner for TRIPROMPT-3D.

Implements:
  - Query-Feature Affinity with scaled dot-product
  - Gumbel-Softmax hard spatial assignment (Lemma 2)
  - TQSOFT  : soft cross-attention for [Qs, Qt]
  - TQHARD  : hard spatial attention for [Qa, Qd]
  - PromptContextAligner: Os = Qs + SoftAttn(Qt) + MaskedAttn(Qa) + MaskedAttn(Qd)
  - Temperature annealing schedule
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Gumbel-Softmax Hard Spatial Assignment
# ─────────────────────────────────────────────

def gumbel_softmax_hard(logits, tau=1.0, dim=-1):
    """
    Differentiable hard assignment via Gumbel-Softmax + straight-through.

    Forward : one-hot argmax  (hard, discrete)
    Backward: soft Gumbel-Softmax gradients

    Args:
        logits: (..., V)  un-normalized scores
        tau   : temperature
        dim   : softmax dimension

    Returns:
        S_hat : (..., V)  hard routing mask with soft gradients
        S_soft: (..., V)  soft Gumbel probabilities
    """
    # Sample Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y      = (logits + gumbel) / tau
    S_soft = F.softmax(y, dim=dim)

    # Hard one-hot selection (straight-through)
    idx    = S_soft.argmax(dim=dim, keepdim=True)
    S_hard = torch.zeros_like(S_soft).scatter_(dim, idx, 1.0)
    S_hat  = S_hard - S_soft.detach() + S_soft   # straight-through

    return S_hat, S_soft


# ─────────────────────────────────────────────
# Soft Cross-Attention (for Qs, Qt)
# ─────────────────────────────────────────────

class SoftCrossAttention(nn.Module):
    """Standard multi-head cross-attention."""

    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, query, key_value):
        """
        Args:
            query    : (B, N, C)
            key_value: (B, M, C)

        Returns:
            out: (B, N, C)
        """
        B, N, C = query.shape
        _, M, _ = key_value.shape
        H       = self.num_heads

        Q = self.q_proj(query).view(B, N, H, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, M, H, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, M, H, self.head_dim).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = self.drop(F.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, C)
        return self.out(out)


# ─────────────────────────────────────────────
# Hard Masked Attention (for Qa, Qd)
# ─────────────────────────────────────────────

class HardMaskedAttention(nn.Module):
    """
    Hard spatial attention gated by Gumbel-Softmax routing mask.
    Enforces sparsity: each query selects exactly one spatial token.
    This bounds prompt interference per Lemma 2.
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out    = nn.Linear(dim, dim)

    def forward(self, query, features, routing_mask):
        """
        Args:
            query       : (B, K, C)  — class query tokens
            features    : (B, V, C)  — flattened spatial feature tokens
            routing_mask: (B, K, V)  — hard assignment (one-hot per row)

        Returns:
            out: (B, K, C)
        """
        B, K, C = query.shape
        _, V, _ = features.shape
        H       = self.num_heads

        Q = self.q_proj(query).view(B, K, H, self.head_dim).transpose(1, 2)
        Kp = self.k_proj(features).view(B, V, H, self.head_dim).transpose(1, 2)
        Vp = self.v_proj(features).view(B, V, H, self.head_dim).transpose(1, 2)

        # Apply routing mask: mask out non-selected tokens
        # routing_mask: (B, K, V) → (B, 1, K, V) for broadcasting over heads
        mask = routing_mask.unsqueeze(1)                      # (B, 1, K, V)
        attn = (Q @ Kp.transpose(-2, -1)) * self.scale       # (B, H, K, V)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)               # handle all-inf rows

        out = (attn @ Vp).transpose(1, 2).contiguous().view(B, K, C)
        return self.out(out)


# ─────────────────────────────────────────────
# TriQuery Integrator (TQ)
# ─────────────────────────────────────────────

class TriQueryIntegrator(nn.Module):
    """
    Multi-scale query refinement:

        [Q̂s, Q̂t] = TQSOFT(Qs, Qt; F)    — soft cross-attn
        Q̂a        = TQHARD(Qa, Ŝ; F)     — hard spatial routing
        Q̂d        = TQHARD(Qd, Ŝ; F)     — hard spatial routing

    All refined at each backbone scale ℓ, then aggregated.
    """

    def __init__(self, dim=256, num_heads=8, tau_init=1.0):
        super().__init__()
        self.dim      = dim
        self.tau      = tau_init

        self.soft_attn = SoftCrossAttention(dim, num_heads)
        self.hard_attn = HardMaskedAttention(dim, num_heads)

        # Per-scale feature projection (4 scales)
        self.scale_projs = nn.ModuleList([
            nn.Conv3d(dim, dim, kernel_size=1) for _ in range(4)
        ])

        # Layer norms
        self.norm_qs = nn.LayerNorm(dim)
        self.norm_qt = nn.LayerNorm(dim)
        self.norm_qa = nn.LayerNorm(dim)
        self.norm_qd = nn.LayerNorm(dim)

    def set_tau(self, tau):
        """Update Gumbel-Softmax temperature (called by training loop)."""
        self.tau = max(tau, 0.07)

    def _compute_affinity(self, Qa, feat_flat, C):
        """
        Scaled dot-product affinity + Gumbel-Softmax hard routing.

        S^(ℓ) = (Qa · X_ℓ^T) / √C
        """
        # feat_flat: (B, V, C), Qa: (B, K, C)
        S = torch.bmm(Qa, feat_flat.transpose(1, 2)) / math.sqrt(C)  # (B, K, V)
        S_hat, _ = gumbel_softmax_hard(S, tau=self.tau, dim=-1)
        return S_hat   # (B, K, V)

    def forward(self, Qs, Qt, Qa, Qd, multi_scale_feats):
        """
        Args:
            Qs   : (B, N, C) segmentation queries
            Qt   : (B, K, C) text prompts
            Qa   : (B, K, C) structural prompts
            Qd   : (B, K, C) deformation prompts
            multi_scale_feats: list of 4 feature tensors (B, C, H_l, W_l, D_l)

        Returns:
            Qs_hat, Qt_hat, Qa_hat, Qd_hat : all (B, ·, C)
        """
        B, K, C = Qa.shape

        Qs_agg = torch.zeros_like(Qs)
        Qt_agg = torch.zeros_like(Qt)
        Qa_agg = torch.zeros_like(Qa)
        Qd_agg = torch.zeros_like(Qd)

        for l, (feat, proj) in enumerate(zip(multi_scale_feats, self.scale_projs)):
            # Project feature to dim C
            feat = proj(feat)                               # (B, C, H, W, D)
            feat_flat = feat.flatten(2).transpose(1, 2)    # (B, V, C)

            # Soft attention for Qs, Qt
            Qs_agg = Qs_agg + self.soft_attn(Qs, feat_flat)
            Qt_agg = Qt_agg + self.soft_attn(Qt, feat_flat)

            # Hard routing mask from Qa
            S_hat  = self._compute_affinity(Qa, feat_flat, C)  # (B, K, V)

            # Hard masked attention for Qa, Qd
            Qa_agg = Qa_agg + self.hard_attn(Qa, feat_flat, S_hat)
            Qd_agg = Qd_agg + self.hard_attn(Qd, feat_flat, S_hat)

        # Normalize by number of scales
        n = len(multi_scale_feats)
        Qs_hat = self.norm_qs(Qs + Qs_agg / n)
        Qt_hat = self.norm_qt(Qt + Qt_agg / n)
        Qa_hat = self.norm_qa(Qa + Qa_agg / n)
        Qd_hat = self.norm_qd(Qd + Qd_agg / n)

        return Qs_hat, Qt_hat, Qa_hat, Qd_hat


# ─────────────────────────────────────────────
# Prompt–Query Context Aligner
# ─────────────────────────────────────────────

class PromptContextAligner(nn.Module):
    """
    Os = Qs + SOFTATTN(Qs, Qt)       — semantic global guidance
            + MASKEDATTN(Qs, Qa, Ŝ) — structural grounding
            + MASKEDATTN(Qs, Qd, Ŝ) — deformation prior

    Separates global semantic attention from spatially grounded
    structural/deformation routing, preventing prompt mixing (Lemma 2).
    """

    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.semantic_attn    = SoftCrossAttention(dim, num_heads)
        self.structural_attn  = SoftCrossAttention(dim, num_heads)
        self.deformation_attn = SoftCrossAttention(dim, num_heads)

        self.norm    = nn.LayerNorm(dim)
        self.ffn     = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, Qs, Qt, Qa, Qd):
        """
        Args:
            Qs: (B, N, C) segmentation queries
            Qt: (B, K, C) text prompts
            Qa: (B, K, C) structural prompts
            Qd: (B, K, C) deformation prompts

        Returns:
            Os: (B, N, C) aligned segmentation queries
        """
        sem  = self.semantic_attn(Qs, Qt)
        struc = self.structural_attn(Qs, Qa)
        deform = self.deformation_attn(Qs, Qd)

        Os = self.norm(Qs + sem + struc + deform)
        Os = self.norm_ffn(Os + self.ffn(Os))
        return Os
