"""
deformation_prompt.py
Population-Level Deformation Prompt (PDP / Qd) for TRIPROMPT-3D.

Implements:
  - EDEF: 3D shape deformation encoder (population prior)
  - Gθ  : posterior head (CNN/MLP) producing (μ_d, Σ_d)
  - Reliability score αc = exp(−τ · tr(Σ_d))
  - Adaptive blending: Q̃_d = αc·μ_d + (1−αc)·Q_d,POP
  - 3D volumetric reconstruction decoder pφ(V*|X, Q̃_d)
  - Training objective: 3D reconstruction loss + KL divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Shape Deformation Encoder  E_DEF(·)
# ─────────────────────────────────────────────

class ShapeDeformationEncoder(nn.Module):
    """
    Maps a binary 3D organ shape mask M_c^(s) ∈ {0,1}^{H×W×D}
    to a compact population deformation token Q_d,POP ∈ R^B.

    Architecture: 4× Conv3D (stride-2, GroupNorm, ReLU) + GAP + MLP
    Parameters: ~1.09 M
    """

    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Layer 1: (1, H, W, D) → (32, H/2, W/2, D/2)
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32), nn.ReLU(inplace=True),
            # Layer 2: → (64, H/4, W/4, D/4)
            nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            # Layer 3: → (128, H/8, W/8, D/8)
            nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            # Layer 4: → (128, H/16, W/16, D/16)
            nn.Conv3d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool3d(1)

        self.mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, latent_dim),
        )

    def forward(self, mask):
        """
        Args:
            mask: (B, K, 1, H, W, D)  binary shape masks, K classes

        Returns:
            Q_d_pop: (B, K, latent_dim)  population deformation tokens
        """
        B, K, C, H, W, D = mask.shape
        x = mask.view(B * K, C, H, W, D)

        x = self.encoder(x)
        x = self.gap(x).flatten(1)       # (B*K, 128)
        x = self.mlp(x)                  # (B*K, latent_dim)

        return x.view(B, K, self.latent_dim)


# ─────────────────────────────────────────────
# Posterior Head  Gθ
# ─────────────────────────────────────────────

class PosteriorHead(nn.Module):
    """
    Predicts image-conditioned Gaussian posterior for deformation:
        Eψ(Q_d | X, Fi, P, V*) = N(μ_d, Σ_d)

    Inputs: concatenation of CLS token (C), textual embedding (C),
            and population prior token (B).
    Outputs: μ_d ∈ R^B, L_c ∈ R^{B×B} (Cholesky factor of Σ_d)

    Parameters: ~0.38 M
    """

    def __init__(self, backbone_dim=256, text_dim=256, latent_dim=128, eps=1e-6):
        super().__init__()
        self.latent_dim = latent_dim
        self.eps        = eps

        in_dim = backbone_dim + text_dim + latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 512),    nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 512),    nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, latent_dim + latent_dim * latent_dim),
        )

    def forward(self, cls_token, qt_mean, q_d_pop):
        """
        Args:
            cls_token : (B, K, backbone_dim) — backbone CLS features per class
            qt_mean   : (B, K, text_dim)      — mean textual embedding
            q_d_pop   : (B, K, latent_dim)    — population prior token

        Returns:
            mu    : (B, K, latent_dim)
            sigma : (B, K, latent_dim, latent_dim)  SPD covariance
        """
        B, K, _ = cls_token.shape
        inp = torch.cat([cls_token, qt_mean, q_d_pop], dim=-1)  # (B,K, in_dim)

        out   = self.mlp(inp)                                    # (B,K, B+B²)
        mu    = out[..., :self.latent_dim]                       # (B,K, latent_dim)
        L_vec = out[..., self.latent_dim:]                       # (B,K, B²)

        # Reshape to lower-triangular Cholesky factor
        L = L_vec.view(B, K, self.latent_dim, self.latent_dim)
        L = torch.tril(L)
        # Enforce positive diagonal to guarantee SPD
        diag_idx = torch.arange(self.latent_dim, device=L.device)
        L[..., diag_idx, diag_idx] = F.softplus(L[..., diag_idx, diag_idx]) + self.eps

        sigma = L @ L.transpose(-1, -2)   # (B, K, latent_dim, latent_dim)
        return mu, sigma


# ─────────────────────────────────────────────
# 3D Volumetric Reconstruction Decoder
# ─────────────────────────────────────────────

class VolumetricDecoder(nn.Module):
    """
    pφ(V*_c | X, Q̃_d): maps 2D features + deformation token
    to a 3D occupancy volume.

    Architecture: project & reshape → 4× 3D upsampling blocks
    """

    def __init__(self, backbone_dim=256, latent_dim=128, out_size=(12, 12, 12)):
        super().__init__()
        self.out_size = out_size

        # Fuse backbone features + deformation token
        self.fuse = nn.Linear(backbone_dim + latent_dim, 256)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.GroupNorm(8, 64),  nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(8, 32),  nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, 1),
        )

    def forward(self, cls_feat, q_d_tilde):
        """
        Args:
            cls_feat   : (B, K, backbone_dim)
            q_d_tilde  : (B, K, latent_dim)

        Returns:
            vol: (B, K, 1, H', W', D')  reconstructed 3D occupancy
        """
        B, K, _ = cls_feat.shape
        x = torch.cat([cls_feat, q_d_tilde], dim=-1)  # (B, K, C+B)
        x = F.relu(self.fuse(x))                       # (B, K, 256)

        # Reshape to spatial for convtranspose: treat K as batch
        x = x.view(B * K, 256, 1, 1, 1)
        x = x.expand(-1, -1, *self.out_size)           # (B*K, 256, D0, D0, D0)

        vol = self.decoder(x)                           # (B*K, 1, H', W', D')
        H, W, D = vol.shape[-3:]
        return vol.view(B, K, 1, H, W, D)


# ─────────────────────────────────────────────
# Full PDP Module
# ─────────────────────────────────────────────

class PopulationDeformationPrompt(nn.Module):
    """
    Full Population-Level Deformation Prompt (PDP) module.

    During training:
        1. Encode cross-subject masks → Q_d,POP  (population prior)
        2. Predict posterior (μ_d, Σ_d) via Gθ
        3. Compute reliability score αc = exp(−τ · tr(Σ_d))
        4. Blend: Q̃_d = αc·μ_d + (1−αc)·Q_d,POP
        5. Decode 3D reconstruction volume
        6. Return Q̃_d + training losses

    During inference:
        - No ground-truth masks available for posterior
        - Sample Q_d from learned prior N(Q_d,POP, γ²I)
        - αc computed from sampled covariance
    """

    def __init__(
        self,
        backbone_dim=256,
        text_dim=256,
        latent_dim=128,
        tau=0.07,
        gamma0=1.0,
        kl_weight=1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau        = tau
        self.gamma0     = gamma0
        self.kl_weight  = kl_weight

        self.edef    = ShapeDeformationEncoder(in_channels=1, latent_dim=latent_dim)
        self.g_theta = PosteriorHead(backbone_dim, text_dim, latent_dim)
        self.decoder = VolumetricDecoder(backbone_dim, latent_dim)

    def reliability_score(self, sigma):
        """αc = exp(−τ · tr(Σ_d))  ∈ (0, 1]"""
        # sigma: (B, K, B, B)
        trace = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1)  # (B, K)
        return torch.exp(-self.tau * trace)                       # (B, K)

    def kl_divergence(self, mu, sigma, q_d_pop):
        """
        KL( N(μ_d, Σ_d) || N(Q_d,POP, γ²I) )
        Closed-form Gaussian KL.
        """
        B, K, d = mu.shape
        gamma2  = self.gamma0 ** 2

        # Trace term
        trace_term = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1) / gamma2

        # Mahalanobis term
        diff       = (mu - q_d_pop)                              # (B, K, d)
        maha       = (diff ** 2).sum(-1) / gamma2                # (B, K)

        # Log-det term
        sign, logdet = torch.linalg.slogdet(sigma)
        logdet_term  = logdet - d * torch.log(torch.tensor(gamma2, device=mu.device))

        kl = 0.5 * (trace_term + maha - d - logdet_term)        # (B, K)
        return kl.mean()

    def reparameterize(self, mu, sigma):
        """Reparameterization trick: z = μ + L·ε, ε ~ N(0,I)"""
        L   = torch.linalg.cholesky(sigma)
        eps = torch.randn_like(mu)
        return mu + (L @ eps.unsqueeze(-1)).squeeze(-1)

    def forward(self, cross_subject_masks, cls_feat, qt_mean,
                gt_volume=None, training=True):
        """
        Args:
            cross_subject_masks: (B, K, 1, H, W, D) — masks from other subjects
            cls_feat           : (B, K, C) — backbone features per class
            qt_mean            : (B, K, C) — text prompt embeddings
            gt_volume          : (B, K, 1, H, W, D) — 3D GT for training loss
            training           : bool

        Returns (train):
            q_d_tilde : (B, K, latent_dim) — final deformation prompt
            alpha     : (B, K)             — reliability scores
            loss_3d   : scalar             — 3D reconstruction loss
            loss_kl   : scalar             — KL divergence loss

        Returns (inference):
            q_d_tilde : (B, K, latent_dim)
            alpha     : (B, K)
            None, None
        """
        # Step 1: Population prior from cross-subject masks
        q_d_pop = self.edef(cross_subject_masks)   # (B, K, latent_dim)

        if training and gt_volume is not None:
            # Step 2: Image-conditioned posterior
            mu, sigma = self.g_theta(cls_feat, qt_mean, q_d_pop)

            # Step 3: Reliability score from posterior covariance
            alpha = self.reliability_score(sigma)   # (B, K)

            # Step 4: Sample deformation from posterior (reparameterization)
            q_d_sampled = self.reparameterize(mu, sigma)

            # Step 5: Adaptive blending
            a = alpha.unsqueeze(-1)                 # (B, K, 1)
            q_d_tilde = a * q_d_sampled + (1 - a) * q_d_pop

            # Step 6: 3D reconstruction loss
            vol_pred  = self.decoder(cls_feat, q_d_tilde)
            loss_3d   = F.binary_cross_entropy_with_logits(
                vol_pred.squeeze(2),
                F.interpolate(
                    gt_volume.squeeze(2).float(),
                    size=vol_pred.shape[-3:],
                    mode="trilinear",
                    align_corners=False,
                ),
            )

            # Step 7: KL divergence loss
            loss_kl = self.kl_divergence(mu, sigma, q_d_pop)

            return q_d_tilde, alpha, loss_3d, loss_kl

        else:
            # Inference: sample from prior N(Q_d,POP, γ²I)
            eps       = torch.randn_like(q_d_pop) * self.gamma0
            q_d_sample = q_d_pop + eps

            # Approximate reliability from prior variance
            B, K, d  = q_d_pop.shape
            sigma_approx = (self.gamma0 ** 2) * torch.eye(
                d, device=q_d_pop.device
            ).unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)

            alpha     = self.reliability_score(sigma_approx)
            a         = alpha.unsqueeze(-1)
            q_d_tilde = a * q_d_sample + (1 - a) * q_d_pop

            return q_d_tilde, alpha, None, None
