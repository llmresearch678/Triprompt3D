"""
structural_prompt.py
Structural Prompt Encoder (Qa) for TRIPROMPT-3D.
Encodes localized 3D anatomical sub-volumes into compact
structural embeddings using a lightweight 3D ResNet-18.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    """Basic 3D residual block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.GroupNorm(8, out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class StructuralPromptEncoder(nn.Module):
    """
    Lightweight 3D ResNet-18 variant that maps a cropped
    anatomical sub-volume J_SUB^(c) to a structural embedding Qa^(c).

    Architecture:
        stem → 4 residual stages → global average pool → projection

    Args:
        in_channels  : input channels (1 for CT)
        embed_dim    : output embedding dimension Ca
        feature_dim  : shared backbone dim C for projection (default 256)
    """

    def __init__(self, in_channels=1, embed_dim=128, feature_dim=256):
        super().__init__()
        self.embed_dim   = embed_dim
        self.feature_dim = feature_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Residual stages
        self.layer1 = self._make_layer(32,  64,  stride=1)
        self.layer2 = self._make_layer(64,  128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, embed_dim, stride=2)

        # Global average pool + projection
        self.gap  = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(embed_dim, feature_dim)

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            ResBlock3D(in_ch, out_ch, stride=stride),
            ResBlock3D(out_ch, out_ch),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_sub):
        """
        Args:
            x_sub: (B, K, 1, Ha, Wa, Da) — K class sub-volumes per batch

        Returns:
            Qa   : (B, K, C)  projected structural prompts
        """
        B, K, C_in, H, W, D = x_sub.shape

        # Flatten batch & class dims for parallel processing
        x = x_sub.view(B * K, C_in, H, W, D)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)           # (B*K, embed_dim)
        x = self.proj(x)                      # (B*K, feature_dim)

        Qa = x.view(B, K, self.feature_dim)   # (B, K, C)
        return Qa
