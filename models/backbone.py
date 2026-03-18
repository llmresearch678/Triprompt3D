"""
backbone.py
Swin-UNETR backbone for TRIPROMPT-3D.
Wraps MONAI's SwinUNETR and exposes multi-scale features.
"""

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRBackbone(nn.Module):
    """
    Hierarchical volumetric encoder based on Swin-UNETR.
    Returns multi-scale feature maps {f_l} and a dense
    embedding map Z used for voxel-level segmentation.

    Args:
        img_size   : (H, W, D) input volume size
        in_channels: number of input channels (1 for CT)
        feature_dim: shared channel dimension C (default 256)
        pretrained : path to pre-trained weights or None
    """

    def __init__(
        self,
        img_size=(96, 96, 96),
        in_channels=1,
        feature_dim=256,
        pretrained=None,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Core Swin-UNETR (encoder + decoder)
        self.swin_unetr = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_dim,
            feature_size=48,
            use_checkpoint=True,
        )

        # Project decoder output to feature_dim if needed
        self.proj = nn.Conv3d(feature_dim, feature_dim, kernel_size=1)

        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _load_pretrained(self, path):
        state = torch.load(path, map_location="cpu")
        # Handle various checkpoint formats
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.swin_unetr.load_state_dict(state, strict=False)
        print(f"[Backbone] Loaded pretrained weights. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W, D) input CT volume

        Returns:
            Z      : (B, C, H, W, D)  dense voxel-level embedding
            hidden : list of intermediate feature maps at 4 scales
        """
        # Extract hidden states from Swin encoder
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)

        # Decoder forward to get dense embedding
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states[0])
        enc2 = self.swin_unetr.encoder3(hidden_states[1])
        enc3 = self.swin_unetr.encoder4(hidden_states[2])
        enc4 = self.swin_unetr.encoder10(hidden_states[4])

        dec4 = self.swin_unetr.decoder5(enc4, hidden_states[3])
        dec3 = self.swin_unetr.decoder4(dec4, enc3)
        dec2 = self.swin_unetr.decoder3(dec3, enc2)
        dec1 = self.swin_unetr.decoder2(dec2, enc1)
        Z    = self.swin_unetr.decoder1(dec1, enc0)
        Z    = self.proj(Z)

        # Multi-scale features for prompt encoders
        multi_scale = [enc1, enc2, enc3, enc4]

        return Z, multi_scale
