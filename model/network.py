"""
model/network.py
----------------
SemiRestoreNet — Joint Noise Removal + 2× Super-Resolution Network.

Architecture:
  - Encoder: 3-level strided-conv encoder (64→128→256→512 channels)
  - Bottleneck: 6× NAFBlock at lowest resolution
  - Decoder: 3-level bilinear-up + skip-concat decoder
  - SR Head: PixelShuffle ×2 to go from 128×128 → 256×256
  - Global residual: bicubic upsampled input added to final output
    (guarantees structural safety even at epoch 0)

Input:  (B, 1, H,   W  )  e.g. (B, 1, 128, 128) NoisyLR
Output: (B, 1, 2H, 2W  )  e.g. (B, 1, 256, 256) Restored
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import NAFBlock, DownsampleBlock, UpsampleBlock, PixelShuffleSRHead, LayerNorm2d


class SemiRestoreNet(nn.Module):
    """
    SemiRestoreNet:  128×128 NoisyLR  →  256×256 Clean HR

    Args:
        in_ch        : Input channels (1 for grayscale)
        out_ch       : Output channels (1 for grayscale)
        base_ch      : Base channel width at first encoder level (default 64)
        bottleneck_blocks: Number of NAFBlocks in the bottleneck (default 6)
        scale        : Super-resolution scale factor (default 2)
    """
    def __init__(
        self,
        in_ch: int  = 1,
        out_ch: int = 1,
        base_ch: int = 64,
        bottleneck_blocks: int = 6,
        scale: int = 2,
    ):
        super().__init__()
        c = base_ch  # shorthand

        # ----------------------------------------------------------------
        # Input projection  →  (B, c, H, W)
        # ----------------------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1, bias=True),
            NAFBlock(c),
        )

        # ----------------------------------------------------------------
        # Encoder  (each level halves spatial dims, doubles channels)
        # ----------------------------------------------------------------
        self.enc1 = DownsampleBlock(c,       c * 2)   # 128 → 64,  ch: 64 → 128
        self.enc2 = DownsampleBlock(c * 2,   c * 4)   #  64 → 32,  ch: 128 → 256
        self.enc3 = DownsampleBlock(c * 4,   c * 8)   #  32 → 16,  ch: 256 → 512

        # ----------------------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------------------
        self.bottleneck = nn.Sequential(
            *[NAFBlock(c * 8) for _ in range(bottleneck_blocks)]
        )

        # ----------------------------------------------------------------
        # Decoder  (each level doubles spatial dims, halves channels)
        # in_ch  = bottleneck out_ch + skip_ch from same encoder level
        # ----------------------------------------------------------------
        self.dec3 = UpsampleBlock(c * 8, c * 4, c * 4)   # 16 → 32
        self.dec2 = UpsampleBlock(c * 4, c * 2, c * 2)   # 32 → 64
        self.dec1 = UpsampleBlock(c * 2, c,     c)        # 64 → 128

        # Extra refinement after decoder
        self.refine = nn.Sequential(
            NAFBlock(c),
            NAFBlock(c),
        )

        # ----------------------------------------------------------------
        # Super-Resolution head  (128→256)
        # ----------------------------------------------------------------
        self.sr_head = PixelShuffleSRHead(in_ch=c, out_ch=out_ch, scale=scale)

        # ----------------------------------------------------------------
        # Global residual: bicubic-upsampled input is added to output
        # ----------------------------------------------------------------
        self._scale = scale

        # Weight init
        self._init_weights()

    def _init_weights(self):
        """Kaiming init for Conv2d, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W)  NoisyLR input, float32, normalised to [0,1]
        Returns:
            out: (B, 1, H*scale, W*scale) restored image
        """
        # Global residual branch (bicubic up of input)
        residual = F.interpolate(
            x,
            scale_factor=self._scale,
            mode='bicubic',
            align_corners=False,
        )

        # Input projection
        x0 = self.input_proj(x)      # (B, c, H, W)

        # Encode
        x1 = self.enc1(x0)           # (B, 2c, H/2, W/2)
        x2 = self.enc2(x1)           # (B, 4c, H/4, W/4)
        x3 = self.enc3(x2)           # (B, 8c, H/8, W/8)

        # Bottleneck
        b  = self.bottleneck(x3)     # (B, 8c, H/8, W/8)

        # Decode with skip connections
        d3 = self.dec3(b,  x2)       # (B, 4c, H/4, W/4)
        d2 = self.dec2(d3, x1)       # (B, 2c, H/2, W/2)
        d1 = self.dec1(d2, x0)       # (B,  c,    H,   W)

        # Refine
        d1 = self.refine(d1)

        # SR head  →  2× spatial
        out = self.sr_head(d1)       # (B, 1, 2H, 2W)

        # Add global residual
        out = out + residual

        return out


# ---------------------------------------------------------------------------
# Convenience function: count parameters
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = SemiRestoreNet(in_ch=1, out_ch=1, base_ch=64, bottleneck_blocks=6)
    print(f"SemiRestoreNet  params: {count_params(net) / 1e6:.2f}M")

    dummy = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        out = net(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")    # Expected: (2, 1, 256, 256)
    assert out.shape == (2, 1, 256, 256), "Shape mismatch!"
    print("✓ Forward pass OK")
