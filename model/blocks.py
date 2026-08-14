"""
model/blocks.py
---------------
Core building blocks for SemiRestoreNet:
  - LayerNorm2d      : Channel-last layer norm for feature maps
  - SimpleGate       : Gated activation (replaces ReLU/GELU)
  - NAFBlock         : Nonlinear Activation Free Block (NAFNet, ECCV 2022)
  - DownsampleBlock  : Strided conv encoder step
  - UpsampleBlock    : Pixel-shuffle decoder step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer Normalization for 2D feature maps  (B, C, H, W)
# ---------------------------------------------------------------------------
class LayerNorm2d(nn.Module):
    """
    Apply LayerNorm over the channel dimension of (B, C, H, W) tensors.
    Equivalent to `nn.LayerNorm(C)` applied per spatial position.
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias   = nn.Parameter(torch.zeros(num_channels))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# ---------------------------------------------------------------------------
# Simple Gate  (replaces ReLU – no trainable params, memory-efficient)
# ---------------------------------------------------------------------------
class SimpleGate(nn.Module):
    """
    Split channels in half and gate them element-wise.
    Input:  (B, 2C, H, W)
    Output: (B,  C, H, W)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# ---------------------------------------------------------------------------
# NAFBlock  (core restoration block)
# ---------------------------------------------------------------------------
class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block from:
        "Simple Baselines for Image Restoration" (NAFNet), ECCV 2022
        https://arxiv.org/abs/2204.04676

    Structure (per channel attention arm):
        LayerNorm → DWConv3×3 → SimpleGate → Conv1×1 → CA → skip
        +
        LayerNorm → FF (Linear→SimpleGate→Linear) → skip
    """
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_ch = c * dw_expand       # channel width inside DW arm
        ffn_ch = c * ffn_expand     # channel width inside FFN arm

        # --- Depth-wise attention arm ---
        self.norm1     = LayerNorm2d(c)
        self.conv1     = nn.Conv2d(c, dw_ch,  1, bias=True)          # expand
        self.conv2     = nn.Conv2d(dw_ch, dw_ch, 3, padding=1,        # DW conv (grouped)
                                   groups=dw_ch, bias=True)
        self.conv3     = nn.Conv2d(dw_ch // 2, c, 1, bias=True)       # contract after gate
        self.gate      = SimpleGate()

        # Simple channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c, c // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // 4, c, bias=False),
            nn.Sigmoid(),
        )

        # --- Feed-forward arm ---
        self.norm2     = LayerNorm2d(c)
        self.ff_conv1  = nn.Conv2d(c, ffn_ch, 1, bias=True)
        self.ff_conv2  = nn.Conv2d(ffn_ch // 2, c, 1, bias=True)
        self.ff_gate   = SimpleGate()

        self.dropout   = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 else nn.Identity()

        # Learnable skip weights (initialised to small values → stable training)
        self.beta  = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1) * 0.01, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Attention arm ---
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.gate(h)                          # (B, dw_ch//2, H, W)
        # channel attention
        ca_w = self.ca(h).view(h.size(0), -1, 1, 1)
        h = h * ca_w
        h = self.conv3(h)
        h = self.dropout(h)
        x = x + h * self.beta

        # --- FFN arm ---
        h = self.norm2(x)
        h = self.ff_conv1(h)
        h = self.ff_gate(h)
        h = self.ff_conv2(h)
        h = self.dropout(h)
        x = x + h * self.gamma
        return x


# ---------------------------------------------------------------------------
# Encoder  (strided convolution + NAFBlock)
# ---------------------------------------------------------------------------
class DownsampleBlock(nn.Module):
    """Downsample by 2 using a strided Conv2d, then refine with NAFBlock."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.naf  = NAFBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.naf(self.down(x))


# ---------------------------------------------------------------------------
# Decoder  (bilinear up + skip concat + NAFBlock)
# ---------------------------------------------------------------------------
class UpsampleBlock(nn.Module):
    """Upsample by 2 (bilinear), concatenate skip, project, refine."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False)
        self.naf  = NAFBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.naf(self.proj(x))


# ---------------------------------------------------------------------------
# Pixel-Shuffle Super-Resolution Head  (×2)
# ---------------------------------------------------------------------------
class PixelShuffleSRHead(nn.Module):
    """
    Efficient sub-pixel convolution upsampler (×scale).
    Uses ICNR initialisation to reduce checkerboard artifacts.
    """
    def __init__(self, in_ch: int, out_ch: int = 1, scale: int = 2):
        super().__init__()
        self.scale = scale
        mid_ch = in_ch * 2  # intermediate channels before pixel-shuffle
        self.conv1  = nn.Conv2d(in_ch, mid_ch,  3, padding=1, bias=True)
        self.gate   = SimpleGate()                # mid_ch → mid_ch//2
        self.conv2  = nn.Conv2d(mid_ch // 2, out_ch * scale * scale, 3, padding=1, bias=True)
        self.ps     = nn.PixelShuffle(scale)
        # Final refinement
        self.refine = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True)

        # ICNR init for pixel shuffle, zero init for final refine layer to match bicubic residual at step 0
        self._icnr_init()
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def _icnr_init(self):
        """ICNR initialization for PixelShuffle conv."""
        with torch.no_grad():
            ni, nf = self.conv2.weight.shape[:2]
            k = self.conv2.weight.shape[2]
            sub_k = k
            g = int(ni / (self.scale ** 2))
            sub = torch.zeros(g, nf, sub_k, sub_k)
            nn.init.kaiming_normal_(sub)
            kernel = sub.repeat_interleave(self.scale ** 2, dim=0)
            self.conv2.weight.data.copy_(kernel[:ni])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.gate(x)    # SimpleGate: (B, mid_ch) → (B, mid_ch//2)
        x = self.conv2(x)
        x = self.ps(x)      # PixelShuffle ×scale
        x = self.refine(x)
        return x
