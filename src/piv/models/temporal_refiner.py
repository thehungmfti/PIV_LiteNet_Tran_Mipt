"""Temporal residual refiner (M1) and a parameter-matched spatial-only ablation.

Input for centre field k (offline, centred, lookahead = 2 samples):
  d0[k-2..k+2]   5 backbone displacement FIELDS on the same Eulerian grid  [B,5,2,H,W]
  images a_k,b_k of the centre pair                                         [B,2,H,W]
  confidence proxy: photometric residual |a_k - warp(b_k, d0_k)|             [B,1,H,W]
  context mask (1 = real field, 0 = missing at sequence edge)               [B,5]
Output: d_hat_k = d0_k + r_phi(context). The last layer is zero-initialised so
the refiner starts as the identity. No smoothing is imposed on the output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def backwarp(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Sample img at x + flow (pixel units), zero outside."""
    n, _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    gx = 2 * (xs + flow[:, 0]) / max(w - 1, 1) - 1
    gy = 2 * (ys + flow[:, 1]) / max(h - 1, 1) - 1
    return F.grid_sample(img, torch.stack([gx, gy], -1), align_corners=True, padding_mode="zeros")


def photometric_residual(a: torch.Tensor, b: torch.Tensor, flow: torch.Tensor, scale: float = 255.0):
    return (a - backwarp(b, flow)).abs() / scale


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, k=3, d=1):
        super().__init__(nn.Conv2d(cin, cout, k, padding=d * (k // 2), dilation=d), nn.LeakyReLU(0.1))


class TemporalRefiner(nn.Module):
    """Per-field shared 2-D encoder -> temporal mixing (Conv3d over the 5 fields) -> residual head."""

    def __init__(self, context: int = 5, width: int = 32, disp_scale: float = 4.0):
        super().__init__()
        assert context % 2 == 1
        self.context = context
        self.disp_scale = disp_scale            # displacement normalisation (pixel)
        self.field_enc = nn.Sequential(ConvBlock(3, width), ConvBlock(width, width))   # (dx,dy,mask)
        self.img_enc = nn.Sequential(ConvBlock(3, width), ConvBlock(width, width))     # (a,b,residual)
        self.temporal = nn.Sequential(
            nn.Conv3d(width, width, (3, 3, 3), padding=(1, 1, 1)), nn.LeakyReLU(0.1),
            nn.Conv3d(width, width, (context, 1, 1)), nn.LeakyReLU(0.1))
        self.fuse = nn.Sequential(ConvBlock(2 * width, width), ConvBlock(width, width, d=2),
                                  ConvBlock(width, width, d=4), ConvBlock(width, width))
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, fields, img_a, img_b, ctx_mask):
        b, t, _, h, w = fields.shape
        c = t // 2
        d0 = fields[:, c]
        m = ctx_mask.view(b, t, 1, 1, 1).expand(-1, -1, 1, h, w)
        f = torch.cat([fields / self.disp_scale * m, m], 2).reshape(b * t, 3, h, w)
        f = self.field_enc(f).view(b, t, -1, h, w).permute(0, 2, 1, 3, 4)     # [B,C,T,H,W]
        f = self.temporal(f).squeeze(2)                                         # [B,C,H,W]
        res = photometric_residual(img_a, img_b, d0)
        g = self.img_enc(torch.cat([img_a / 255.0, img_b / 255.0, res], 1))
        return self.fuse(torch.cat([f, g], 1)), d0

    def forward(self, fields, img_a, img_b, ctx_mask):
        feat, d0 = self.features(fields, img_a, img_b, ctx_mask)
        return d0 + self.head(feat) * self.disp_scale


class SpatialRefiner(nn.Module):
    """Ablation: only the centre field (no temporal context), parameter budget matched
    to TemporalRefiner by widening the spatial path."""

    def __init__(self, context: int = 5, width: int = 32, disp_scale: float = 4.0, extra_width: int | None = None):
        super().__init__()
        self.context = context
        self.disp_scale = disp_scale
        ew = extra_width or width
        self.field_enc = nn.Sequential(ConvBlock(3, width), ConvBlock(width, width),
                                       ConvBlock(width, ew), ConvBlock(ew, width))
        self.img_enc = nn.Sequential(ConvBlock(3, width), ConvBlock(width, width))
        self.fuse = nn.Sequential(ConvBlock(2 * width, width), ConvBlock(width, width, d=2),
                                  ConvBlock(width, width, d=4), ConvBlock(width, width))
        self.head = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def features(self, fields, img_a, img_b, ctx_mask):
        c = fields.shape[1] // 2
        d0 = fields[:, c]
        ones = torch.ones_like(d0[:, :1])
        f = self.field_enc(torch.cat([d0 / self.disp_scale, ones], 1))
        res = photometric_residual(img_a, img_b, d0)
        g = self.img_enc(torch.cat([img_a / 255.0, img_b / 255.0, res], 1))
        return self.fuse(torch.cat([f, g], 1)), d0

    def forward(self, fields, img_a, img_b, ctx_mask):
        feat, d0 = self.features(fields, img_a, img_b, ctx_mask)
        return d0 + self.head(feat) * self.disp_scale


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def matched_spatial(temporal: TemporalRefiner, width: int = 32) -> SpatialRefiner:
    """Choose extra_width so that the spatial ablation has >= the temporal model's parameters."""
    target = n_params(temporal)
    best = None
    for ew in range(8, 256, 2):
        s = SpatialRefiner(temporal.context, width, temporal.disp_scale, ew)
        if n_params(s) >= target:
            best = s
            break
    return best
