"""Uncertainty head (M3): per-pixel, per-component log-variance of the residual.

This is a predictive variance under the chosen Gaussian-diagonal model/loss; it
does not separate epistemic and aleatoric uncertainty and is not a covariance.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from piv.models.temporal_refiner import ConvBlock


class UncertaintyHead(nn.Module):
    def __init__(self, width: int = 32, s_min: float = -9.0, s_max: float = 5.0):
        super().__init__()
        self.s_min, self.s_max = s_min, s_max
        self.net = nn.Sequential(ConvBlock(width + 2, width), ConvBlock(width, width), nn.Conv2d(width, 2, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, feat: torch.Tensor, d_hat: torch.Tensor, disp_scale: float = 4.0) -> torch.Tensor:
        s = self.net(torch.cat([feat, d_hat / disp_scale], 1))
        return torch.clamp(s, self.s_min, self.s_max)       # log variance (pixel^2)


class RefinerWithUQ(nn.Module):
    def __init__(self, refiner: nn.Module, width: int = 32):
        super().__init__()
        self.refiner = refiner
        self.uq = UncertaintyHead(width)

    def forward(self, fields, img_a, img_b, ctx_mask):
        feat, d0 = self.refiner.features(fields, img_a, img_b, ctx_mask)
        d_hat = d0 + self.refiner.head(feat) * self.refiner.disp_scale
        return d_hat, self.uq(feat, d_hat, self.refiner.disp_scale)
