"""Gaussian diagonal NLL: 0.5 [exp(-s) (d_hat - d_gt)^2 + s], constant dropped."""
from __future__ import annotations

import torch

from piv.losses.supervised import masked_mean


def gaussian_nll(pred, logvar, gt, mask, s_min: float = -9.0, s_max: float = 5.0):
    gt = torch.nan_to_num(gt)
    s = torch.clamp(logvar, s_min, s_max)
    nll = 0.5 * (torch.exp(-s) * (pred - gt) ** 2 + s)          # [B,2,H,W]
    return masked_mean(nll, mask.unsqueeze(1))
