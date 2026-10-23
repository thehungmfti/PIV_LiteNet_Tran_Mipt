"""Masked supervised losses. All reduce by the number of valid elements; an
all-invalid batch returns an exact zero that is still connected to the graph."""
from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x [B,C,H,W] or [B,H,W]; mask broadcastable bool. NaNs in invalid region are ignored."""
    m = mask.to(x.dtype).expand_as(x)
    x = torch.where(m > 0, x, torch.zeros_like(x))
    n = m.sum()
    return x.sum() / n.clamp_min(1.0)


def charbonnier(pred, gt, mask, eps: float = 0.01):
    """sqrt(|e|^2 + eps^2) per pixel on the 2-vector, eps in pixel."""
    gt = torch.nan_to_num(gt)
    e2 = ((pred - gt) ** 2).sum(1)
    return masked_mean(torch.sqrt(e2 + eps ** 2) - eps, mask)


def gradient_loss(pred, gt, mask, eps: float = 0.01):
    """|grad(pred) - grad(gt)| with forward differences; a difference is used only
    if both of its pixels are valid (no stencil across invalid/boundary pixels)."""
    gt = torch.nan_to_num(gt)
    m = mask.to(pred.dtype)
    loss = 0.0
    for dim in (-1, -2):
        dp = torch.diff(pred, dim=dim)
        dg = torch.diff(gt, dim=dim)
        mm = torch.minimum(m.narrow(dim, 0, m.shape[dim] - 1), m.narrow(dim, 1, m.shape[dim] - 1))
        e = torch.sqrt(((dp - dg) ** 2).sum(1) + eps ** 2) - eps
        loss = loss + masked_mean(e, mm > 0)
    return loss
