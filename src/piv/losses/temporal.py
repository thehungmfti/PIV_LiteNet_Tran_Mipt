"""Supervised temporal-derivative loss (Eulerian, not material acceleration).

||(v_hat[k+1]-v_hat[k]) - (v_gt[k+1]-v_gt[k])|| / dt_sample with v = d / dt_pair,
so variable dt_pair is handled by converting to velocity before differencing.
Never ||d[k+1]-d[k]||^2 alone (that would push towards a frozen field).
"""
from __future__ import annotations

import torch

from piv.losses.supervised import masked_mean


def temporal_derivative_loss(pred_seq, gt_seq, mask_seq, dt_pair, dt_sample: float, eps: float = 0.01):
    """pred_seq/gt_seq [B,T,2,H,W], mask_seq [B,T,H,W], dt_pair [B,T] (s)."""
    gt_seq = torch.nan_to_num(gt_seq)
    dtp = dt_pair.view(*dt_pair.shape, 1, 1, 1)
    vp = pred_seq / dtp
    vg = gt_seq / dtp
    # Eulerian derivative dv/dt (pixel/s^2), multiplied by the constant dt_pair_mean*dt_sample
    # only for numerical conditioning (result is in pixel/pair change per sample)
    norm = dt_pair.mean() * dt_sample
    dvp = torch.diff(vp, dim=1) / dt_sample * norm
    dvg = torch.diff(vg, dim=1) / dt_sample * norm
    m = torch.minimum(mask_seq[:, 1:].float(), mask_seq[:, :-1].float())
    e = torch.sqrt(((dvp - dvg) ** 2).sum(2) + eps ** 2) - eps
    return masked_mean(e, m > 0)
