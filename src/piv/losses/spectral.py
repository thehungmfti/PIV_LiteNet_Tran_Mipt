"""Spectral loss: log-PSD of probe fluctuations, prediction vs GT, identical processing.

Same detrend (mean removal), window (Hann), normalisation (one-sided density) and
frequency band for both signals. Segment length N gives Delta f = fs / N.
"""
from __future__ import annotations

import math

import torch


def psd_torch(x: torch.Tensor, fs: float, window: str = "hann", detrend: str = "mean"):
    """x [..., N] real -> (freqs [N//2+1], one-sided PSD [..., N//2+1]) in unit^2/Hz.

    Normalisation matches scipy.signal.periodogram(scaling='density').
    """
    n = x.shape[-1]
    if detrend == "mean":
        x = x - x.mean(-1, keepdim=True)
    elif detrend not in (None, "none"):
        raise ValueError(detrend)
    if window == "hann":
        w = torch.hann_window(n, periodic=True, dtype=x.dtype, device=x.device)
    elif window in ("boxcar", None, "none"):
        w = torch.ones(n, dtype=x.dtype, device=x.device)
    else:
        raise ValueError(window)
    X = torch.fft.rfft(x * w, dim=-1)
    p = (X.real ** 2 + X.imag ** 2) / (fs * (w ** 2).sum())
    if n % 2 == 0:
        p[..., 1:-1] *= 2
    else:
        p[..., 1:] *= 2
    f = torch.fft.rfftfreq(n, 1.0 / fs).to(x.dtype)
    return f, p


def spectral_loss(pred_probe: torch.Tensor, gt_probe: torch.Tensor, fs: float, fmin: float, fmax: float,
                  window: str = "hann", floor_rel: float = 1e-6):
    """pred/gt [B, P, C, N] probe signals (pixel/pair). Loss = mean |log PSD_pred - log PSD_gt|
    over the declared band (DC excluded). A relative floor avoids log(0)."""
    f, pp = psd_torch(pred_probe, fs, window)
    _, pg = psd_torch(gt_probe, fs, window)
    band = (f >= max(fmin, f[1].item())) & (f <= fmax)
    if not band.any():
        raise ValueError("empty spectral band")
    floor = floor_rel * pg[..., band].mean(-1, keepdim=True).clamp_min(1e-12)
    return (torch.log(pp[..., band] + floor) - torch.log(pg[..., band] + floor)).abs().mean()
