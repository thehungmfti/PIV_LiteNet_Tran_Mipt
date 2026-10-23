"""Vector-error metrics on valid pixels (all in pixel/pair unless calibrated).

EPE = sqrt(ex^2 + ey^2); RMSE per component; bias = mean(pred - gt) per component;
normalised EPE = mean EPE / mean |gt| over the same valid pixels (declared denominator);
outlier fraction = share of valid pixels with EPE > threshold (declared, pixel).
"""
from __future__ import annotations

import numpy as np

OUTLIER_THRESHOLDS = (0.5, 1.0)


def _valid(pred, gt, mask):
    m = np.isfinite(pred).all(-3) & np.isfinite(gt).all(-3)
    if mask is not None:
        m &= mask.astype(bool)
    return m


def vector_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None,
                   thresholds=OUTLIER_THRESHOLDS) -> dict:
    """pred/gt [..., 2, H, W]; mask [..., H, W]."""
    m = _valid(pred, gt, mask)
    ex = (pred[..., 0, :, :] - gt[..., 0, :, :])[m]
    ey = (pred[..., 1, :, :] - gt[..., 1, :, :])[m]
    if ex.size == 0:
        return {"n_valid": 0}
    epe = np.sqrt(ex ** 2 + ey ** 2)
    gmag = np.sqrt(gt[..., 0, :, :][m] ** 2 + gt[..., 1, :, :][m] ** 2)
    out = {
        "n_valid": int(ex.size),
        "epe_mean": float(epe.mean()),
        "epe_median": float(np.median(epe)),
        "epe_p95": float(np.percentile(epe, 95)),
        "rmse_vec": float(np.sqrt((epe ** 2).mean())),
        "rmse_dx": float(np.sqrt((ex ** 2).mean())),
        "rmse_dy": float(np.sqrt((ey ** 2).mean())),
        "bias_dx": float(ex.mean()),
        "bias_dy": float(ey.mean()),
        "std_dx": float(ex.std()),
        "std_dy": float(ey.std()),
        "norm_epe": float(epe.mean() / max(gmag.mean(), 1e-12)),
    }
    for t in thresholds:
        out[f"outlier_frac_{t:g}px"] = float((epe > t).mean())
    return out


def epe_map(pred, gt):
    return np.sqrt(((pred - gt) ** 2).sum(-3))


def per_frame_epe(pred, gt, mask=None) -> np.ndarray:
    """[N,2,H,W] -> [N] mean EPE per pair (for block bootstrap)."""
    e = epe_map(pred, gt)
    m = _valid(pred, gt, mask)
    return np.array([e[k][m[k]].mean() if m[k].any() else np.nan for k in range(len(e))])
