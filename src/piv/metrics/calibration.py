"""Uncertainty calibration metrics: marginal per-component coverage (68/95 %),
interval width, NLL, reliability bins, and a scalar variance recalibration fitted
on VALIDATION data only. Marginal 95 % per component is not joint 95 % per vector.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def coverage(err: np.ndarray, std: np.ndarray, levels=(0.68, 0.95)) -> dict:
    out = {}
    for lv in levels:
        z = norm.ppf(0.5 + lv / 2)
        out[f"coverage_{int(lv * 100)}"] = float(np.mean(np.abs(err) <= z * std))
        out[f"width_{int(lv * 100)}_mean"] = float(np.mean(2 * z * std))
    out["nll_gauss"] = float(np.mean(0.5 * (err ** 2 / std ** 2 + np.log(2 * np.pi * std ** 2))))
    return out


def uq_metrics(pred, std, gt, mask) -> dict:
    out = {}
    for c, name in enumerate(("dx", "dy")):
        m = mask & np.isfinite(pred[:, c]) & np.isfinite(gt[:, c])
        e = (pred[:, c] - gt[:, c])[m]
        s = std[:, c][m]
        for k, v in coverage(e, s).items():
            out[f"{name}_{k}"] = v
        out[f"{name}_reliability"] = reliability(e, s)
    return out


def reliability(err, std, n_bins: int = 10) -> list[dict]:
    q = np.quantile(std, np.linspace(0, 1, n_bins + 1))
    rows = []
    for a, b in zip(q[:-1], q[1:]):
        sel = (std >= a) & (std <= b)
        if sel.sum() < 10:
            continue
        rows.append({"std_pred_rms": float(np.sqrt(np.mean(std[sel] ** 2))),
                     "err_rms": float(np.sqrt(np.mean(err[sel] ** 2))), "n": int(sel.sum())})
    return rows


def fit_variance_scale(err_val: np.ndarray, std_val: np.ndarray) -> float:
    """MLE of a single factor k with std' = k * std (validation split only)."""
    return float(np.sqrt(np.mean(err_val ** 2 / std_val ** 2)))
