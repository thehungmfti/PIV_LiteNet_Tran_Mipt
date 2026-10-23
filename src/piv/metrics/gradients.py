"""Velocity-gradient and vorticity errors on the valid interior.

Central differences (np.gradient, 2nd order) in image coordinates. Vorticity in
the physical frame (y up) is omega = dv/dx - du/dy = -(d dy_img/dx + d dx/dy_img)
for displacement components (dx, dy_img); the sign convention is applied here
explicitly. The mask is eroded by one pixel so no stencil touches invalid data.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def gradients(d: np.ndarray):
    """d [...,2,H,W] -> dict of du/dx, du/dy_img, dv/dx, dv/dy_img (image-frame components)."""
    ux = np.gradient(d[..., 0, :, :], axis=-1)
    uy = np.gradient(d[..., 0, :, :], axis=-2)
    vx = np.gradient(d[..., 1, :, :], axis=-1)
    vy = np.gradient(d[..., 1, :, :], axis=-2)
    return {"dudx": ux, "dudy": uy, "dvdx": vx, "dvdy": vy}


def vorticity_phys(d: np.ndarray) -> np.ndarray:
    g = gradients(d)
    # u_phys = dx, v_phys = -dy_img, y_phys = -y_img  => dv/dx = -dvdx_img, du/dy = -dudy_img
    return -g["dvdx"] + g["dudy"]


def interior(mask: np.ndarray, it: int = 1) -> np.ndarray:
    st = np.zeros((3,) * mask.ndim, bool)
    st[(1,) * (mask.ndim - 2) + (slice(None), slice(None))] = True
    m = binary_erosion(mask, structure=st, iterations=it, border_value=0)
    return m


def gradient_metrics(pred, gt, mask=None) -> dict:
    if mask is None:
        mask = np.ones(pred.shape[:-3] + pred.shape[-2:], bool)
    m = interior(mask & np.isfinite(pred).all(-3) & np.isfinite(gt).all(-3))
    gp, gg = gradients(np.nan_to_num(pred)), gradients(np.nan_to_num(gt))
    out = {}
    for k in gp:
        e = (gp[k] - gg[k])[m]
        out[f"rmse_{k}"] = float(np.sqrt((e ** 2).mean()))
    wp, wg = vorticity_phys(np.nan_to_num(pred))[m], vorticity_phys(np.nan_to_num(gt))[m]
    out["rmse_vorticity"] = float(np.sqrt(((wp - wg) ** 2).mean()))
    out["rms_vorticity_gt"] = float(np.sqrt((wg ** 2).mean()))
    out["div_rms_pred"] = float(np.sqrt(((gp["dudx"] + gp["dvdy"])[m] ** 2).mean()))
    return out
