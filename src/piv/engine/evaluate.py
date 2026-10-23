"""Evaluation of a prediction file against a reference dataset.

Dense predictions are scored on all valid pixels ('full'), on the interior
(margin declared) and on full-context pairs only. Vector-grid predictions
(cross-correlation, e.g. PIVlab/OpenPIV) are scored at their own grid points
against the EXACT Lagrangian GT at those points (RK4 from the stored flow
definition), and dense methods can be sampled at the same points for a fair
like-for-like comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from piv.data import pairing
from piv.data.synthetic import converged_advect, make_flow
from piv.io.hdf5 import read_dataset, read_prediction
from piv.metrics import calibration as calm
from piv.metrics import gradients as gradm
from piv.metrics import spectra as specm
from piv.metrics import velocity as velm


def flow_of(d):
    if "flow_json" not in d.extra:
        return None, None
    spec = json.loads(bytes(d.extra["flow_json"]).decode())
    return make_flow(spec), spec


def reference_frequency(spec: dict | None):
    if spec is None:
        return None
    if spec["type"] == "vortex_street":
        return float(spec["f_shed"])
    if spec["type"] in ("cellular", "orbiting_vortices"):
        return float(spec["freqs"][0])
    return None


def gt_on_points(flow, xs: np.ndarray, ys: np.ndarray, t_a, t_b) -> np.ndarray:
    """Exact forward displacement at arbitrary points (0-based pixel coords) -> [N,2,ny,nx]."""
    X, Y = np.meshgrid(xs, ys)
    out = []
    for a, b in zip(t_a, t_b):
        x1, y1, _, _ = converged_advect(flow, X, Y, a, b)
        out.append(np.stack([x1 - X, y1 - Y]))
    return np.stack(out).astype(np.float32)


def sample_dense(disp: np.ndarray, xs, ys) -> np.ndarray:
    """Bilinear sampling of dense [N,2,H,W] at grid points -> [N,2,ny,nx]."""
    X, Y = np.meshgrid(xs, ys)
    out = np.empty((disp.shape[0], 2) + X.shape, np.float32)
    for n in range(disp.shape[0]):
        for c in range(2):
            out[n, c] = map_coordinates(disp[n, c], [Y, X], order=1, mode="nearest")
    return out


def default_probes(h, w, step=16, offset=8):
    return [(y, x) for y in range(offset, h, step) for x in range(offset, w, step)]


def evaluate_dense(pred: dict, d, ecfg: dict) -> dict:
    disp = pred["displacement"]
    gt = d.gt_displacement
    if gt is None:
        raise ValueError("reference has no /gt/displacement")
    valid = d.valid if d.valid is not None else np.ones(gt.shape[:1] + gt.shape[2:], bool)
    valid = valid & pred["valid"].astype(bool)
    m = int(ecfg.get("interior_margin", 8))
    inner = np.zeros_like(valid)
    inner[:, m:-m, m:-m] = True
    full_ctx = np.array(pred["provenance"].get("full_context_pairs", [1] * len(disp)), bool)
    res = {"full": velm.vector_metrics(disp, gt, valid),
           "interior": velm.vector_metrics(disp, gt, valid & inner),
           "interior_full_context": velm.vector_metrics(disp[full_ctx], gt[full_ctx], (valid & inner)[full_ctx]),
           "edge_context_only": (velm.vector_metrics(disp[~full_ctx], gt[~full_ctx], (valid & inner)[~full_ctx])
                                 if (~full_ctx).any() else None),
           "gradients_interior": gradm.gradient_metrics(disp[full_ctx], gt[full_ctx], (valid & inner)[full_ctx])}
    res["per_pair_epe_interior"] = velm.per_frame_epe(disp, gt, valid & inner).tolist()
    if "std" in pred:
        res["uq"] = calm.uq_metrics(disp[full_ctx], pred["std"][full_ctx], gt[full_ctx], (valid & inner)[full_ctx])
    # temporal spectra at probe pixels (full-context range, uniform sampling required)
    try:
        fs = 1.0 / pairing.dt_sample(d.t_a, d.t_b)
        sc = ecfg.get("spectral", {})
        idx = np.where(full_ctx)[0]
        probes = default_probes(*gt.shape[-2:], int(sc.get("probe_step", 16)), int(sc.get("probe_offset", 8)))
        _, spec = flow_of(d)
        res["spectral"] = specm.spectral_metrics(disp[idx], gt[idx], fs, probes, tuple(sc.get("band", (15, 300))),
                                                 reference_frequency(spec), int(sc.get("nperseg", 64)))
    except ValueError as e:
        res["spectral"] = {"skipped": str(e)}
    return res


def evaluate_grid(disp_grid: np.ndarray, xs, ys, d, ecfg: dict, gt_grid: np.ndarray | None = None) -> dict:
    flow, spec = flow_of(d)
    if gt_grid is None:
        gt_grid = gt_on_points(flow, xs, ys, d.t_a, d.t_b)
    res = {"grid": velm.vector_metrics(disp_grid, gt_grid),
           "grid_shape": [len(ys), len(xs)],
           "per_pair_epe_grid": velm.per_frame_epe(disp_grid, gt_grid).tolist()}
    res["gradients_grid"] = grid_gradient_metrics(disp_grid, gt_grid, xs, ys)
    try:
        fs = 1.0 / pairing.dt_sample(d.t_a, d.t_b)
        sc = ecfg.get("spectral", {})
        k = int(ecfg.get("context_trim", 2))
        sl = slice(k, len(disp_grid) - k)
        ny, nx = disp_grid.shape[-2:]
        probes = [(j, i) for j in range(1, ny - 1, 2) for i in range(1, nx - 1, 2)]
        res["spectral_grid"] = specm.spectral_metrics(disp_grid[sl], gt_grid[sl], fs, probes,
                                                      tuple(sc.get("band", (15, 300))), reference_frequency(spec),
                                                      int(sc.get("nperseg", 64)))
    except ValueError as e:
        res["spectral_grid"] = {"skipped": str(e)}
    return res, gt_grid


def grid_gradient_metrics(pred, gt, xs, ys) -> dict:
    """Vorticity error on the vector grid (spacing taken from the grid), interior points."""
    dx = float(np.mean(np.diff(xs)))
    dy = float(np.mean(np.diff(ys)))
    def vort(f):
        dvdx = np.gradient(f[:, 1], dx, axis=-1)
        dudy = np.gradient(f[:, 0], dy, axis=-2)
        return (-dvdx + dudy)[:, 1:-1, 1:-1]
    wp, wg = vort(pred), vort(gt)
    return {"rmse_vorticity": float(np.sqrt(np.nanmean((wp - wg) ** 2))),
            "rms_vorticity_gt": float(np.sqrt(np.mean(wg ** 2)))}


def evaluate(prediction: str, reference: str, ecfg: dict | None = None) -> dict:
    ecfg = ecfg or {}
    pred = read_prediction(prediction)
    d = read_dataset(reference)
    out = {"prediction": str(prediction), "reference": str(reference),
           "run_id": d.attrs.get("run_id"), "noise": d.attrs.get("noise"), "split": d.attrs.get("split"),
           "flow_type": d.attrs.get("flow_type"), "demo_only": bool(pred["provenance"].get("demo_only", False)),
           "model": pred["provenance"].get("model_id") or pred["provenance"].get("backbone", {}).get("model_id")}
    if "grid_x" in pred:
        res, _ = evaluate_grid(pred["displacement"], pred["grid_x"], pred["grid_y"], d, ecfg)
        out.update(res)
    else:
        out.update(evaluate_dense(pred, d, ecfg))
    return out
