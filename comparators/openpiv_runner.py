"""OpenPIV-Python (GPLv3, used as an external tool, not redistributed) multi-pass window-deformation PIV.

Uses openpiv.windef first_pass + multipass_img_deform WITHOUT transform_coordinates,
so x, y are 0-based image coordinates of the vector positions, u > 0 right, v > 0 down.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

from piv.io.hdf5 import read_dataset, write_prediction

DEFAULT = dict(windowsizes=(32, 16, 16), overlap=(24, 8, 8), num_iterations=3, subpixel_method="gaussian",
               deformation_method="symmetric", interpolation_order=3, correlation_method="circular",
               median_threshold=3, std_threshold=10, sig2noise_threshold=1.0, sig2noise_validate=True,
               replace_vectors=True, smoothn=False)


def piv_pair(a: np.ndarray, b: np.ndarray, settings):
    from openpiv import filters, validation, windef
    x, y, u, v, s2n = windef.first_pass(a, b, settings)
    gm = np.zeros_like(u, bool)
    u, v = np.ma.array(u, mask=gm), np.ma.array(v, mask=gm)
    flags = validation.typical_validation(u, v, s2n, settings)
    u, v = filters.replace_outliers(u, v, flags)
    for i in range(1, settings.num_iterations):
        x, y, u, v, gm, flags = windef.multipass_img_deform(a, b, i, x, y, u, v, settings)
    u = np.ma.filled(np.ma.fix_invalid(u, fill_value=np.nan), np.nan)
    v = np.ma.filled(np.ma.fix_invalid(v, fill_value=np.nan), np.nan)
    return x, y, np.asarray(u), np.asarray(v)


def run(dataset: str, output: str, settings: dict | None = None) -> dict:
    import contextlib
    import io

    import openpiv
    from openpiv import windef
    s = {**DEFAULT, **(settings or {})}
    st = windef.PIVSettings()
    for k, v in s.items():
        setattr(st, k, v)
    st.show_plot = False
    d = read_dataset(dataset)
    U, V = [], []
    t0 = time.perf_counter()
    for k in range(d.n_pairs):
        a, b = d.pair(k)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            x, y, u, v = piv_pair(a.astype(np.float64), b.astype(np.float64), st)
        U.append(u); V.append(v)
    secs = time.perf_counter() - t0
    disp = np.stack([np.stack(U), np.stack(V)], 1).astype(np.float32)
    prov = {"model_id": "C1_OpenPIV_windef", "openpiv_version": openpiv.__version__, "settings": s,
            "seconds_total": secs, "seconds_per_pair": secs / d.n_pairs, "dataset": str(dataset),
            "units": "pixel/pair", "grid_convention": "openpiv windef x,y (0-based, before transform_coordinates)",
            "demo_only": False}
    write_prediction(output, disp, d.t_a, d.t_b, valid=np.all(np.isfinite(disp), 1),
                     grid_x=x[0, :].astype(float), grid_y=y[:, 0].astype(float), provenance=prov)
    return prov
