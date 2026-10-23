"""Reliability / parameter analysis of the computation (B0 vs PIVlab vs OpenPIV, + B1 if trained).

Factors (one at a time, others at the reference value):
  displacement magnitude, particle density, particle diameter, read noise, out-of-plane
  dropout proxy, spatial wavelength (response gain), image size not multiple of 32,
  16-bit input path, PIVlab settings (window size / passes / sub-pixel / step),
  vector-position convention (reported vs half-pixel shifted) on shear flow.
All errors are against exact GT (uniform / shear / sine flows). Output:
outputs/reliability/*.csv + *.png + summary.json

python scripts/reliability_analysis.py [--quick] [--with-b1]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zlib
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from comparators import openpiv_runner, pivlab_runner  # noqa: E402
from piv.data import degradation as dg  # noqa: E402
from piv.data.synthetic import (Flow, ParticleConfig, ParticleField, ShearFlow, UniformFlow,  # noqa: E402
                                converged_advect, gt_displacement)
from piv.engine.evaluate import sample_dense  # noqa: E402
from piv.metrics.spatial_response import SineDisplacement, fitted_amplitude  # noqa: E402

OUT = ROOT / "outputs" / "reliability"
PIVLAB_ROOT = str(ROOT / "third_party" / "PIVlab")
REF_P = ParticleConfig()


def render_pairs(flow: Flow, n: int, size: int, pc: ParticleConfig, deg: dg.Degradation, seed: int):
    A, B, G = [], [], None
    for r in range(n):
        rng = np.random.default_rng(seed + r)
        pf = ParticleField(size, size, pc, rng)
        a = pf.render()
        pf.advect(flow, 0.0, 1.0)
        pf.apply_dropout(pc.dropout)
        b = pf.render()
        A.append(dg.apply(a, deg, 0.0, rng)); B.append(dg.apply(b, deg, 1.0, rng))
    return np.stack(A), np.stack(B)


def gt_points(flow, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    x1, y1, _, _ = converged_advect(flow, X, Y, 0.0, 1.0)
    return np.stack([x1 - X, y1 - Y])


class Methods:
    def __init__(self, with_b1: bool):
        from piv.models.baseline_adapter import LiteFlowNetBackend
        d = ROOT / "third_party/PIV-LiteFlowNet-en/caffe/models/PIV-LiteFlowNet-en"
        self.b0 = LiteFlowNetBackend(d / "PIV-LiteFlowNet-en_deploy.prototxt", d / "PIV-LiteFlowNet-en.caffemodel")
        self.b1 = None
        if with_b1 and (ROOT / "outputs/train_B1/B1.pt").is_file():
            import torch
            self.b1 = LiteFlowNetBackend(d / "PIV-LiteFlowNet-en_deploy.prototxt", d / "PIV-LiteFlowNet-en.caffemodel",
                                         model_id="B1")
            self.b1.load_finetuned(torch.load(ROOT / "outputs/train_B1/B1.pt", weights_only=False)["backbone"])

    def run_all(self, A, B, pivlab_settings=None, bits=None):
        out = {}
        out["B0"] = ("dense", self.b0.predict(A, B, bit_depth=bits))
        if self.b1 is not None:
            out["B1"] = ("dense", self.b1.predict(A, B, bit_depth=bits))
        if A.dtype == np.uint8:
            gx, gy, d, _ = pivlab_runner.run_arrays(A, B, PIVLAB_ROOT, pivlab_settings)
            out["PIVlab"] = ("grid", (gx, gy, d))
            out["OpenPIV"] = ("grid", self.openpiv(A, B))
        return out

    @staticmethod
    def openpiv(A, B):
        import contextlib, io, warnings
        from openpiv import windef
        st = windef.PIVSettings()
        for k, v in openpiv_runner.DEFAULT.items():
            setattr(st, k, v)
        res = []
        for a, b in zip(A, B):
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                x, y, u, v = openpiv_runner.piv_pair(a.astype(float), b.astype(float), st)
            res.append(np.stack([u, v]))
        return x[0, :].astype(float), y[:, 0].astype(float), np.stack(res).astype(np.float32)


def errors(results, flow, size, margin=8):
    """EPE at the common grid for every method (dense sampled) + dense interior EPE."""
    xs = np.arange(8, size - 8 + 1, 8, dtype=float)
    ys = xs
    gtg = gt_points(flow, xs, ys)
    gtd, _, _ = gt_displacement(flow, size, size, 0.0, 1.0)
    row = {}
    for m, (kind, val) in results.items():
        if kind == "dense":
            d = val
            e = np.sqrt(((d - gtd[None]) ** 2).sum(1))[:, margin:-margin, margin:-margin]
            row[f"{m}_dense_epe"] = float(e.mean())
            g = sample_dense(d, xs, ys)
        else:
            gx, gy, g = val
            if not (np.allclose(gx, xs) and np.allclose(gy, ys)):
                gtg_m = gt_points(flow, gx, gy)
            else:
                gtg_m = gtg
            e = np.sqrt(((g - gtg_m[None]) ** 2).sum(1))
            row[f"{m}_grid_epe"] = float(np.nanmean(e))
            row[f"{m}_grid_outlier_1px"] = float(np.nanmean(e > 1.0))
            continue
        e = np.sqrt(((g - gtg[None]) ** 2).sum(1))
        row[f"{m}_grid_epe"] = float(e.mean())
        row[f"{m}_grid_outlier_1px"] = float((e > 1.0).mean())
    return row


def sweep(name, values, make, M, n, size, write_rows):
    rows = []
    for v in values:
        flow, pc, deg, extra = make(v)
        A, B = render_pairs(flow, n, size, pc, deg, seed=int(1000 * (zlib.crc32(name.encode()) % 997) + 13))
        res = M.run_all(A, B, **extra)
        row = {"factor": name, "value": v, **errors(res, flow, size)}
        rows.append(row)
        print(name, v, {k: round(x, 4) for k, x in row.items() if k.endswith("grid_epe") or k.endswith("dense_epe")},
              flush=True)
    write_rows(name, rows)
    return rows


COLORS = {"B0": "#2a78d6", "B1": "#eb6834", "PIVlab": "#6250d6", "OpenPIV": "#e34948"}


def plot_sweep(name, rows, xlabel, logx=False):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, ls, mk in (("B0_dense_epe", "-", "o"), ("B0_grid_epe", "--", "o"), ("B1_dense_epe", "-", "s"),
                        ("B1_grid_epe", "--", "s"), ("PIVlab_grid_epe", "-", "^"), ("OpenPIV_grid_epe", "-", "v")):
        if key in rows[0] and rows[0][key] not in (None, ""):
            m = key.split("_")[0]
            ax.plot([float(r["value"]) for r in rows], [float(r[key]) for r in rows], ls=ls, marker=mk, ms=7, lw=2,
                    color=COLORS[m], label=key.replace("_epe", "").replace("_", " "))
    ax.set_xlabel(xlabel); ax.set_ylabel("mean EPE (px)"); ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False); ax.set_title(name.replace("_", " "))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"{name}.png", dpi=130); plt.close(fig)


SWEEP_LABELS = {"displacement_magnitude": ("|d| (px/pair), uniform flow at 30 deg", False),
                "particle_density_ppp": ("particles per pixel", True),
                "particle_diameter_px": ("particle image diameter (px)", False),
                "read_noise_counts": ("Gaussian read noise std (counts, 8-bit)", False),
                "dropout_fraction": ("fraction of particles lost between frames (out-of-plane proxy)", False),
                "low_intensity_peak": ("mean particle peak intensity (counts)", False)}


def replot():
    for name, (lab, logx) in SWEEP_LABELS.items():
        f = OUT / f"{name}.csv"
        if f.is_file():
            plot_sweep(name, list(csv.DictReader(open(f))), lab, logx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-b1", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only redraw figures from the CSV files")
    args = ap.parse_args()
    if args.replot:
        replot()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    n = 2 if args.quick else 6
    size = 128
    M = Methods(args.with_b1)
    noiseless = dg.Degradation()
    ref_deg = dg.PRESETS["low"]
    direction = (math.cos(math.radians(30)), math.sin(math.radians(30)))
    summary = {}

    def write_rows(name, rows):
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("factor", "value"), k))
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
        summary[name] = rows

    uni = lambda m: UniformFlow(u=m * direction[0], v=m * direction[1])
    r = sweep("displacement_magnitude", [0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16],
              lambda m: (uni(m), REF_P, ref_deg, {}), M, n, size, write_rows)
    plot_sweep("displacement_magnitude", r, "|d| (px/pair), uniform flow at 30 deg")
    r = sweep("particle_density_ppp", [0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
              lambda p: (uni(2.5), replace(REF_P, ppp=p), ref_deg, {}), M, n, size, write_rows)
    plot_sweep("particle_density_ppp", r, "particles per pixel", logx=True)
    r = sweep("particle_diameter_px", [1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
              lambda dd: (uni(2.5), replace(REF_P, d_mean=dd, d_std=0.1 * dd, d_min=0.5), ref_deg, {}), M, n, size,
              write_rows)
    plot_sweep("particle_diameter_px", r, "particle image diameter (px)")
    r = sweep("read_noise_counts", [0, 2, 5, 10, 20, 30],
              lambda s: (uni(2.5), REF_P, dg.Degradation(read_noise=s, poisson_gain=2.0 if s else 0.0), {}),
              M, n, size, write_rows)
    plot_sweep("read_noise_counts", r, "Gaussian read noise std (counts, 8-bit)")
    r = sweep("dropout_fraction", [0.0, 0.05, 0.1, 0.2, 0.3],
              lambda f: (uni(2.5), replace(REF_P, dropout=f), ref_deg, {}), M, n, size, write_rows)
    plot_sweep("dropout_fraction", r, "fraction of particles lost between frames (out-of-plane proxy)")
    r = sweep("low_intensity_peak", [20, 40, 80, 150, 220],
              lambda pk: (uni(2.5), replace(REF_P, peak_mean=pk, peak_std=0.25 * pk), ref_deg, {}), M, n, size,
              write_rows)
    plot_sweep("low_intensity_peak", r, "mean particle peak intensity (counts)")

    # PIVlab settings check on a moderately sheared field (gradient matters for window size)
    shear = ShearFlow(gamma=0.04, y0=64)
    rows = []
    for label, st in [("64-32-16 step8 3pass gauss3", {}),
                      ("64-32 step8 2pass", dict(passes=2, ia2=32, step=8)),
                      ("64-32-16-16 step8 4pass", dict(passes=4, ia4=16)),
                      ("64-32-16 2Dgauss", dict(subpix=2)),
                      ("64-32-16 spline deform", dict(imdeform="*spline")),
                      ("64-32-16 no validation", dict(validated=False))]:
        A, B = render_pairs(shear, n, size, REF_P, ref_deg, 77)
        try:
            gx, gy, d, secs = pivlab_runner.run_arrays(A, B, PIVLAB_ROOT, st)
        except RuntimeError as e:   # e.g. '*spline' deformation: Octave interp2 limitation, not PIVlab's
            rows.append({"factor": "pivlab_settings", "value": label, "status": "NOT RUN",
                         "reason": str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else str(e)})
            print(rows[-1], flush=True)
            continue
        g = gt_points(shear, gx, gy)
        g_half = gt_points(shear, gx - 0.5, gy - 0.5)
        e = np.sqrt(((d - g[None]) ** 2).sum(1))
        eh = np.sqrt(((d - g_half[None]) ** 2).sum(1))
        rows.append({"factor": "pivlab_settings", "value": label, "grid_epe": float(np.nanmean(e)),
                     "grid_epe_if_positions_shifted_-0.5px": float(np.nanmean(eh)),
                     "bias_dx": float(np.nanmean(d[:, 0] - g[None, 0])), "n_vectors": int(gx.size * gy.size),
                     "sec_per_pair": secs / n})
        print(rows[-1], flush=True)
    write_rows("pivlab_settings_shear", rows)

    # spatial response (gain vs wavelength), 256 px images
    rows = []
    for lam in [8, 12, 16, 24, 32, 48, 64, 128]:
        A, B = [], []
        for rr in range(max(2, n // 2)):
            rng = np.random.default_rng(500 + rr + lam)
            pf = ParticleField(256, 256, REF_P, rng)
            A.append(np.clip(np.round(pf.render()), 0, 255).astype(np.uint8))
            pf.advect(SineDisplacement(1.0, lam), 0.0, 1.0)
            B.append(np.clip(np.round(pf.render()), 0, 255).astype(np.uint8))
        A, B = np.stack(A), np.stack(B)
        res = M.run_all(A, B)
        row = {"factor": "spatial_response", "value": lam}
        for m, (kind, val) in res.items():
            if kind == "dense":
                gains = [fitted_amplitude(d[1], lam, 16) for d in val]
            else:
                gx, gy, d = val
                if lam <= 2 * float(np.mean(np.diff(gx))):   # at/below the grid Nyquist: aliased, undefined
                    row[f"{m}_gain"] = None
                    continue
                gains = []
                for dd in d:
                    prof = np.nanmean(dd[1, 2:-2, :], 0)
                    S = np.stack([np.sin(2 * np.pi * gx / lam), np.cos(2 * np.pi * gx / lam), np.ones_like(gx)], 1)
                    c, *_ = np.linalg.lstsq(S[2:-2], prof[2:-2], rcond=None)
                    gains.append(math.hypot(c[0], c[1]))
            row[f"{m}_gain"] = float(np.mean(gains))
        rows.append(row)
        print(row, flush=True)
    write_rows("spatial_response", rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in [k for k in rows[0] if k.endswith("_gain")]:
        pts = [(r["value"], r[k]) for r in rows if r[k] is not None]
        ax.semilogx([p[0] for p in pts], [p[1] for p in pts], "-o", label=k[:-5])
    ax.axhline(1, color="k", lw=0.5); ax.set_xlabel("wavelength (px)"); ax.set_ylabel("amplitude gain")
    ax.set_title("sinusoidal response (amplitude 1 px)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "spatial_response.png", dpi=120); plt.close(fig)

    # image size not multiple of 32 and 16-bit path (B0 only)
    rows = []
    for hw in [(128, 128), (100, 150), (250, 250), (96, 200)]:
        h, w = hw
        flow = uni(2.5)
        A, B = [], []
        for rr in range(n):
            rng = np.random.default_rng(900 + rr)
            pf = ParticleField(h, w, REF_P, rng)
            A.append(dg.apply(pf.render(), ref_deg, 0, rng)); pf.advect(flow, 0, 1); B.append(dg.apply(pf.render(), ref_deg, 1, rng))
        d = M.b0.predict(np.stack(A), np.stack(B))
        e = np.sqrt(((d - np.array([flow.u, flow.v])[None, :, None, None]) ** 2).sum(1))[:, 8:-8, 8:-8]
        rows.append({"factor": "image_size", "value": f"{h}x{w}", "B0_dense_epe": float(e.mean()),
                     "B0_bias_dx": float((d[:, 0] - flow.u)[:, 8:-8, 8:-8].mean())})
    A8, B8 = render_pairs(uni(2.5), n, size, REF_P, ref_deg, 31)
    d8 = M.b0.predict(A8, B8)
    d16 = M.b0.predict((A8.astype(np.uint16) * 257), (B8.astype(np.uint16) * 257), bit_depth=16)
    d12 = M.b0.predict((A8.astype(np.uint16) * 16), (B8.astype(np.uint16) * 16), bit_depth=12)
    rows.append({"factor": "bit_depth_path", "value": "8 vs 16-bit(x257) vs 12-bit(x16)",
                 "max_abs_diff_16": float(np.abs(d8 - d16).max()), "max_abs_diff_12": float(np.abs(d8 - d12).max())})
    print(rows, flush=True)
    write_rows("image_size_bitdepth", rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
