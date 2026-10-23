"""Figures from benchmark outputs (PNG, light theme): error vs noise, spectra, UQ reliability, field maps.

python scripts/make_figures.py
Colors: validated 8-slot categorical palette, one fixed slot per method (never re-assigned by rank).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from piv.io.hdf5 import read_dataset, read_prediction  # noqa: E402
from piv.metrics import spectra  # noqa: E402

B = ROOT / "outputs" / "benchmark"
FIG = ROOT / "outputs" / "figures"
COLOR = {"B0": "#2a78d6", "B1": "#eb6834", "M1": "#1baf7a", "S1": "#eda100", "M2": "#e87ba4",
         "M3": "#008300", "PIVlab": "#6250d6", "OpenPIV": "#e34948"}
MARK = {"B0": "o", "B1": "s", "M1": "^", "S1": "v", "M2": "D", "M3": "P", "PIVlab": "X", "OpenPIV": "*"}
NOISE = ["clean", "low", "medium", "high"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.edgecolor": "#52514e", "font.size": 9, "lines.linewidth": 2})


def summary(tag="main"):
    p = B / f"summary_{tag}.json"
    return json.loads(p.read_text()) if p.is_file() else []


def plot_metric_vs_noise(rows, metric, ylabel, fname, split="test", methods=None, logy=True):
    rows = [r for r in rows if r["split"] == split and r.get(metric) is not None]
    methods = methods or [m for m in COLOR if any(r["method"] == m for r in rows)]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for m in methods:
        pts = [(NOISE.index(r["noise"]), r[metric]) for r in rows if r["method"] == m and r["noise"] in NOISE]
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[m], ms=8, color=COLOR[m], label=m)
    ax.set_xticks(range(len(NOISE)), NOISE)
    ax.set_xlabel("image noise level (test split)")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=140); plt.close(fig)


def plot_psd_example(run_stem, methods, probe=(64, 64), nperseg=64):
    ref = next((ROOT / "data" / "research").rglob(f"{run_stem}.h5"), None)
    if ref is None:
        return
    d = read_dataset(ref, load_images=False)
    fs = 1.0 / float(np.mean(np.diff(0.5 * (d.t_a + d.t_b))))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    sl = slice(2, -2)
    for c, name in enumerate(("dx", "dy")):
        f, P = spectra.psd(d.gt_displacement[sl, c, probe[0], probe[1]], fs, nperseg, nperseg // 2)
        ax[c].semilogy(f[1:], P[1:], color="#0b0b0b", lw=2.5, label="GT")
        for m in methods:
            pp = B / "pred" / m / f"{run_stem}.h5"
            if not pp.is_file():
                continue
            p = read_prediction(pp)
            if "grid_x" in p:
                gi = int(np.argmin(np.abs(p["grid_y"] - probe[0]))); gj = int(np.argmin(np.abs(p["grid_x"] - probe[1])))
                sig = p["displacement"][sl, c, gi, gj]
            else:
                sig = p["displacement"][sl, c, probe[0], probe[1]]
            f, P = spectra.psd(sig, fs, nperseg, nperseg // 2)
            ax[c].semilogy(f[1:], P[1:], color=COLOR[m], lw=1.6, label=m)
        ax[c].set_xlabel("frequency (Hz)"); ax[c].set_ylabel(f"PSD of {name} (px²/Hz)")
        ax[c].set_title(f"probe y={probe[0]}, x={probe[1]}; Welch, Hann, N={nperseg}")
    ax[1].legend(fontsize=8, frameon=False, ncol=2)
    fig.suptitle(run_stem, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"psd_{run_stem}.png", dpi=140); plt.close(fig)


def plot_uq(method="M3", split="test"):
    rel = {}
    for p in sorted((B / "metrics" / method).glob("*.json")):
        r = json.loads(p.read_text())
        if r.get("split") != split or "uq" not in r:
            continue
        rel.setdefault(r["noise"], []).append(r["uq"])
    if not rel:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, nz in enumerate(n for n in NOISE if n in rel):
        rows = [b for u in rel[nz] for b in u["dx_reliability"]]
        xs = [b["std_pred_rms"] for b in rows]; ys = [b["err_rms"] for b in rows]
        ax[0].scatter(xs, ys, s=18, color=list(COLOR.values())[i], label=nz)
        cov68 = np.mean([u["dx_coverage_68"] for u in rel[nz]]); cov95 = np.mean([u["dx_coverage_95"] for u in rel[nz]])
        ax[1].bar([i - 0.2, i + 0.2], [cov68, cov95], width=0.36, color=[list(COLOR.values())[i]] * 2,
                  edgecolor="white", linewidth=2)
    lim = ax[0].get_xlim()[1]
    ax[0].plot([0, lim], [0, lim], color="#52514e", lw=1, ls="--", label="ideal")
    ax[0].set_xlabel("predicted σ (px), bin RMS"); ax[0].set_ylabel("actual error RMS (px)")
    ax[0].set_title(f"{method} reliability (dx), test"); ax[0].legend(fontsize=8, frameon=False)
    ax[1].axhline(0.68, color="#52514e", lw=1, ls="--"); ax[1].axhline(0.95, color="#52514e", lw=1, ls="--")
    ax[1].set_xticks(range(len(rel)), [n for n in NOISE if n in rel])
    ax[1].set_ylabel("empirical coverage (dx)"); ax[1].set_title("bars: 68 % (left) and 95 % (right) intervals")
    fig.tight_layout(); fig.savefig(FIG / f"uq_{method}.png", dpi=140); plt.close(fig)


def plot_field_maps(run_stem, methods, k=100):
    ref = next((ROOT / "data" / "research").rglob(f"{run_stem}.h5"), None)
    if ref is None:
        return
    d = read_dataset(ref)
    gt = d.gt_displacement[k]
    cols = len(methods) + 1
    fig, ax = plt.subplots(2, cols, figsize=(2.6 * cols, 5.2))
    vmax = np.abs(gt).max()
    ax[0, 0].imshow(d.pair(k)[0], cmap="gray"); ax[0, 0].set_title("image a")
    w = np.gradient(gt[0], axis=0) - np.gradient(gt[1], axis=1)
    ax[1, 0].imshow(w, cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max()); ax[1, 0].set_title("GT vorticity")
    for j, m in enumerate(methods, start=1):
        p = read_prediction(B / "pred" / m / f"{run_stem}.h5")
        disp = p["displacement"][k]
        if "grid_x" in p:
            from piv.engine.evaluate import gt_on_points, flow_of
            flow, _ = flow_of(d)
            g = gt_on_points(flow, p["grid_x"], p["grid_y"], d.t_a[k:k + 1], d.t_b[k:k + 1])[0]
            err = np.sqrt(((disp - g) ** 2).sum(0))
            extent = (p["grid_x"][0] - 4, p["grid_x"][-1] + 4, p["grid_y"][-1] + 4, p["grid_y"][0] - 4)
            ax[0, j].imshow(disp[1], cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=extent)
            ax[1, j].imshow(err, cmap="magma", vmin=0, vmax=0.3, extent=extent)
        else:
            err = np.sqrt(((disp - gt) ** 2).sum(0))
            ax[0, j].imshow(disp[1], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            im = ax[1, j].imshow(err, cmap="magma", vmin=0, vmax=0.3)
        ax[0, j].set_title(f"{m}: dy"); ax[1, j].set_title(f"EPE mean {np.nanmean(err):.3f}")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle(f"{run_stem}, pair {k}; EPE colour scale 0-0.3 px", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"fields_{run_stem}.png", dpi=130); plt.close(fig)


def main():
    rows = summary("main")
    for metric, lab, fn in [("dense_epe", "mean EPE, dense interior (px)", "epe_dense_vs_noise.png"),
                            ("grid_epe", "mean EPE at common 8-px grid (px)", "epe_grid_vs_noise.png"),
                            ("dense_bandpower_relerr", "band-power rel. error (median), dense", "bandpower_dense_vs_noise.png"),
                            ("grid_bandpower_relerr", "band-power rel. error (median), grid", "bandpower_grid_vs_noise.png"),
                            ("dense_logpsd_l1", "mean |log PSD_pred - log PSD_GT|, 15-300 Hz", "logpsd_dense_vs_noise.png"),
                            ("grid_vort_rmse", "vorticity RMSE at grid (1/pair)", "vorticity_grid_vs_noise.png")]:
        plot_metric_vs_noise(rows, metric, lab, fn)
    ex = sorted((B / "pred" / "B0").glob("test_vortex_street*__high.h5"))
    for e in ex[:1] + sorted((B / "pred" / "B0").glob("test_cellular*__high.h5"))[:1]:
        plot_psd_example(e.stem, ["B0", "B1", "M1", "M2", "PIVlab"])
        plot_field_maps(e.stem, [m for m in ("B0", "B1", "M2", "PIVlab", "OpenPIV") if (B / "pred" / m / f"{e.stem}.h5").is_file()])
    plot_uq("M3")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
