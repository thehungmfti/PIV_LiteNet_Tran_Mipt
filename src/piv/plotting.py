"""Figures (PNG): vector/error maps and probe spectra."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from piv.io.hdf5 import read_dataset, read_prediction
from piv.metrics import spectra


def plot_pair_result(data_path, pred_path, out_png, k: int = 0, step: int = 8):
    d = read_dataset(data_path)
    p = read_prediction(pred_path)
    a, _ = d.pair(k)
    gt = d.gt_displacement[k]
    pr = p["displacement"][k]
    err = np.sqrt(((pr - gt) ** 2).sum(0))
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(a, cmap="gray"); ax[0].set_title(f"image a (pair {k})")
    for axi, f, t in ((ax[1], gt, "ground truth"), (ax[2], pr, p["provenance"].get("backbone", {}).get("model_id", "pred"))):
        mag = np.sqrt((f ** 2).sum(0))
        im = axi.imshow(mag, cmap="viridis")
        yy, xx = np.mgrid[0:f.shape[1]:step, 0:f.shape[2]:step]
        axi.quiver(xx, yy, f[0, ::step, ::step], -f[1, ::step, ::step], color="w", scale_units="xy", angles="xy")
        axi.set_title(f"{t} |d| (px)"); plt.colorbar(im, ax=axi, fraction=0.046)
    im = ax[3].imshow(err, cmap="magma"); ax[3].set_title(f"EPE (px), mean {np.nanmean(err):.3f}")
    plt.colorbar(im, ax=ax[3], fraction=0.046)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def plot_spectra(data_path, pred_path, out_png, probe=None, nperseg: int = 64):
    d = read_dataset(data_path, load_images=False)
    p = read_prediction(pred_path)
    fs = 1.0 / float(np.mean(np.diff(0.5 * (d.t_a + d.t_b))))
    h, w = d.gt_displacement.shape[-2:]
    probe = probe or (h // 2, w // 2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, name in enumerate(("dx", "dy")):
        for arr, lab, ls in ((d.gt_displacement, "GT", "-"), (p["displacement"], "pred", "--")):
            f, P = spectra.psd(arr[:, c, probe[0], probe[1]], fs, nperseg, nperseg // 2)
            ax[c].semilogy(f[1:], P[1:], ls, label=lab)
        ax[c].set_xlabel("f (Hz)"); ax[c].set_ylabel(f"PSD {name} (px^2/Hz)"); ax[c].legend()
        ax[c].set_title(f"probe (y={probe[0]}, x={probe[1]}), Welch nperseg={nperseg}")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
