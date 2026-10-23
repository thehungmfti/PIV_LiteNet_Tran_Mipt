"""Run PIVlab (via GNU Octave or MATLAB) on a schema-v1 dataset and write a grid prediction.

PIVlab output convention (checked in tests/test_comparators.py on uniform/shear flows):
x, y are 1-based pixel coordinates of the vector positions; u > 0 right, v > 0 down
(image coordinates, uncalibrated pixel/pair). We store grid_x = x - 1, grid_y = y - 1.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from piv.io.hdf5 import read_dataset, write_prediction

HERE = Path(__file__).resolve().parent
DEFAULT = dict(ia1=64, step=8, passes=3, ia2=32, ia3=16, ia4=16, subpix=1, imdeform="*linear", repeat_last=0,
               do_stdev=1, stdthresh=7, do_median=1, neigh_thresh=3, validated=True)


def find_engine(prefer: str | None = None) -> list[str]:
    if prefer == "matlab" or (prefer is None and shutil.which("matlab")):
        return ["matlab", "-batch"]
    if shutil.which("octave-cli"):
        return ["octave-cli", "--no-gui", "--quiet", "--eval"]
    raise RuntimeError("neither MATLAB nor GNU Octave found: PIVlab comparator NOT RUN")


def run_arrays(A: np.ndarray, B: np.ndarray, pivlab_root: str, settings: dict | None = None, engine=None,
               chunk: int = 400):
    """A, B [N,H,W] uint8 -> (grid_x 0-based, grid_y 0-based, disp [N,2,ny,nx], seconds)."""
    s = {**DEFAULT, **(settings or {})}
    if A.dtype != np.uint8:
        raise ValueError("this runner passes 8-bit images to PIVlab; convert explicitly first")
    cmd = find_engine(engine)
    adapter = HERE / "pivlab_octave"
    Us, Vs = [], []
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(0, len(A), chunk):
            fin, fout = Path(tmp) / f"in_{i}.mat", Path(tmp) / f"out_{i}.mat"
            savemat(fin, {**{k: v for k, v in s.items() if k != "validated"},
                          "images_a": A[i:i + chunk], "images_b": B[i:i + chunk]}, do_compression=True)
            call = f"addpath('{adapter}'); run_pivlab_batch('{fin}', '{fout}', '{pivlab_root}');"
            r = subprocess.run(cmd + [call], capture_output=True, text=True)
            if r.returncode != 0 or not fout.is_file():
                raise RuntimeError(f"PIVlab failed ({r.returncode}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            o = loadmat(fout)
            key = ("Uf", "Vf") if s["validated"] else ("U", "V")
            Us.append(o[key[0]].reshape(-1, *o["x"].shape)); Vs.append(o[key[1]].reshape(-1, *o["x"].shape))
            x, y = o["x"], o["y"]
    disp = np.stack([np.concatenate(Us), np.concatenate(Vs)], 1).astype(np.float32)
    return x[0, :] - 1.0, y[:, 0] - 1.0, disp, time.perf_counter() - t0


def run(dataset: str, output: str, pivlab_root: str, settings: dict | None = None, engine=None,
        chunk: int = 400) -> dict:
    s = {**DEFAULT, **(settings or {})}
    d = read_dataset(dataset)
    A = np.stack([d.pair(k)[0] for k in range(d.n_pairs)])
    B = np.stack([d.pair(k)[1] for k in range(d.n_pairs)])
    gx, gy, disp, secs = run_arrays(A, B, pivlab_root, s, engine, chunk)
    commit = subprocess.run(["git", "-C", pivlab_root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = {"model_id": "C1_PIVlab_FFTmulti", "engine": find_engine(engine)[0], "pivlab_commit": commit, "settings": s,
            "adapter": "comparators/pivlab_octave (interface-only changes)", "seconds_total": secs,
            "seconds_per_pair": secs / d.n_pairs, "dataset": str(dataset), "units": "pixel/pair",
            "grid_convention": "PIVlab x,y (1-based) minus 1", "demo_only": False}
    write_prediction(output, disp, d.t_a, d.t_b, valid=np.all(np.isfinite(disp), 1), grid_x=gx, grid_y=gy,
                     provenance=prov)
    return prov
