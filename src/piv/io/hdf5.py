"""HDF5 schema v1 reader/writer (see docs/DATA_PROTOCOL.md).

Logical axes are stored as attributes on every dataset (``axes``), because
MATLAB's h5read reverses the dimension order relative to h5py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from piv import SCHEMA_VERSION

AXES = {
    "/images": "T,H,W",
    "/images_a": "N,H,W",
    "/images_b": "N,H,W",
    "/pairs/index_a": "N",
    "/pairs/index_b": "N",
    "/time/t_a": "N",
    "/time/t_b": "N",
    "/gt/displacement": "N,C,H,W",
    "/gt/velocity_mid": "N,C,H,W",
    "/masks/valid": "N,H,W",
    "/prediction/displacement": "N,C,H,W",
    "/prediction/velocity": "N,C,H,W",
    "/prediction/std": "N,C,H,W",
    "/prediction/valid": "N,H,W",
    "/prediction/t_a": "N",
    "/prediction/t_b": "N",
    "/prediction/grid_x": "W",
    "/prediction/grid_y": "H",
}
UNITS = {
    "/time/t_a": "s", "/time/t_b": "s",
    "/gt/displacement": "pixel/pair", "/gt/velocity_mid": "pixel/s",
    "/prediction/displacement": "pixel/pair", "/prediction/std": "pixel/pair",
    "/prediction/velocity": "m/s", "/prediction/t_a": "s", "/prediction/t_b": "s",
    "/prediction/grid_x": "pixel", "/prediction/grid_y": "pixel",
}


@dataclass
class PIVDataset:
    """In-memory representation of one schema-v1 dataset file."""
    pairs_a: np.ndarray                       # [N] int
    pairs_b: np.ndarray                       # [N] int
    t_a: np.ndarray                           # [N] float64 s
    t_b: np.ndarray                           # [N] float64 s
    images: np.ndarray | None = None          # [T,H,W] (continuous)
    images_a: np.ndarray | None = None        # [N,H,W] (double-frame)
    images_b: np.ndarray | None = None
    gt_displacement: np.ndarray | None = None  # [N,2,H,W] float32
    valid: np.ndarray | None = None           # [N,H,W] bool
    calibration: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)  # free-form arrays under /meta

    @property
    def n_pairs(self) -> int:
        return int(len(self.pairs_a))

    @property
    def shape(self) -> tuple[int, int]:
        im = self.images if self.images is not None else self.images_a
        if im is None:      # opened with load_images=False
            return tuple(self.extra["_shape"][-2:])
        return tuple(im.shape[-2:])

    def pair(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.images is not None:
            return self.images[self.pairs_a[k]], self.images[self.pairs_b[k]]
        return self.images_a[k], self.images_b[k]


def _write(f: h5py.File, name: str, data, compress=True):
    data = np.asarray(data)
    kw = dict(compression="gzip", compression_opts=4) if compress and data.size > 1024 else {}
    ds = f.create_dataset(name, data=data, **kw)
    if name in AXES:
        ds.attrs["axes"] = AXES[name]
    if name in UNITS:
        ds.attrs["units"] = UNITS[name]
    return ds


def write_dataset(path: str | Path, d: PIVDataset) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["origin"] = "pixel (0,0) = centre of top-left pixel; x right, y_image down"
        f.attrs["axes"] = "displacement channel 0 = dx (right), channel 1 = dy_image (down)"
        f.attrs["units"] = "displacement in pixel/pair; time in s"
        for k, v in d.attrs.items():
            f.attrs[k] = v if not isinstance(v, (dict, list)) else json.dumps(v)
        if d.images is not None:
            _write(f, "/images", d.images)
        if d.images_a is not None:
            _write(f, "/images_a", d.images_a)
            _write(f, "/images_b", d.images_b)
        _write(f, "/pairs/index_a", d.pairs_a.astype(np.int64))
        _write(f, "/pairs/index_b", d.pairs_b.astype(np.int64))
        _write(f, "/time/t_a", d.t_a.astype(np.float64))
        _write(f, "/time/t_b", d.t_b.astype(np.float64))
        if d.gt_displacement is not None:
            _write(f, "/gt/displacement", d.gt_displacement.astype(np.float32))
        if d.valid is not None:
            _write(f, "/masks/valid", d.valid.astype(np.uint8))
        g = f.create_group("calibration")
        for k, v in d.calibration.items():
            g.attrs[k] = v
        for k, v in d.extra.items():
            _write(f, f"/meta/{k}", v)


def read_dataset(path: str | Path, load_images: bool = True) -> PIVDataset:
    with h5py.File(path, "r") as f:
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        get = lambda n: f[n][()] if n in f else None
        d = PIVDataset(
            pairs_a=f["/pairs/index_a"][()], pairs_b=f["/pairs/index_b"][()],
            t_a=f["/time/t_a"][()], t_b=f["/time/t_b"][()],
            images=get("/images") if load_images else None,
            images_a=get("/images_a") if load_images else None,
            images_b=get("/images_b") if load_images else None,
            gt_displacement=get("/gt/displacement"),
            valid=(get("/masks/valid").astype(bool) if "/masks/valid" in f else None),
            calibration=dict(f["calibration"].attrs) if "calibration" in f else {},
            attrs=attrs,
            extra={k: f["meta"][k][()] for k in f["meta"]} if "meta" in f else {},
        )
        if not load_images:
            d.extra["_shape"] = f["/images" if "/images" in f else "/images_a"].shape
    return d


def write_prediction(path: str | Path, displacement: np.ndarray, t_a: np.ndarray, t_b: np.ndarray,
                     valid: np.ndarray | None = None, std: np.ndarray | None = None,
                     velocity: np.ndarray | None = None, grid_x=None, grid_y=None,
                     provenance: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    disp = np.asarray(displacement, np.float32)
    if valid is None:
        valid = np.all(np.isfinite(disp), axis=1)
    disp = np.where(valid[:, None], disp, np.nan).astype(np.float32)  # NaN outside domain
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["axes"] = "displacement channel 0 = dx (right), channel 1 = dy_image (down)"
        _write(f, "/prediction/displacement", disp)
        _write(f, "/prediction/valid", valid.astype(np.uint8))
        _write(f, "/prediction/t_a", np.asarray(t_a, np.float64))
        _write(f, "/prediction/t_b", np.asarray(t_b, np.float64))
        if std is not None:
            _write(f, "/prediction/std", np.where(valid[:, None], std, np.nan).astype(np.float32))
        if velocity is not None:
            _write(f, "/prediction/velocity", velocity.astype(np.float32))
        if grid_x is not None:
            _write(f, "/prediction/grid_x", np.asarray(grid_x, np.float64))
            _write(f, "/prediction/grid_y", np.asarray(grid_y, np.float64))
        if provenance:
            f.attrs["provenance"] = json.dumps(provenance, default=str)
    if provenance:
        path.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))


def read_prediction(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        out = {k.split("/")[-1]: f[k][()] for k in AXES if k.startswith("/prediction/") and k in f}
        out["valid"] = out["valid"].astype(bool)
        out["provenance"] = json.loads(f.attrs["provenance"]) if "provenance" in f.attrs else {}
    return out
