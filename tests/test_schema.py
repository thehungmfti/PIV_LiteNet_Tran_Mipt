import numpy as np
import pytest

from piv.data.schema import validate_config, validate_dataset, validate_spectral
from piv.io.hdf5 import PIVDataset, read_dataset, read_prediction, write_dataset, write_prediction


def make(n=6, h=8, w=10, mode="continuous"):
    t = np.arange(n + 1) * 1e-3
    imgs = np.arange((n + 1) * h * w, dtype=np.uint16).reshape(n + 1, h, w) % 255
    gt = np.random.default_rng(0).normal(size=(n, 2, h, w)).astype(np.float32)
    return PIVDataset(pairs_a=np.arange(n), pairs_b=np.arange(1, n + 1), t_a=t[:-1], t_b=t[1:],
                      images=imgs.astype(np.uint8), gt_displacement=gt, valid=np.ones((n, h, w), bool),
                      calibration={"mode": "scalar", "sx_m_per_px": 1e-4, "sy_m_per_px": 1e-4},
                      attrs={"acquisition_mode": mode, "bit_depth": 8, "case_id": "c", "run_id": "r"})


def test_roundtrip_distinct_axes(tmp_path):
    """Every logical axis gets distinct values so a transposition would be detected."""
    d = make()
    n, _, h, w = d.gt_displacement.shape
    g = (np.arange(n)[:, None, None, None] * 1000 + np.arange(2)[None, :, None, None] * 100
         + np.arange(h)[None, None, :, None] * 10 + np.arange(w)[None, None, None, :] * 0.1).astype(np.float32)
    d.gt_displacement = g
    p = tmp_path / "dir with spaces" / "x.h5"
    write_dataset(p, d)
    r = read_dataset(p)
    assert np.array_equal(r.gt_displacement, g)
    assert np.array_equal(r.images, d.images)
    assert r.calibration["sx_m_per_px"] == 1e-4
    import h5py
    with h5py.File(p) as f:
        assert f["/gt/displacement"].attrs["axes"] == "N,C,H,W"
        assert f.attrs["schema_version"] == "1.0"


def test_prediction_nan_outside_valid(tmp_path):
    disp = np.ones((2, 2, 4, 4), np.float32)
    valid = np.ones((2, 4, 4), bool)
    valid[0, 0, 0] = False
    write_prediction(tmp_path / "p.h5", disp, [0, 1], [0.5, 1.5], valid=valid, provenance={"model_id": "x"})
    p = read_prediction(tmp_path / "p.h5")
    assert np.isnan(p["displacement"][0, :, 0, 0]).all()
    assert np.isfinite(p["displacement"][1]).all()
    assert p["provenance"]["model_id"] == "x"


def test_validate_ok_and_errors():
    d = make()
    assert validate_dataset(d, need_calibration=True).ok
    d2 = make()
    d2.calibration = {}
    r = validate_dataset(d2, need_calibration=True)
    assert not r.ok and any("calibration" in e for e in r.errors)
    d3 = make()
    d3.valid[:] = False
    assert not validate_dataset(d3).ok
    d4 = make()
    d4.t_b = d4.t_a.copy()
    assert not validate_dataset(d4).ok


def test_irregular_timestamps_rejected_for_spectra():
    d = make()
    d.t_a = d.t_a.copy(); d.t_b = d.t_b.copy()
    d.t_a[3] += 2e-4; d.t_b[3] += 2e-4
    r = validate_dataset(d, spectral={"segment_length": 4, "fmax": 100})
    assert not r.ok and any("irregular" in e for e in r.errors)


def test_spectral_validator():
    r = validate_spectral({"segment_length": 256, "fmax": 600}, fs=1000, n_samples=128)
    assert any("Nyquist" in e for e in r.errors)
    assert any("segment_length" in e for e in r.errors)


def test_config_validator():
    bad = {"project": {}, "backend": {}, "model": {"context_fields": 4},
           "training": {"sequence_length": 64}, "spectral": {"segment_length": 128}}
    r = validate_config(bad)
    assert len(r.errors) == 2
