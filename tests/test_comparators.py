"""Cross-correlation comparators: sign, scale and vector-position conventions on known flows."""
import shutil

import numpy as np
import pytest

from piv.data.synthetic import ParticleConfig, ParticleField, ShearFlow, UniformFlow, converged_advect
from tests.conftest import ROOT

PIVLAB = ROOT / "third_party" / "PIVlab"


def pairs(flow, n=2, size=128):
    A, B = [], []
    for s in range(n):
        pf = ParticleField(size, size, ParticleConfig(), np.random.default_rng(s))
        A.append(np.clip(np.round(pf.render()), 0, 255).astype(np.uint8))
        pf.advect(flow, 0, 1)
        B.append(np.clip(np.round(pf.render()), 0, 255).astype(np.uint8))
    return np.stack(A), np.stack(B)


@pytest.mark.skipif(not (shutil.which("octave-cli") or shutil.which("matlab")) or not (PIVLAB / "+piv").is_dir(),
                    reason="PIVlab or Octave/MATLAB not available")
def test_pivlab_uniform_and_shear():
    from comparators.pivlab_runner import run_arrays
    A, B = pairs(UniformFlow(u=2.3, v=-1.4))
    gx, gy, d, _ = run_arrays(A, B, str(PIVLAB))
    assert np.allclose(gx, np.arange(8, 121, 8)) and np.allclose(gy, gx)
    assert abs(np.nanmean(d[:, 0]) - 2.3) < 0.05 and abs(np.nanmean(d[:, 1]) + 1.4) < 0.05
    sh = ShearFlow(gamma=0.03, y0=64)
    A, B = pairs(sh)
    gx, gy, d, _ = run_arrays(A, B, str(PIVLAB))
    X, Y = np.meshgrid(gx, gy)
    x1, _, _, _ = converged_advect(sh, X, Y, 0, 1)
    err_reported = np.nanmean(np.abs(d[:, 0] - (x1 - X)))
    x1h, _, _, _ = converged_advect(sh, X - 0.5, Y - 0.5, 0, 1)
    err_shift = np.nanmean(np.abs(d[:, 0] - (x1h - X + 0.5)))
    assert err_reported <= err_shift      # reported positions are the better reference


def test_openpiv_uniform():
    pytest.importorskip("openpiv")
    from comparators import openpiv_runner
    from openpiv import windef
    A, B = pairs(UniformFlow(u=2.3, v=-1.4), n=1)
    st = windef.PIVSettings()
    for k, v in openpiv_runner.DEFAULT.items():
        setattr(st, k, v)
    x, y, u, v = openpiv_runner.piv_pair(A[0].astype(float), B[0].astype(float), st)
    assert np.allclose(x[0], np.arange(8, 121, 8))
    assert abs(np.nanmean(u) - 2.3) < 0.05 and abs(np.nanmean(v) + 1.4) < 0.05
