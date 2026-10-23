"""Sign / axis conventions: channel 0 = dx right, channel 1 = dy_image down; v_phys = -sy*dy/dt."""
import numpy as np
import pytest

from piv.data.calibration import CalibrationError, displacement_to_velocity
from piv.data.synthetic import ParticleConfig, ParticleField, UniformFlow, gt_displacement
from tests.conftest import needs_b0


def test_gt_uniform_signs():
    d, _, _ = gt_displacement(UniformFlow(u=2.0, v=-1.5), 16, 16, 0.0, 1.0)
    assert np.allclose(d[0], 2.0, atol=1e-6) and np.allclose(d[1], -1.5, atol=1e-6)


def test_renderer_moves_particles_right_and_down():
    rng = np.random.default_rng(1)
    pf = ParticleField(64, 64, ParticleConfig(ppp=0.02), rng)
    a = pf.render()
    pf.advect(UniformFlow(u=3.0, v=2.0), 0, 1)
    b = pf.render()
    # cross-correlation peak location
    F = np.fft.ifft2(np.conj(np.fft.fft2(a - a.mean())) * np.fft.fft2(b - b.mean())).real
    iy, ix = np.unravel_index(np.argmax(F), F.shape)
    assert (ix, iy) == (3, 2)


def test_velocity_conversion_signs():
    disp = np.zeros((1, 2, 2, 2), np.float32)
    disp[:, 0] = 2.0     # right
    disp[:, 1] = 3.0     # down in the image
    v = displacement_to_velocity(disp, np.array([1e-3]), {"mode": "scalar", "sx_m_per_px": 1e-4, "sy_m_per_px": 2e-4})
    assert np.allclose(v[:, 0], 0.2) and np.allclose(v[:, 1], -0.6)


def test_missing_calibration_refused():
    with pytest.raises(CalibrationError):
        displacement_to_velocity(np.zeros((1, 2, 2, 2)), np.array([1e-3]), {})
    with pytest.raises(CalibrationError):
        displacement_to_velocity(np.zeros((1, 2, 2, 2)), np.array([0.0]),
                                 {"mode": "scalar", "sx_m_per_px": 1e-4, "sy_m_per_px": 1e-4})


@needs_b0
def test_b0_sign_and_scale(b0):
    rng = np.random.default_rng(3)
    pf = ParticleField(64, 64, ParticleConfig(), rng)
    a = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
    pf.advect(UniformFlow(u=2.5, v=-1.5), 0, 1)
    b = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
    d = b0.predict(a, b)
    c = d[:, 8:-8, 8:-8]
    assert abs(c[0].mean() - 2.5) < 0.1 and abs(c[1].mean() + 1.5) < 0.1
