"""Sinusoidal spatial response: amplitude gain of a method versus wavelength.

A steady displacement field dy = A sin(2 pi x / lambda) is imaged; the measured
amplitude is the least-squares projection of the prediction on the sine along x
(interior columns only). Gain < 1 indicates spatial filtering. Effective
resolution must be read from this curve, never from the vector grid spacing.
"""
from __future__ import annotations

import math

import numpy as np

from piv.data.synthetic import Flow, ParticleConfig, ParticleField


class SineDisplacement(Flow):
    """Steady flow whose exact pair displacement is (0, A sin(2 pi x / lambda)) for dt = 1:
    v depends on x only and u = 0, so particles move straight in y."""
    name = "sine"

    def __init__(self, amp: float, wavelength: float):
        self.amp, self.wavelength = amp, wavelength

    def velocity(self, x, y, t):
        return np.zeros_like(x, dtype=np.float64), self.amp * np.sin(2 * math.pi * x / self.wavelength)


def fitted_amplitude(field_dy: np.ndarray, wavelength: float, margin: int) -> float:
    h, w = field_dy.shape
    x = np.arange(w)
    s = np.sin(2 * math.pi * x / wavelength)
    c = np.cos(2 * math.pi * x / wavelength)
    cols = slice(margin, w - margin)
    rows = slice(margin, h - margin)
    prof = np.nanmean(field_dy[rows, :], 0)[cols]
    A = np.stack([s[cols], c[cols], np.ones_like(s[cols])], 1)
    coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
    return float(math.hypot(coef[0], coef[1]))


def response_curve(predict, wavelengths, amp: float = 2.0, size: int = 256, seed: int = 0,
                   n_rep: int = 2, particles: ParticleConfig | None = None, margin: int = 16):
    """predict(img_a, img_b) -> [2,H,W]. Returns list of dicts (wavelength, gain)."""
    rows = []
    for lam in wavelengths:
        gains = []
        for r in range(n_rep):
            rng = np.random.default_rng(seed + 1000 * r + int(lam))
            pf = ParticleField(size, size, particles or ParticleConfig(), rng)
            a = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
            pf.advect(SineDisplacement(amp, lam), 0.0, 1.0)
            b = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
            d = predict(a, b)
            gains.append(fitted_amplitude(d[1], lam, margin) / amp)
        rows.append({"wavelength_px": float(lam), "gain_mean": float(np.mean(gains)),
                     "gain_std": float(np.std(gains)), "n_rep": n_rep})
    return rows
