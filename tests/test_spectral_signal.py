"""Spectral acceptance: tone in the right FFT bin (+-1), integrated PSD = variance, torch PSD == scipy."""
import numpy as np
import torch
from scipy import signal

from piv.losses.spectral import psd_torch, spectral_loss
from piv.metrics.spectra import band_power, peak_frequency, psd, spectral_metrics


def test_tone_in_correct_bin():
    fs, n = 1000.0, 128
    for f0 in (62.5, 70.0, 123.4, 300.0):
        t = np.arange(n) / fs
        x = np.sin(2 * np.pi * f0 * t) + 0.01 * np.random.default_rng(0).normal(size=n)
        f, p = psd(x, fs)
        fp = peak_frequency(f, p, (5, 499))
        assert abs(fp - f0) <= fs / n + 1e-9


def test_integrated_psd_equals_variance():
    fs, n = 500.0, 256
    x = np.random.default_rng(1).normal(size=n)
    f, p = signal.periodogram(x, fs=fs, window="boxcar", detrend="constant", scaling="density")
    assert np.isclose(np.sum(p) * (f[1] - f[0]), x.var(), rtol=1e-10)
    f, p = psd(x, fs)                        # Hann: equality holds on average (window power normalised)
    vals = []
    for s in range(200):
        y = np.random.default_rng(s).normal(size=n)
        f, p = psd(y, fs)
        vals.append(np.sum(p) * (f[1] - f[0]) / y.var())
    assert abs(np.mean(vals) - 1) < 0.02


def test_torch_psd_matches_scipy():
    fs, n = 1000.0, 128
    x = np.random.default_rng(2).normal(size=(3, n))
    f1, p1 = signal.periodogram(x, fs=fs, window="hann", detrend="constant", scaling="density", axis=-1)
    f2, p2 = psd_torch(torch.from_numpy(x), fs)
    assert np.allclose(f1, f2.numpy()) and np.allclose(p1, p2.numpy(), rtol=1e-8, atol=1e-14)


def test_spectral_loss_zero_for_identical_and_gradient_flows():
    x = torch.randn(1, 4, 2, 128, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone()
    assert spectral_loss(x, y, 1000.0, 15, 300).item() < 1e-12
    l = spectral_loss(x, y + 0.1 * torch.randn_like(y), 1000.0, 15, 300)
    l.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_band_power_and_metrics_on_known_tone():
    fs, n = 1000.0, 256
    t = np.arange(n) / fs
    gt = np.zeros((n, 2, 4, 4)); gt[:, 0] = np.sin(2 * np.pi * 125 * t)[:, None, None]
    pred = 1.1 * gt
    m = spectral_metrics(pred, gt, fs, [(1, 1), (2, 2)], (50, 200), f_ref=125.0, nperseg=64)
    assert abs(m["band_power_ratio_mean"] - 1.21) < 1e-6
    assert m["peak_freq_match_frac"] == 1.0
    assert m["phase_err_at_fref_deg_mean"] < 1e-6
