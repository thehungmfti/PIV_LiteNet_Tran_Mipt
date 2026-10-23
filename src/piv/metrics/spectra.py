"""Temporal spectral metrics at fixed probes (same fs, detrend, window for every method).

* PSD: one-sided density (unit^2/Hz). Periodogram for a single segment, Welch
  (nperseg, noverlap declared) for averaged estimates.
* f_peak: argmax of PSD inside the declared band (resolution fs/nperseg).
* band power: trapezoidal integral of PSD over the band.
* phase at f_ref: arg of the cross-spectrum (pred vs GT) from Welch; reported only
  where the magnitude-squared coherence >= coh_min. Coherence always uses Welch
  averaging (a single periodogram has coherence identically 1).
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def psd(x: np.ndarray, fs: float, nperseg: int | None = None, noverlap: int | None = None,
        window: str = "hann", detrend: str = "constant"):
    x = np.asarray(x, np.float64)
    n = x.shape[-1]
    if nperseg is None or nperseg >= n:
        return signal.periodogram(x, fs=fs, window=window, detrend=detrend, scaling="density", axis=-1)
    return signal.welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend,
                        scaling="density", axis=-1)


def band_power(f, p, band):
    sel = (f >= band[0]) & (f <= band[1])
    if sel.sum() < 2:
        return float(np.sum(p[..., sel], -1) * (f[1] - f[0]))
    return np.trapezoid(p[..., sel], f[sel], axis=-1)


def peak_frequency(f, p, band):
    sel = np.where((f >= band[0]) & (f <= band[1]))[0]
    return f[sel[np.argmax(p[..., sel], -1)]]


def probe_signals(seq: np.ndarray, probes: list[tuple[int, int]]) -> np.ndarray:
    """seq [N,2,H,W] -> [P,2,N] time series at pixel probes (row=y, col=x)."""
    return np.stack([seq[:, :, y, x].T for (y, x) in probes])


def spectral_metrics(pred_seq, gt_seq, fs: float, probes, band, f_ref: float | None = None,
                     nperseg: int = 64, noverlap: int | None = None, coh_min: float = 0.5,
                     window: str = "hann", active_frac: float = 0.1) -> dict:
    """Per-signal (probe x component) metrics. Relative band-power, peak, phase and coherence
    statistics use only ENERGETIC signals: GT band power >= active_frac x the maximum over the
    signals of this run (a fixed definition from GT, declared before evaluation). The global
    ratio sum(P_pred)/sum(P_gt) over all signals is reported as well."""
    sp = probe_signals(pred_seq, probes)
    sg = probe_signals(gt_seq, probes)
    noverlap = nperseg // 2 if noverlap is None else noverlap
    f, pp = psd(sp, fs, nperseg, noverlap, window)
    _, pg = psd(sg, fs, nperseg, noverlap, window)
    bp_p = band_power(f, pp, band)
    bp_g = band_power(f, pg, band)
    fp_p = peak_frequency(f, pp, band)
    fp_g = peak_frequency(f, pg, band)
    # relative band-power errors are only defined where the GT actually fluctuates in the band
    active = bp_g >= active_frac * float(np.max(bp_g)) if np.max(bp_g) > 1e-14 else np.zeros_like(bp_g, bool)
    rel = np.abs(bp_p - bp_g)[active] / bp_g[active]
    out = {
        "fs": fs, "nperseg": nperseg, "noverlap": noverlap, "df": fs / nperseg, "band": list(band),
        "n_signals": int(bp_g.size), "n_signals_active": int(active.sum()),
        "band_power_rel_err_mean": float(rel.mean()) if rel.size else None,
        "band_power_rel_err_median": float(np.median(rel)) if rel.size else None,
        "band_power_ratio_mean": float(np.mean(bp_p[active] / bp_g[active])) if rel.size else None,
        "band_power_pred_mean": float(bp_p.mean()), "band_power_gt_mean": float(bp_g.mean()),
        "band_power_ratio_global": float(bp_p.sum() / bp_g.sum()) if bp_g.sum() > 0 else None,
        "active_frac": active_frac,
        "peak_freq_abs_err_mean_hz": float(np.mean(np.abs(fp_p - fp_g)[active])) if active.any() else None,
        "peak_freq_match_frac": float(np.mean((np.abs(fp_p - fp_g) <= fs / nperseg + 1e-9)[active]))
        if active.any() else None,
        "rms_rel_err_mean": float(np.mean((np.abs(sp.std(-1) - sg.std(-1)) / np.maximum(sg.std(-1), 1e-20))[active]))
        if active.any() else None,
        "log_psd_l1_band": float(np.mean(np.abs(np.log(pp[..., (f >= band[0]) & (f <= band[1])] + 1e-20)
                                               - np.log(pg[..., (f >= band[0]) & (f <= band[1])] + 1e-20))[active]))
        if active.any() else None,
    }
    if f_ref is not None:
        fc, pxy = signal.csd(sp, sg, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, axis=-1)
        _, coh = signal.coherence(sp, sg, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, axis=-1)
        k = int(np.argmin(np.abs(fc - f_ref)))
        ph = np.angle(pxy[..., k])
        ok = (coh[..., k] >= coh_min) & active
        out.update({
            "f_ref": f_ref, "f_ref_bin": float(fc[k]),
            "coherence_at_fref_mean": float(coh[..., k][active].mean()) if active.any() else None,
            "phase_err_at_fref_deg_mean": float(np.degrees(np.abs(ph[ok])).mean()) if ok.any() else None,
            "phase_valid_frac": float(ok.sum() / max(active.sum(), 1)),
            "coherence_band_mean": float(coh[..., (fc >= band[0]) & (fc <= band[1])][active].mean())
            if active.any() else None,
        })
    return out
