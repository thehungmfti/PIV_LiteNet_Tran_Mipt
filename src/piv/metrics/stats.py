"""Confidence intervals by moving-block bootstrap over time within sequences and
resampling of sequences (runs). Pixels are never treated as independent samples."""
from __future__ import annotations

import numpy as np


def block_bootstrap_ci(per_frame: dict[str, np.ndarray], block: int = 16, n_boot: int = 2000,
                       level: float = 0.95, seed: int = 0, stat=np.nanmean) -> dict:
    """per_frame: run_id -> [T] per-pair metric. Returns point estimate and CI of the
    mean over all frames, resampling runs (with replacement) then moving blocks in each run."""
    rng = np.random.default_rng(seed)
    runs = list(per_frame)
    point = float(stat(np.concatenate([per_frame[r] for r in runs])))
    boots = []
    for _ in range(n_boot):
        sample = []
        for r in rng.choice(runs, len(runs), replace=True):
            x = per_frame[r]
            T = len(x)
            b = min(block, T)
            starts = rng.integers(0, T - b + 1, int(np.ceil(T / b)))
            sample.append(np.concatenate([x[s:s + b] for s in starts])[:T])
        boots.append(stat(np.concatenate(sample)))
    lo, hi = np.percentile(boots, [50 * (1 - level), 100 - 50 * (1 - level)])
    return {"mean": point, "ci_low": float(lo), "ci_high": float(hi), "level": level,
            "block": block, "n_boot": n_boot, "n_runs": len(runs)}


def paired_difference_ci(a: dict[str, np.ndarray], b: dict[str, np.ndarray], **kw) -> dict:
    """CI of mean(a - b) with the same block structure (paired by run and frame)."""
    return block_bootstrap_ci({r: a[r] - b[r] for r in a}, **kw)
