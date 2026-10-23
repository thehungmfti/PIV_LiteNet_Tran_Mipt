"""Python side of matlab/validate_roundtrip.m (axis-distinct tensors, 1-based pattern)."""
import sys
from pathlib import Path

import h5py
import numpy as np

N, C, H, W = 3, 2, 4, 5


def pattern():
    n, c, h, w = np.meshgrid(np.arange(1, N + 1), np.arange(1, C + 1), np.arange(1, H + 1), np.arange(1, W + 1),
                             indexing="ij")
    return (1000 * n + 100 * c + 10 * h + w).astype(np.float32)


if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "check":
        with h5py.File(path) as f:
            g = f["/gt/displacement"][()]
            ok = g.shape == (N, C, H, W) and np.array_equal(g, pattern()) and f["/images"].shape == (N + 1, H, W)
        print("h5py read", g.shape, "OK" if ok else "MISMATCH")
        sys.exit(0 if ok else 1)
    with h5py.File(path, "w") as f:
        f["/gt/displacement"] = pattern()
        f["/gt/displacement"].attrs["axes"] = "N,C,H,W"
    print("written", path)
