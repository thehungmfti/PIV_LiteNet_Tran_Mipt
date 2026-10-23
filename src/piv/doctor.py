"""Environment check: imports, GPU, HDF5 (path with spaces), third-party files, MATLAB/Octave."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD = ROOT / "third_party"
EXPECTED = {
    "third_party/PIV-LiteFlowNet-en/caffe/models/PIV-LiteFlowNet-en/PIV-LiteFlowNet-en.caffemodel":
        "see configs/baseline.yaml weights_sha256",
}


def run_doctor() -> dict:
    rep = {"checks": {}, "ok": True}

    def check(name, fn, required=True):
        try:
            rep["checks"][name] = {"status": "PASS", "detail": fn()}
        except Exception as e:  # noqa: BLE001
            rep["checks"][name] = {"status": "FAIL" if required else "NOT AVAILABLE", "detail": str(e)}
            if required:
                rep["ok"] = False

    from piv.io.manifest import versions
    check("versions", versions)

    def torch_check():
        import torch
        return {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "threads": torch.get_num_threads()}
    check("torch", torch_check)

    def h5_check():
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory(prefix="piv doctor ") as tmp:
            p = Path(tmp) / "dir with spaces" / "t.h5"
            p.parent.mkdir()
            a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            with h5py.File(p, "w") as f:
                f["x"] = a
            with h5py.File(p, "r") as f:
                assert (f["x"][()] == a).all()
        return {"h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version, "path_with_spaces": "ok"}
    check("hdf5", h5_check)

    def weights_check():
        import yaml
        cfg = yaml.safe_load((ROOT / "configs" / "baseline.yaml").read_text())
        p = ROOT / cfg["backend"]["weights_path"]
        if not p.is_file():
            raise FileNotFoundError(f"{p} missing: run scripts/fetch_third_party.sh (B0 unavailable)")
        from piv.io.manifest import sha256_file
        sha = sha256_file(p)
        if sha != cfg["backend"]["weights_sha256"]:
            raise ValueError(f"sha256 mismatch {sha}")
        return {"caffemodel": str(p), "sha256": sha}
    check("baseline_weights", weights_check, required=False)

    check("matlab", lambda: shutil.which("matlab") or (_ for _ in ()).throw(FileNotFoundError("matlab not on PATH")),
          required=False)

    def octave_check():
        o = shutil.which("octave-cli")
        if not o:
            raise FileNotFoundError("octave-cli not on PATH")
        v = subprocess.run([o, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
        return {"path": o, "version": v}
    check("octave", octave_check, required=False)
    check("pivlab_source", lambda: str(THIRD / "PIVlab") if (THIRD / "PIVlab" / "+piv").is_dir()
          else (_ for _ in ()).throw(FileNotFoundError("third_party/PIVlab missing")), required=False)
    check("openpiv", lambda: __import__("openpiv").__version__, required=False)
    return rep


if __name__ == "__main__":
    print(json.dumps(run_doctor(), indent=2))
