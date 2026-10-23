"""Run provenance: seed, resolved config, git SHA, versions, data/checkpoint hashes, timing."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def git_sha(repo: str | Path | None = None) -> str:
    try:
        repo = repo or Path(__file__).resolve().parents[3]
        sha = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "-C", str(repo), "diff", "--quiet"], stderr=subprocess.DEVNULL)
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown (not a git checkout)"


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch", "h5py", "yaml", "matplotlib", "openpiv", "skimage"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = "not installed"
    return out


class RunManifest:
    def __init__(self, run_dir: str | Path, config: dict, kind: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.data = {
            "kind": kind,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "git_sha": git_sha(),
            "versions": versions(),
            "config": config,
            "demo_only": bool(config.get("backend", {}).get("demo_only", False)),
            "inputs": {}, "outputs": {}, "timing_s": {}, "warnings": [],
        }
        self._t = {}

    def add_input(self, name: str, path: str | Path):
        p = Path(path)
        self.data["inputs"][name] = {"path": str(p), "sha256": sha256_file(p) if p.is_file() else None}

    def add_output(self, name: str, path: str | Path):
        self.data["outputs"][name] = str(path)

    def warn(self, msg: str):
        self.data["warnings"].append(msg)

    def tic(self, key):
        self._t[key] = time.perf_counter()

    def toc(self, key):
        self.data["timing_s"][key] = self.data["timing_s"].get(key, 0.0) + time.perf_counter() - self._t.pop(key)

    def set(self, key, value):
        self.data[key] = value

    def save(self, name: str = "manifest.json") -> Path:
        p = self.run_dir / name
        p.write_text(json.dumps(self.data, indent=2, default=str))
        return p
