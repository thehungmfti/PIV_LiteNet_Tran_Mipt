"""Backbone adapters producing dense displacement [N,2,H,W] (dx right, dy_image down, pixel/pair).

B0  : original PIV-LiteFlowNet-en weights + original preprocessing, executed by
      piv.models.caffe_graph (a PyTorch interpreter of the deploy prototxt).
      It is NOT Caffe itself; parity status is recorded in docs/BASELINE_AUDIT.md.
B1  : same graph, weights fine-tuned on project data (checkpoint).
DEMO: scikit-image ILK optical flow. Clearly flagged demo_only; never a baseline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from piv.models.caffe_graph import CaffeGraph, load_caffemodel, load_net_param


class BackendUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_net_intensity(img: np.ndarray, bit_depth: int | None = None, black: float = 0.0,
                     white: float | None = None) -> np.ndarray:
    """Original demo feeds double(imread(...)) of 8-bit images (0..255) to the net.

    For other bit depths the image is mapped linearly to 0..255 using the declared
    black/white levels (project choice, recorded as preprocessing_id
    'cai2019_matlab_demo+linear_to_8bit'); there is no per-frame normalisation.
    """
    img = np.asarray(img)
    if bit_depth is None:
        bit_depth = 8 if img.dtype == np.uint8 else (16 if img.dtype == np.uint16 else None)
    if bit_depth == 8 and white is None and black == 0.0:
        return img.astype(np.float32)
    if bit_depth is None:
        raise ValueError("float images need an explicit bit_depth/white level")
    white = white if white is not None else 2 ** bit_depth - 1
    return ((img.astype(np.float32) - black) / (white - black) * 255.0).astype(np.float32)


class LiteFlowNetBackend(torch.nn.Module):
    """Wraps CaffeGraph instances (one per input size) sharing ONE set of weights."""

    def __init__(self, prototxt: str | Path, caffemodel: str | Path, expected_sha256: str | None = None,
                 model_id: str = "B0", device: str = "cpu"):
        super().__init__()
        self.prototxt = Path(prototxt)
        self.caffemodel = Path(caffemodel)
        for p in (self.prototxt, self.caffemodel):
            if not p.is_file():
                raise BackendUnavailable(
                    f"{model_id}: required file not found: {p}. Run scripts/fetch_third_party.sh "
                    "(no substitute output is produced under the B0 name).")
        self.weights_sha256 = _sha256(self.caffemodel)
        if expected_sha256 and expected_sha256 != self.weights_sha256:
            raise BackendUnavailable(f"{model_id}: caffemodel hash mismatch {self.weights_sha256}")
        self.model_id = model_id
        self.device = torch.device(device)
        self._weights = load_caffemodel(self.caffemodel)
        self._graphs: dict = {}
        self.shared = None  # ParameterDict shared by all graphs
        self.preprocessing_id = "cai2019_matlab_demo"
        self._graph(256, 256)

    def _graph(self, h: int, w: int) -> CaffeGraph:
        if (h, w) not in self._graphs:
            g = CaffeGraph(load_net_param(self.prototxt, w, h), self._weights).to(self.device)
            if self.shared is None:
                self.shared = g.params
                self.filled_from_filler = g.filled_from_filler
                self.unused_weights = sorted(g.unused_weights)
            else:
                g.params = self.shared
            self._graphs[(h, w)] = g
            self.add_module(f"graph_{h}x{w}", g)
        return self._graphs[(h, w)]

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """img_* : [N,1,H,W] or [N,3,H,W] float in 0..255 net intensity -> [N,2,H,W]."""
        if img_a.shape[1] == 1:   # original demo replicates gray into 3 channels
            img_a = img_a.expand(-1, 3, -1, -1)
            img_b = img_b.expand(-1, 3, -1, -1)
        h, w = img_a.shape[-2:]
        return self._graph(h, w)(img_a.contiguous(), img_b.contiguous())

    @torch.no_grad()
    def predict(self, img_a: np.ndarray, img_b: np.ndarray, bit_depth=None, batch: int = 4) -> np.ndarray:
        """img_* : [N,H,W] or [H,W] numpy -> displacement [N,2,H,W] float32."""
        a = np.asarray(img_a)
        b = np.asarray(img_b)
        single = a.ndim == 2
        if single:
            a, b = a[None], b[None]
        out = []
        self.eval()
        for i in range(0, len(a), batch):
            ta = torch.from_numpy(np.stack([to_net_intensity(x, bit_depth) for x in a[i:i + batch]])[:, None])
            tb = torch.from_numpy(np.stack([to_net_intensity(x, bit_depth) for x in b[i:i + batch]])[:, None])
            out.append(self(ta.to(self.device), tb.to(self.device)).cpu().numpy())
        out = np.concatenate(out).astype(np.float32)
        return out[0] if single else out

    def load_finetuned(self, state: dict):
        self.shared.load_state_dict(state)

    def finetuned_state(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.shared.items()}

    def info(self) -> dict:
        return {"model_id": self.model_id, "prototxt": str(self.prototxt), "caffemodel": str(self.caffemodel),
                "caffemodel_sha256": self.weights_sha256, "preprocessing_id": self.preprocessing_id,
                "implementation": "piv.models.caffe_graph (PyTorch interpreter of the Caffe deploy graph)",
                "filler_initialised_layers": self.filled_from_filler,
                "caffemodel_blobs_not_used_by_deploy": self.unused_weights,
                "demo_only": False}


class DemoBackend:
    """DEMO ONLY: scikit-image iterative Lucas-Kanade. Never reported as a baseline."""
    model_id = "DEMO_skimage_ilk"

    def __init__(self, radius: int = 7):
        self.radius = radius

    def predict(self, img_a, img_b, bit_depth=None, batch=None):
        from skimage.registration import optical_flow_ilk
        a, b = np.asarray(img_a), np.asarray(img_b)
        single = a.ndim == 2
        if single:
            a, b = a[None], b[None]
        out = []
        for x, y in zip(a, b):
            v, u = optical_flow_ilk(x.astype(np.float32), y.astype(np.float32), radius=self.radius)
            out.append(np.stack([u, v]))
        out = np.stack(out).astype(np.float32)
        return out[0] if single else out

    def info(self):
        return {"model_id": self.model_id, "demo_only": True,
                "note": "classical optical flow used only to exercise the pipeline when weights are absent"}
