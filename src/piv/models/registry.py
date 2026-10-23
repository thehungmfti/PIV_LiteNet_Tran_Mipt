"""Model registry: name -> constructed backbone from a resolved config."""
from __future__ import annotations

from pathlib import Path

import torch

from piv.models.baseline_adapter import BackendUnavailable, DemoBackend, LiteFlowNetBackend

ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def build_backbone(backend_cfg: dict, device: str = "cpu"):
    name = backend_cfg["name"]
    if name in ("B0", "B1"):
        bk = LiteFlowNetBackend(resolve(backend_cfg["prototxt"]), resolve(backend_cfg["weights_path"]),
                                backend_cfg.get("weights_sha256"), model_id=name, device=device)
        if name == "B1":
            ck = resolve(backend_cfg.get("checkpoint"))
            if ck is None or not ck.is_file():
                raise BackendUnavailable(f"B1 checkpoint missing: {ck} (train with configs/finetune.yaml)")
            state = torch.load(ck, map_location=device, weights_only=False)
            bk.load_finetuned(state["backbone"])
            bk.checkpoint = str(ck)
        return bk
    if name == "demo":
        if not backend_cfg.get("demo_only", False):
            raise ValueError("the demo backend must be declared demo_only: true")
        return DemoBackend()
    raise KeyError(f"unknown backend {name}")
