"""Checkpoints with full resume state: model, optimizer, scheduler, scaler, step, RNGs, sampler."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch


def rng_state(sampler_rng: np.random.Generator | None = None) -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "sampler": sampler_rng.bit_generator.state if sampler_rng is not None else None}


def set_rng_state(st: dict, sampler_rng: np.random.Generator | None = None):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if sampler_rng is not None and st.get("sampler") is not None:
        sampler_rng.bit_generator.state = st["sampler"]


def save(path: str | Path, *, model_state: dict, optimizer=None, scheduler=None, scaler=None, step: int = 0,
         epoch: int = 0, sampler_rng=None, config: dict | None = None, extra: dict | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model_state, "step": step, "epoch": epoch, "rng": rng_state(sampler_rng),
               "optimizer": optimizer.state_dict() if optimizer else None,
               "scheduler": scheduler.state_dict() if scheduler else None,
               "scaler": scaler.state_dict() if scaler else None,
               "config": config, "extra": extra or {}}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return file_hash(path)


def load(path: str | Path, *, model=None, optimizer=None, scheduler=None, scaler=None, sampler_rng=None,
         restore_rng: bool = True) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if restore_rng:
        set_rng_state(ck["rng"], sampler_rng)
    return ck


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
