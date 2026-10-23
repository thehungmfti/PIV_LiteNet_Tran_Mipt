"""Inference: backbone (B0/B1/demo) and optional temporal refiner (M1/M2/M3).

Backbone fields are cached per sequence; the cache key combines the backbone
weights/checkpoint hash, the preprocessing id and the dataset file hash.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from piv.data import calibration as calmod
from piv.data import pairing
from piv.data.dataset import context_indices
from piv.engine import checkpoint as ckmod
from piv.io.hdf5 import read_dataset, read_prediction, write_prediction
from piv.io.manifest import RunManifest, sha256_file
from piv.models.baseline_adapter import to_net_intensity
from piv.models.registry import build_backbone, resolve
from piv.models.temporal_refiner import SpatialRefiner, TemporalRefiner
from piv.models.uncertainty_head import RefinerWithUQ


def backbone_key(backbone, data_path: str | Path) -> str:
    info = backbone.info()
    parts = [info["model_id"], info.get("caffemodel_sha256", ""), info.get("preprocessing_id", ""),
             getattr(backbone, "checkpoint_sha256", ""), sha256_file(data_path)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def run_backbone(backbone, d, batch: int = 8) -> np.ndarray:
    bits = int(d.attrs.get("bit_depth", 8))
    out = []
    for i in range(0, d.n_pairs, batch):
        idx = range(i, min(i + batch, d.n_pairs))
        a = np.stack([d.pair(k)[0] for k in idx])
        b = np.stack([d.pair(k)[1] for k in idx])
        out.append(backbone.predict(a, b, bit_depth=bits, batch=batch))
    return np.concatenate(out)


def cached_backbone(backbone, data_path, cache_dir: str | Path | None, d=None, batch: int = 8):
    """Returns (fields [N,2,H,W], cache_path or None, seconds)."""
    t0 = time.perf_counter()
    if cache_dir is not None:
        key = backbone_key(backbone, data_path)
        cp = Path(cache_dir) / f"{Path(data_path).stem}__{backbone.info()['model_id']}__{key}.h5"
        if cp.is_file():
            return read_prediction(cp)["displacement"], cp, 0.0
    d = d or read_dataset(data_path)
    fields = run_backbone(backbone, d, batch)
    dt = time.perf_counter() - t0
    if cache_dir is not None:
        write_prediction(cp, fields, d.t_a, d.t_b, provenance={**backbone.info(), "cache_key": key,
                                                              "data": str(data_path), "seconds": dt})
        return fields, cp, dt
    return fields, None, dt


def build_refiner(kind: str, mcfg: dict) -> torch.nn.Module:
    width = int(mcfg.get("width", 32))
    ctx = int(mcfg.get("context_fields", 5))
    base = TemporalRefiner(ctx, width) if kind in ("temporal", "M1", "M2", "M3") else None
    if kind == "spatial":
        from piv.models.temporal_refiner import matched_spatial
        return matched_spatial(TemporalRefiner(ctx, width), width)
    if kind == "M3":
        return RefinerWithUQ(base, width)
    return base


def load_refiner(path: str | Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_refiner(ck["extra"]["refiner_kind"], cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def run_refiner(model, fields: np.ndarray, d, edge: str = "mask", batch: int = 8):
    """Apply the refiner to every pair. Returns (disp [N,2,H,W], std or None, full_context [N] bool)."""
    n = len(fields)
    ctx = model.refiner.context if hasattr(model, "refiner") else model.context
    out, stds = [], []
    full = np.zeros(n, bool)
    for i in range(0, n, batch):
        ks = list(range(i, min(i + batch, n)))
        F, M, A, B = [], [], [], []
        for k in ks:
            idx, m = context_indices(k, n, ctx, edge)
            full[k] = ctx // 2 <= k < n - ctx // 2
            F.append(fields[idx] * m[:, None, None, None])
            M.append(m)
            a, b = d.pair(k)
            A.append(to_net_intensity(a, int(d.attrs.get("bit_depth", 8))))
            B.append(to_net_intensity(b, int(d.attrs.get("bit_depth", 8))))
        F = torch.from_numpy(np.stack(F)).float()
        M = torch.from_numpy(np.stack(M)).float()
        A = torch.from_numpy(np.stack(A))[:, None]
        B = torch.from_numpy(np.stack(B))[:, None]
        r = model(F, A, B, M)
        if isinstance(r, tuple):
            out.append(r[0].numpy())
            stds.append(torch.exp(0.5 * r[1]).numpy())
        else:
            out.append(r.numpy())
    std = np.concatenate(stds) if stds else None
    return np.concatenate(out), std, full


def infer(cfg: dict, input_path: str, output_path: str, device: str = "cpu") -> dict:
    run_dir = Path(output_path).parent
    man = RunManifest(run_dir, cfg, kind="infer")
    man.add_input("dataset", input_path)
    d = read_dataset(input_path)
    bcfg = cfg["backend"]
    man.tic("backbone_build")
    backbone = build_backbone(bcfg, device)
    man.toc("backbone_build")
    if bcfg.get("demo_only"):
        man.warn("DEMO backend: results must not be used as a scientific baseline")
    if getattr(backbone, "checkpoint", None):
        backbone.checkpoint_sha256 = sha256_file(backbone.checkpoint)
    man.tic("backbone")
    fields, cache, _ = cached_backbone(backbone, input_path, cfg.get("cache_dir"), d)
    man.toc("backbone")
    prov = {"backbone": backbone.info(), "dataset": str(input_path), "dataset_sha256": sha256_file(input_path),
            "run_id": cfg.get("project", {}).get("run_id", run_dir.name), "units": "pixel/pair",
            "demo_only": bool(bcfg.get("demo_only", False))}
    disp, std, full = fields, None, np.ones(len(fields), bool)
    rcfg = cfg.get("refiner")
    if rcfg and rcfg.get("checkpoint"):
        ck = resolve(rcfg["checkpoint"])
        model, meta = load_refiner(ck)
        man.tic("refiner")
        disp, std, full = run_refiner(model, fields, d, edge=rcfg.get("edge", "mask"))
        man.toc("refiner")
        prov["refiner"] = {"checkpoint": str(ck), "sha256": sha256_file(ck), "kind": meta["extra"]["refiner_kind"],
                           "mode": "offline centred", "lookahead_samples": model_context(model) // 2,
                           "edge_handling": rcfg.get("edge", "mask")}
        if std is not None and rcfg.get("variance_scale"):
            std = std * float(rcfg["variance_scale"])
            prov["refiner"]["variance_scale"] = float(rcfg["variance_scale"])
    velocity = None
    if cfg.get("export", {}).get("velocity", False):
        calmod.require(d.calibration)           # refuses m/s export without calibration
        velocity = calmod.displacement_to_velocity(disp, pairing.dt_pair(d.t_a, d.t_b), d.calibration)
    prov["full_context_pairs"] = full.astype(int).tolist()
    prov["timing_s"] = man.data["timing_s"]
    write_prediction(output_path, disp, d.t_a, d.t_b, valid=d.valid, std=std, velocity=velocity, provenance=prov)
    man.add_output("prediction", output_path)
    man.save(Path(output_path).stem + ".manifest.json")
    return prov


def model_context(model) -> int:
    return model.refiner.context if hasattr(model, "refiner") else model.context
