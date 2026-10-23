"""Training stages (spec section 10).

S1 'B1'      : fine-tune the PIV-LiteFlowNet-en graph on project train data (final-output
               Charbonnier EPE; the original multi-level loss is not reproduced -> documented).
S2 'M1'      : frozen B1, cached fields, temporal residual refiner (L_disp + L_grad).
   'spatial' : parameter-matched spatial-only ablation, same losses / budget.
S3 'M2'      : M1 + spectral loss on long segments (initialised from M1).
S4 'M3'      : M2 frozen + uncertainty head (head-only, Gaussian NLL).
Early stopping: best validation score, score declared in the config before training.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from piv.data.dataset import PairSampler, SegmentSampler, SequenceStore, WindowSampler
from piv.data.split import read_manifest
from piv.engine import checkpoint as ckmod
from piv.engine.infer import build_refiner, cached_backbone
from piv.io.manifest import RunManifest, sha256_file
from piv.losses.spectral import spectral_loss
from piv.losses.supervised import charbonnier, gradient_loss, masked_mean
from piv.losses.uncertainty import gaussian_nll
from piv.models.registry import build_backbone, resolve


def seed_all(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def split_files(cfg: dict, split: str) -> list[str]:
    dc = cfg["data"]
    root = resolve(dc["root"])
    man = read_manifest(root / "split_manifest.json")
    files = sorted(man["by_split"].get(split, []))
    noise = dc.get("noise_filter")
    if noise:
        files = [f for f in files if any(f.endswith(f"__{n}.h5") for n in noise)]
    lim = dc.get(f"max_files_{split}")
    if lim:
        files = files[:lim]
    return [str(root / f) for f in files]


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a")

    def log(self, **kw):
        kw["time"] = time.time()
        self.f.write(json.dumps(kw, default=float) + "\n")
        self.f.flush()


def _epe(pred, gt, mask):
    return masked_mean(torch.sqrt(((pred - torch.nan_to_num(gt)) ** 2).sum(1)), mask).item()


# --------------------------------------------------------------------------- B1
def train_b1(cfg: dict, run_dir: Path, man: RunManifest, log: Logger):
    tc = cfg["training"]
    backbone = build_backbone({**cfg["backend"], "name": "B0"})
    for p in backbone.shared.values():
        p.requires_grad_(True)
    train = SequenceStore(split_files(cfg, "train"))
    val = SequenceStore(split_files(cfg, "val"))
    rng = np.random.default_rng(cfg["project"]["seed"])
    sampler = PairSampler(train, rng)
    vrng = np.random.default_rng(12345)          # fixed validation subset
    vbatches = [PairSampler(val, vrng, aug=False).batch(4) for _ in range(int(tc.get("val_batches", 12)))]
    opt = torch.optim.Adam(backbone.shared.values(), lr=float(tc["lr"]), weight_decay=float(tc.get("wd", 0.0)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tc["iters"]))
    step, best = 0, math.inf
    ck_last = run_dir / "last.pt"
    if tc.get("resume") and ck_last.is_file():
        ck = ckmod.load(ck_last, optimizer=opt, scheduler=sched, sampler_rng=rng)
        backbone.load_finetuned(ck["model"])
        step, best = ck["step"], ck["extra"]["best"]

    def evaluate():
        backbone.eval()
        with torch.no_grad():
            e = [_epe(backbone(a, b), g, v) for a, b, g, v in vbatches]
        backbone.train()
        return float(np.mean(e))

    if step == 0:
        v0 = evaluate()
        log.log(step=0, val_epe=v0, note="B0 weights before fine-tuning")
        best = v0
        ckmod.save(run_dir / "best.pt", model_state=backbone.finetuned_state(), step=0, config=cfg,
                   extra={"best": best, "stage": "B1", "note": "initial = B0"})
    backbone.train()
    eps = float(cfg["losses"].get("eps", 0.01))
    while step < int(tc["iters"]):
        a, b, g, v = sampler.batch(int(tc["batch_size"]))
        pred = backbone(a, b)
        loss = charbonnier(pred, g, v, eps)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(backbone.shared.values(), float(tc.get("clip", 10.0)))
        if not torch.isfinite(loss) or not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite loss/grad at step {step}")
        opt.step()
        sched.step()
        step += 1
        log.log(step=step, loss_disp=loss.item(), grad_norm=gn.item(), lr=sched.get_last_lr()[0])
        if step % int(tc.get("val_every", 50)) == 0 or step == int(tc["iters"]):
            ve = evaluate()
            log.log(step=step, val_epe=ve)
            if ve < best:
                best = ve
                ckmod.save(run_dir / "best.pt", model_state=backbone.finetuned_state(), step=step, config=cfg,
                           extra={"best": best, "stage": "B1"})
            ckmod.save(ck_last, model_state=backbone.finetuned_state(), optimizer=opt, scheduler=sched, step=step,
                       sampler_rng=rng, config=cfg, extra={"best": best, "stage": "B1"})
    # B1 checkpoint consumed by the registry
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    torch.save({"backbone": ck["model"], "step": ck["step"], "val_epe": ck["extra"]["best"], "config": cfg},
               run_dir / "B1.pt")
    man.add_output("B1_checkpoint", run_dir / "B1.pt")
    return {"best_val_epe": best, "best_step": ck["step"]}


# --------------------------------------------------------------------------- refiner stages
def load_fields(cfg: dict, files: list[str], man: RunManifest) -> dict:
    bcfg = cfg["backend"]
    backbone = build_backbone(bcfg)
    if getattr(backbone, "checkpoint", None):
        backbone.checkpoint_sha256 = sha256_file(backbone.checkpoint)
    out = {}
    for f in files:
        fields, cp, dt = cached_backbone(backbone, f, resolve(cfg["cache_dir"]))
        out[f] = fields
        man.data["inputs"].setdefault("cache", []).append(str(cp))
    return out


def refiner_step_losses(model, batch, cfg, uq: bool = False):
    F, M, A, B, G, V = batch
    lc = cfg["losses"]
    eps = float(lc.get("eps", 0.01))
    if uq:
        pred, s = model(F, A, B, M)
        nll = gaussian_nll(pred, s, G, V, float(lc.get("s_min", -9)), float(lc.get("s_max", 5)))
        return {"nll": nll}, pred
    pred = model(F, A, B, M)
    out = {"disp": charbonnier(pred, G, V, eps)}
    if float(lc.get("w_grad", 0)) > 0:
        out["grad"] = gradient_loss(pred, G, V, eps)
    return out, pred


def weighted(parts: dict, lc: dict) -> torch.Tensor:
    return sum(float(lc.get(f"w_{k}", 1.0)) * v for k, v in parts.items())


def train_refiner(cfg: dict, run_dir: Path, man: RunManifest, log: Logger, kind: str):
    tc, lc, sc = cfg["training"], cfg["losses"], cfg.get("spectral", {})
    tr_files, va_files = split_files(cfg, "train"), split_files(cfg, "val")
    fields = load_fields(cfg, tr_files + va_files, man)
    train = SequenceStore(tr_files, fields)
    val = SequenceStore(va_files, fields)
    rng = np.random.default_rng(cfg["project"]["seed"])
    model = build_refiner(kind, cfg["model"])
    init = cfg["model"].get("init_from")
    if init:
        ck0 = torch.load(resolve(init), map_location="cpu", weights_only=False)
        target = model.refiner if kind == "M3" else model
        target.load_state_dict(ck0["model"])
        man.add_input("init_from", resolve(init))
    uq = kind == "M3"
    if uq and cfg["model"].get("head_only", True):
        for p in model.refiner.parameters():
            p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    man.set("n_trainable_params", int(sum(p.numel() for p in params)))
    opt = torch.optim.Adam(params, lr=float(tc["lr"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tc["iters"]))
    wsamp = WindowSampler(train, rng, int(cfg["model"].get("context_fields", 5)), float(tc.get("p_edge", 0.1)),
                          crop=tc.get("crop"))
    ssamp = SegmentSampler(train, rng, int(sc.get("segment_length", 128)), crop=int(sc.get("crop", 32))) \
        if float(lc.get("w_spec", 0)) > 0 else None
    vrng = np.random.default_rng(777)
    vw = WindowSampler(val, vrng, int(cfg["model"].get("context_fields", 5)), 0.0, aug=False)
    vbatches = [vw.batch(8) for _ in range(int(tc.get("val_batches", 10)))]
    vseg = [SegmentSampler(val, vrng, int(sc.get("segment_length", 128)), crop=int(sc.get("crop", 32))).segment()
            for _ in range(int(tc.get("val_segments", 4)))] if ssamp else []
    fs = float(sc.get("fs", 1000.0))
    probe_idx = None
    step, best = 0, math.inf
    ck_last = run_dir / "last.pt"
    if tc.get("resume") and ck_last.is_file():
        ck = ckmod.load(ck_last, model=model, optimizer=opt, scheduler=sched, sampler_rng=rng)
        step, best = ck["step"], ck["extra"]["best"]

    def spec_loss_of(seg):
        F, M, A, B, G, V = seg
        nonlocal probe_idx
        pred = model(F, A, B, M)                                   # [N,2,h,w] consecutive centres
        n, _, h, w = pred.shape
        if probe_idx is None:
            prng = np.random.default_rng(int(sc.get("probe_seed", 0)))
            probe_idx = torch.from_numpy(prng.choice(h * w, int(sc.get("n_probes", 64)), replace=False))
        pp = pred.reshape(n, 2, h * w)[:, :, probe_idx].permute(2, 1, 0)[None]   # [1,P,2,N]
        gp = torch.nan_to_num(G).reshape(n, 2, h * w)[:, :, probe_idx].permute(2, 1, 0)[None]
        ls = spectral_loss(pp, gp, fs, float(sc["fmin"]), float(sc["fmax"]), sc.get("window", "hann"))
        sup = {"disp": charbonnier(pred, G, V, float(lc.get("eps", 0.01)))}
        if float(lc.get("w_grad", 0)) > 0:
            sup["grad"] = gradient_loss(pred, G, V, float(lc.get("eps", 0.01)))
        return ls, sup, pred

    def evaluate():
        model.eval()
        with torch.no_grad():
            if uq:
                v = float(np.mean([refiner_step_losses(model, b, cfg, True)[0]["nll"].item() for b in vbatches]))
                res = {"val_nll": v, "score": v}
            else:
                e = [_epe(model(F, A, B, M), G, V) for F, M, A, B, G, V in vbatches]
                res = {"val_epe": float(np.mean(e))}
                if vseg:
                    res["val_spec"] = float(np.mean([spec_loss_of(s)[0].item() for s in vseg]))
                # declared validation score: EPE (+ w_spec_val * spectral term for M2)
                res["score"] = res["val_epe"] + float(tc.get("w_spec_val", 0.0)) * res.get("val_spec", 0.0)
        model.train()
        return res

    if step == 0:
        r0 = evaluate()
        log.log(step=0, **r0, note="initial (identity residual / init_from)")
        best = r0["score"]
        ckmod.save(run_dir / "best.pt", model_state=(model.state_dict()), step=0, config=cfg,
                   extra={"best": best, "refiner_kind": kind, "val": r0})
    model.train()
    while step < int(tc["iters"]):
        parts, _ = refiner_step_losses(model, wsamp.batch(int(tc["batch_size"])), cfg, uq)
        if ssamp is not None and step % int(tc.get("spec_every", 1)) == 0:
            ls, sup, _ = spec_loss_of(ssamp.segment())
            parts["spec"] = ls
            for k, v in sup.items():
                parts[f"seg_{k}"] = v
        loss = weighted(parts, lc)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, float(tc.get("clip", 5.0)))
        if not torch.isfinite(loss) or not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite loss/grad at step {step}")
        opt.step()
        sched.step()
        step += 1
        log.log(step=step, loss=loss.item(), grad_norm=gn.item(),
                **{f"raw_{k}": v.item() for k, v in parts.items()},
                **{f"weighted_{k}": float(lc.get(f"w_{k}", 1.0)) * v.item() for k, v in parts.items()})
        if step % int(tc.get("val_every", 100)) == 0 or step == int(tc["iters"]):
            r = evaluate()
            log.log(step=step, **r)
            if r["score"] < best:
                best = r["score"]
                ckmod.save(run_dir / "best.pt", model_state=model.state_dict(), step=step, config=cfg,
                           extra={"best": best, "refiner_kind": kind, "val": r})
            ckmod.save(ck_last, model_state=model.state_dict(), optimizer=opt, scheduler=sched, step=step,
                       sampler_rng=rng, config=cfg, extra={"best": best, "refiner_kind": kind})
    if kind == "M3":
        fit_recalibration(model, val, cfg, run_dir)
    return {"best_score": best}


def fit_recalibration(model, val_store, cfg, run_dir: Path):
    """Scalar variance recalibration on the VALIDATION split (never test)."""
    from piv.metrics.calibration import fit_variance_scale
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    vw = WindowSampler(val_store, np.random.default_rng(99), int(cfg["model"].get("context_fields", 5)), 0.0, aug=False)
    errs, stds = [], []
    with torch.no_grad():
        for _ in range(20):
            F, M, A, B, G, V = vw.batch(8)
            p, s = model(F, A, B, M)
            m = V[:, None].expand_as(p)
            errs.append((p - G)[m].numpy())
            stds.append(torch.exp(0.5 * s)[m].numpy())
    k = fit_variance_scale(np.concatenate(errs), np.concatenate(stds))
    ck["extra"]["variance_scale_val"] = k
    torch.save(ck, run_dir / "best.pt")
    (run_dir / "recalibration.json").write_text(json.dumps({"variance_scale": k, "fitted_on": "val"}))


def train(cfg: dict) -> dict:
    run_dir = resolve(cfg["project"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_all(int(cfg["project"]["seed"]))
    man = RunManifest(run_dir, cfg, kind=f"train_{cfg['training']['stage']}")
    man.add_input("split_manifest", resolve(cfg["data"]["root"]) / "split_manifest.json")
    log = Logger(run_dir / "train_log.jsonl")
    stage = cfg["training"]["stage"]
    t0 = time.perf_counter()
    if stage == "B1":
        res = train_b1(cfg, run_dir, man, log)
    elif stage in ("M1", "spatial", "M2", "M3"):
        res = train_refiner(cfg, run_dir, man, log, {"M1": "temporal", "M2": "temporal"}.get(stage, stage))
    else:
        raise KeyError(stage)
    man.data["timing_s"]["train_total"] = time.perf_counter() - t0
    man.set("result", res)
    if (run_dir / "best.pt").is_file():
        man.set("best_checkpoint_sha256", sha256_file(run_dir / "best.pt"))
    man.save()
    return res
