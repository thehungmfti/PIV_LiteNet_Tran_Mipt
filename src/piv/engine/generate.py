"""Config-driven synthetic dataset generation (tier A).

One *parent* = one flow realisation + one particle realisation. Every noise
variant of a parent is written to its own schema-v1 file and inherits the
parent's split, so variants never leak across splits.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from piv.data import degradation as dg
from piv.data import split as splitmod
from piv.data.synthetic import ParticleConfig, SequenceSpec, make_flow, render_sequence
from piv.io.hdf5 import PIVDataset, write_dataset


def sample_flow(kind: str, rng: np.random.Generator, h: int, w: int, fs: float) -> dict:
    """Random flow parameters; speeds are drawn in pixel/pair and converted with fs."""
    if kind == "vortex_street":
        U = rng.uniform(1.5, 3.2) * fs
        r_c = rng.uniform(5.0, 9.0)
        return dict(type=kind, U_inf=U, U_def=rng.uniform(0.1, 0.3) * U, sigma_w=rng.uniform(12, 24),
                    y_c=h / 2 + rng.uniform(-10, 10), h=rng.uniform(20, 36), Uc=rng.uniform(0.75, 0.9) * U,
                    f_shed=rng.uniform(50, 140), gamma=rng.uniform(0.3, 0.6) * U * 2 * math.pi * r_c * 1.4,
                    r_c=r_c, x_s=-160.0, x_end=w + 140.0, ramp=60.0)
    if kind == "cellular":
        L = rng.uniform(48, 128)
        speed = rng.uniform(1.0, 3.0) * fs
        f1, f2 = rng.uniform(30, 120), rng.uniform(120, 220)
        return dict(type=kind, A0=speed / (2 * math.pi / L), Lx=L, Ly=rng.uniform(48, 128),
                    amps=[rng.uniform(0.2, 0.5), rng.uniform(0.1, 0.3)], freqs=[f1, f2],
                    phases=rng.uniform(0, 2 * math.pi, 2).tolist(),
                    U=rng.uniform(-1, 1) * fs, V=rng.uniform(-1, 1) * fs)
    if kind == "orbiting_vortices":
        n = int(rng.integers(3, 5))
        r_c = rng.uniform(6, 10)
        return dict(type=kind, centers=rng.uniform(0.2, 0.8, (n, 2)) * [w, h], radii=rng.uniform(6, 16, n).tolist(),
                    freqs=rng.uniform(30, 180, n).tolist(),
                    gammas=(rng.choice([-1, 1], n) * rng.uniform(0.8, 2.0, n) * fs * 2 * math.pi * r_c).tolist(),
                    r_c=r_c, U=rng.uniform(-1.5, 1.5) * fs, V=rng.uniform(-1.0, 1.0) * fs)
    if kind == "uniform":
        return dict(type=kind, u=rng.uniform(-3, 3) * fs, v=rng.uniform(-3, 3) * fs)
    if kind == "shear":
        return dict(type=kind, gamma=rng.uniform(0.01, 0.04) * fs, y0=h / 2)
    raise KeyError(kind)


def _noise_name(noise) -> str:
    return noise if isinstance(noise, str) else noise.get("name", "custom")


def _job(job: dict) -> list[str]:
    expected = [str(Path(job["out_dir"]) / job["split"] / f"{job['parent']}__{_noise_name(n)}.h5") for n in job["noise"]]
    if not job.get("overwrite") and all(Path(p).is_file() for p in expected):
        return expected          # deterministic given the seed: reuse
    rng = np.random.default_rng(job["seed"])
    spec_flow = job.get("flow_spec") or sample_flow(job["flow"], rng, job["h"], job["w"], job["fs"])
    flow = make_flow({k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in spec_flow.items()})
    pc = replace(ParticleConfig(), **job.get("particles", {}))
    spec = SequenceSpec(h=job["h"], w=job["w"], n_frames=job["n_frames"], fs=job["fs"], mode=job["mode"],
                        dt_pair=job.get("dt_pair"), particles=pc)
    r = render_sequence(flow, spec, seed=job["seed"] + 1)
    written = []
    for noise in job["noise"]:
        deg = dg.PRESETS[noise] if isinstance(noise, str) else \
            dg.Degradation(**{k: v for k, v in noise.items() if k != "name"})
        nname = _noise_name(noise)
        nseed = job["seed"] + 7919 * (1 + list(dg.PRESETS).index(nname) if nname in dg.PRESETS else 99)
        if spec.mode == "continuous":
            t_frames = np.concatenate([r["t_a"], r["t_b"][-1:]])
            images = dg.apply_sequence(r["images"], t_frames, deg, nseed)
            kw = dict(images=images)
        else:
            ia = dg.apply_sequence(r["images"][0], r["t_a"], deg, nseed)
            ib = dg.apply_sequence(r["images"][1], r["t_b"], deg, nseed + 1)
            kw = dict(images_a=ia, images_b=ib)
        n = len(r["t_a"])
        run_id = f"{job['parent']}__{nname}"
        d = PIVDataset(pairs_a=r["index_a"], pairs_b=r["index_b"], t_a=r["t_a"], t_b=r["t_b"],
                       gt_displacement=r["gt"], valid=np.ones((n, job["h"], job["w"]), bool), **kw,
                       calibration=job["calibration"],
                       attrs={"case_id": job["parent"], "run_id": run_id, "acquisition_mode": spec.mode,
                              "split": job["split"], "ood": job.get("ood", "none"), "flow_type": spec_flow["type"],
                              "noise": nname, "bit_depth": deg.bit_depth, "black_level": deg.black_level,
                              "white_level": 2 ** deg.bit_depth - 1, "fs_frames_hz": job["fs"],
                              "gt_convention": "forward Lagrangian displacement of the particle at the pixel "
                                               "centre at t_a (RK4, step-doubling converged)",
                              "tier": "A (analytic)", "demo": bool(job.get("demo", False))},
                       extra={"flow_json": np.bytes_(json.dumps(spec_flow, default=lambda o: np.asarray(o).tolist())),
                              "particles_json": np.bytes_(json.dumps(asdict(pc))),
                              "degradation_json": np.bytes_(json.dumps(deg.as_dict())),
                              "rk4_json": np.bytes_(json.dumps(r["meta"]))})
        path = Path(job["out_dir"]) / job["split"] / f"{run_id}.h5"
        write_dataset(path, d)
        written.append(str(path))
    return written


def build_jobs(cfg: dict) -> list[dict]:
    dc = cfg["data"]
    base_seed = int(cfg["project"]["seed"])
    jobs = []
    for gi, g in enumerate(dc["groups"]):
        for i in range(int(g["n_parents"])):
            parent = f"{g['split']}_{g['flow']}_{gi:02d}_{i:03d}"
            seed = base_seed * 1_000_003 + gi * 10_007 + i * 101
            jobs.append(dict(parent=parent, split=g["split"], ood=g.get("ood", "none"), flow=g["flow"],
                             flow_spec=g.get("flow_spec"), seed=seed, h=dc["h"], w=dc["w"], fs=dc["fs"],
                             mode=g.get("mode", dc.get("mode", "continuous")), dt_pair=g.get("dt_pair"),
                             n_frames=int(g["n_frames"]), noise=g["noise"],
                             particles={**dc.get("particles", {}), **g.get("particles", {})},
                             calibration=dc.get("calibration", {}), out_dir=dc["out_dir"],
                             demo=cfg.get("backend", {}).get("demo_only", False)))
    return jobs


def generate(cfg: dict, workers: int = 4) -> dict:
    jobs = build_jobs(cfg)
    out_dir = Path(cfg["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_job, jobs))
    else:
        results = [_job(j) for j in jobs]
    for j, paths in zip(jobs, results):
        for p in paths:
            files[str(Path(p).relative_to(out_dir))] = j["parent"]
    mapping = {j["parent"]: j["split"] for j in jobs}
    digest = splitmod.write_manifest(out_dir / "split_manifest.json", mapping, files, int(cfg["project"]["seed"]))
    splitmod.read_manifest(out_dir / "split_manifest.json")   # leakage check
    return {"n_files": len(files), "manifest_sha256": digest, "out_dir": str(out_dir)}
