"""Benchmark orchestration: predictions, evaluation and aggregated tables.

python scripts/benchmark.py predict  --method B0 --splits test ood_flow ood_imaging
python scripts/benchmark.py evaluate --methods B0 B1 M1 S1 M2 M3 PIVlab OpenPIV --splits test ...
python scripts/benchmark.py aggregate --methods ... --ref B1

Dense methods are scored on pixels (interior, full-context pairs) AND sampled at the
common cross-correlation vector grid (x, y = 8, 16, ..., W-8; 0-based), where all
methods including PIVlab/OpenPIV are compared against the exact Lagrangian GT.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from piv.cli import load_config, deep_merge  # noqa: E402
from piv.data.split import read_manifest  # noqa: E402
from piv.engine.evaluate import evaluate_dense, evaluate_grid, gt_on_points, flow_of, sample_dense  # noqa: E402
from piv.io.hdf5 import read_dataset, read_prediction  # noqa: E402
from piv.metrics.stats import block_bootstrap_ci, paired_difference_ci  # noqa: E402

OUT = ROOT / "outputs" / "benchmark"
DATA = ROOT / "data" / "research"
DENSE = {
    "B0": {"config": "configs/baseline.yaml"},
    "B1": {"config": "configs/baseline.yaml", "backend": {"name": "B1", "checkpoint": "outputs/train_B1/B1.pt"}},
    "M1": {"config": "configs/temporal.yaml", "refiner": "outputs/train_M1/best.pt"},
    "S1": {"config": "configs/temporal.yaml", "refiner": "outputs/train_S1/best.pt"},
    "M2": {"config": "configs/temporal.yaml", "refiner": "outputs/train_M2/best.pt"},
    "M3": {"config": "configs/temporal.yaml", "refiner": "outputs/train_M3/best.pt", "uq": True},
}
GRID_METHODS = ("PIVlab", "OpenPIV")


def files_for(splits):
    man = read_manifest(DATA / "split_manifest.json")
    return [DATA / f for s in splits for f in sorted(man["by_split"].get(s, []))]


def method_cfg(m: str) -> dict:
    spec = DENSE[m]
    cfg = load_config(str(ROOT / spec["config"]))
    if "backend" in spec:
        cfg = deep_merge(cfg, {"backend": spec["backend"]})
    if "refiner" in spec:
        cfg["refiner"] = {"checkpoint": spec["refiner"], "edge": "mask"}
        if spec.get("uq"):
            rec = ROOT / Path(spec["refiner"]).parent / "recalibration.json"
            if rec.is_file():
                cfg["refiner"]["variance_scale"] = json.loads(rec.read_text())["variance_scale"]
    cfg["cache_dir"] = str(ROOT / "outputs" / "cache")
    return cfg


def predict(method: str, splits, overwrite=False, pivlab_settings=None, tag=None):
    name = tag or method
    for f in files_for(splits):
        out = OUT / "pred" / name / f"{f.stem}.h5"
        if out.is_file() and not overwrite:
            continue
        t0 = time.perf_counter()
        if method in DENSE:
            from piv.engine.infer import infer
            infer(method_cfg(method), str(f), str(out))
        elif method == "PIVlab":
            from comparators import pivlab_runner
            pivlab_runner.run(str(f), str(out), str(ROOT / "third_party" / "PIVlab"), pivlab_settings)
        elif method == "OpenPIV":
            from comparators import openpiv_runner
            openpiv_runner.run(str(f), str(out))
        else:
            raise KeyError(method)
        print(f"[{name}] {f.stem}: {time.perf_counter() - t0:.1f}s", flush=True)


def common_grid(h, w, step=8, first=8):
    return np.arange(first, w - first + 1, step, dtype=float), np.arange(first, h - first + 1, step, dtype=float)


def gt_grid_cached(d, f: Path, xs, ys):
    p = OUT / "gt_grid" / f"{f.stem}.npy"
    if p.is_file():
        return np.load(p)
    flow, _ = flow_of(d)
    g = gt_on_points(flow, xs, ys, d.t_a, d.t_b)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, g)
    return g


def evaluate_all(methods, splits, ecfg, force=False):
    for f in files_for(splits):
        d = read_dataset(f)
        h, w = d.shape
        xs, ys = common_grid(h, w)
        gtg = None
        for m in methods:
            pp = OUT / "pred" / m / f"{f.stem}.h5"
            mp = OUT / "metrics" / m / f"{f.stem}.json"
            if not pp.is_file() or (not force and mp.is_file() and mp.stat().st_mtime > pp.stat().st_mtime):
                continue
            pred = read_prediction(pp)
            res = {"run_id": d.attrs.get("run_id"), "noise": d.attrs.get("noise"), "split": d.attrs.get("split"),
                   "flow_type": d.attrs.get("flow_type"), "method": m,
                   "seconds_per_pair": pred["provenance"].get("seconds_per_pair") or
                   (sum(pred["provenance"].get("timing_s", {}).get(k, 0) for k in ("backbone", "refiner")) / d.n_pairs)}
            if "grid_x" in pred:
                if not (np.allclose(pred["grid_x"], xs) and np.allclose(pred["grid_y"], ys)):
                    res["grid_warning"] = f"method grid differs from common grid: {pred['grid_x'][:3]}"
                    gx, gy = pred["grid_x"], pred["grid_y"]
                    r, _ = evaluate_grid(pred["displacement"], gx, gy, d, ecfg)
                else:
                    gtg = gt_grid_cached(d, f, xs, ys) if gtg is None else gtg
                    r, _ = evaluate_grid(pred["displacement"], xs, ys, d, ecfg, gtg)
                res.update(r)
            else:
                res.update(evaluate_dense(pred, d, ecfg))
                gtg = gt_grid_cached(d, f, xs, ys) if gtg is None else gtg
                r, _ = evaluate_grid(sample_dense(pred["displacement"], xs, ys), xs, ys, d, ecfg, gtg)
                res.update(r)
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps(res, default=float))
            print(f"eval {m} {f.stem}", flush=True)


METRICS = [
    ("dense_epe", lambda r: r.get("interior_full_context", {}).get("epe_mean")),
    ("dense_rmse", lambda r: r.get("interior_full_context", {}).get("rmse_vec")),
    ("dense_outlier_0.5px", lambda r: r.get("interior_full_context", {}).get("outlier_frac_0.5px")),
    ("dense_vort_rmse", lambda r: r.get("gradients_interior", {}).get("rmse_vorticity")),
    ("dense_bandpower_relerr", lambda r: r.get("spectral", {}).get("band_power_rel_err_median")),
    ("dense_bandpower_ratio", lambda r: r.get("spectral", {}).get("band_power_ratio_global")),
    ("dense_logpsd_l1", lambda r: r.get("spectral", {}).get("log_psd_l1_band")),
    ("dense_peak_match", lambda r: r.get("spectral", {}).get("peak_freq_match_frac")),
    ("dense_coh_fref", lambda r: r.get("spectral", {}).get("coherence_at_fref_mean")),
    ("dense_phase_err_deg", lambda r: r.get("spectral", {}).get("phase_err_at_fref_deg_mean")),
    ("grid_epe", lambda r: r.get("grid", {}).get("epe_mean")),
    ("grid_rmse", lambda r: r.get("grid", {}).get("rmse_vec")),
    ("grid_bias_dx", lambda r: r.get("grid", {}).get("bias_dx")),
    ("grid_outlier_0.5px", lambda r: r.get("grid", {}).get("outlier_frac_0.5px")),
    ("grid_vort_rmse", lambda r: r.get("gradients_grid", {}).get("rmse_vorticity")),
    ("grid_bandpower_relerr", lambda r: r.get("spectral_grid", {}).get("band_power_rel_err_median")),
    ("grid_bandpower_ratio", lambda r: r.get("spectral_grid", {}).get("band_power_ratio_global")),
    ("grid_logpsd_l1", lambda r: r.get("spectral_grid", {}).get("log_psd_l1_band")),
    ("grid_peak_match", lambda r: r.get("spectral_grid", {}).get("peak_freq_match_frac")),
    ("grid_coh_fref", lambda r: r.get("spectral_grid", {}).get("coherence_at_fref_mean")),
    ("sec_per_pair", lambda r: r.get("seconds_per_pair")),
]


def load_metrics(methods, splits):
    rows = {}
    for m in methods:
        for p in sorted((OUT / "metrics" / m).glob("*.json")):
            r = json.loads(p.read_text())
            if r.get("split") in splits:
                rows.setdefault(m, {})[r["run_id"]] = r
    return rows


def per_frame(r, key):
    k = "per_pair_epe_interior" if key == "dense" else "per_pair_epe_grid"
    if k not in r:
        return None
    x = np.asarray(r[k], float)
    return x[2:-2]          # full-context range for every method (refiner edges excluded)


def aggregate(methods, splits, ref: str | None, tag: str):
    rows = load_metrics(methods, splits)
    out_rows = []
    groups = {}
    for m, rr in rows.items():
        for run, r in rr.items():
            groups.setdefault((r["split"], r["noise"]), {}).setdefault(m, {})[run] = r
    for (split, noise), by_m in sorted(groups.items()):
        for m in methods:
            if m not in by_m:
                continue
            rr = by_m[m]
            row = {"split": split, "noise": noise, "method": m, "n_runs": len(rr)}
            for name, fn in METRICS:
                vals = [fn(r) for r in rr.values() if fn(r) is not None]
                row[name] = float(np.mean(vals)) if vals else None
            for key in ("dense", "grid"):
                pf = {run: per_frame(r, key) for run, r in rr.items() if per_frame(r, key) is not None}
                if pf:
                    ci = block_bootstrap_ci(pf, block=16, n_boot=1000)
                    row[f"{key}_epe_ci"] = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
                    if ref and ref in by_m and m != ref:
                        pr = {run: per_frame(by_m[ref][run], key) for run in pf if run in by_m[ref]
                              and per_frame(by_m[ref][run], key) is not None}
                        if pr and all(len(pr[k]) == len(pf[k]) for k in pr):
                            dci = paired_difference_ci({k: pf[k] for k in pr}, pr, block=16, n_boot=1000)
                            row[f"{key}_epe_minus_{ref}"] = f"{dci['mean']:+.4f} [{dci['ci_low']:+.4f}, {dci['ci_high']:+.4f}]"
            out_rows.append(row)
    keys = sorted({k for r in out_rows for k in r}, key=lambda k: (["split", "noise", "method", "n_runs"] + [n for n, _ in METRICS]).index(k)
                  if k in ["split", "noise", "method", "n_runs"] + [n for n, _ in METRICS] else 999)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"summary_{tag}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(out_rows)
    (OUT / f"summary_{tag}.json").write_text(json.dumps(out_rows, indent=2, default=float))
    print(f"wrote {OUT / f'summary_{tag}.csv'} ({len(out_rows)} rows)")
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict"); p.add_argument("--method", required=True); p.add_argument("--splits", nargs="+", required=True)
    p.add_argument("--overwrite", action="store_true"); p.add_argument("--tag")
    e = sub.add_parser("evaluate"); e.add_argument("--methods", nargs="+", required=True); e.add_argument("--splits", nargs="+", required=True)
    e.add_argument("--force", action="store_true")
    a = sub.add_parser("aggregate"); a.add_argument("--methods", nargs="+", required=True); a.add_argument("--splits", nargs="+", required=True)
    a.add_argument("--ref"); a.add_argument("--tag", default="main")
    args = ap.parse_args()
    ecfg = load_config(str(ROOT / "configs" / "eval.yaml"))["evaluation"]
    if args.cmd == "predict":
        predict(args.method, args.splits, args.overwrite, tag=args.tag)
    elif args.cmd == "evaluate":
        evaluate_all(args.methods, args.splits, ecfg, args.force)
    else:
        aggregate(args.methods, args.splits, args.ref, args.tag)


if __name__ == "__main__":
    main()
