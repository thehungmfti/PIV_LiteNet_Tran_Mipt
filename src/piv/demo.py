"""One-command CPU demo: generate -> validate -> infer -> evaluate -> figures/CSV/report.

Uses B0 if the original weights are present; otherwise the DEMO backend, and every
output is then stamped demo_only (never usable as a baseline result).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from piv.cli import load_config


def run_demo(config_path: str, out_dir: str) -> dict:
    from piv.data.schema import validate_dataset
    from piv.engine.evaluate import evaluate
    from piv.engine.generate import generate
    from piv.engine.infer import infer
    from piv.io.hdf5 import read_dataset
    from piv.models.baseline_adapter import BackendUnavailable
    from piv.plotting import plot_pair_result, plot_spectra

    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg["data"]["out_dir"] = str(out / "data")
    gen = generate(cfg, workers=1)
    files = sorted((out / "data").rglob("*.h5"))
    rows, notes = [], []
    for f in files:
        d = read_dataset(f, load_images=False)
        rep = validate_dataset(d, need_calibration=True)
        if not rep.ok:
            raise RuntimeError(f"validation failed for {f}: {rep.errors}")
        icfg = dict(cfg)
        try:
            pred = out / "pred" / f"{f.stem}__B0.h5"
            infer(icfg, str(f), str(pred))
        except BackendUnavailable as e:
            notes.append(f"B0 unavailable ({e}); DEMO backend used, results flagged demo_only")
            icfg = {**cfg, "backend": {"name": "demo", "demo_only": True}}
            pred = out / "pred" / f"{f.stem}__DEMO.h5"
            infer(icfg, str(f), str(pred))
        res = evaluate(str(pred), str(f), cfg.get("evaluation", {}))
        Path(str(pred).replace(".h5", ".metrics.json")).write_text(json.dumps(res, indent=2, default=float))
        plot_pair_result(str(f), str(pred), out / "figures" / f"{pred.stem}.png", k=len(res["per_pair_epe_interior"]) // 2)
        if "skipped" not in res.get("spectral", {"skipped": 1}):
            plot_spectra(str(f), str(pred), out / "figures" / f"{pred.stem}_psd.png",
                         nperseg=int(cfg.get("evaluation", {}).get("spectral", {}).get("nperseg", 64)))
        rows.append({"file": f.name, "model": res["model"], "demo_only": res["demo_only"],
                     "epe_interior": res["interior"]["epe_mean"], "rmse_interior": res["interior"]["rmse_vec"],
                     "outlier_0.5px": res["interior"]["outlier_frac_0.5px"],
                     "band_power_rel_err": res.get("spectral", {}).get("band_power_rel_err_mean")})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "metrics.json").write_text(json.dumps({"rows": rows, "notes": notes, "generate": gen}, indent=2, default=str))
    return {"out_dir": str(out), "n_files": len(files), "rows": rows, "notes": notes}
