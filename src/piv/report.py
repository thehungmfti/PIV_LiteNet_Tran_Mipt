"""Aggregate *.metrics.json under a run directory into CSV + Markdown."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def _flat(res: dict) -> dict:
    row = {"run_id": res.get("run_id"), "model": res.get("model"), "noise": res.get("noise"),
           "split": res.get("split"), "flow": res.get("flow_type"), "demo_only": res.get("demo_only")}
    for sect in ("interior_full_context", "grid"):
        if sect in res and res[sect]:
            for k in ("epe_mean", "rmse_vec", "bias_dx", "bias_dy", "outlier_frac_0.5px", "outlier_frac_1px"):
                row[f"{sect}.{k}"] = res[sect].get(k)
    for sect in ("spectral", "spectral_grid"):
        if sect in res and "skipped" not in res[sect]:
            for k in ("band_power_rel_err_mean", "peak_freq_match_frac", "coherence_at_fref_mean",
                      "phase_err_at_fref_deg_mean"):
                row[f"{sect}.{k}"] = res[sect].get(k)
    return row


def make_report(run_dir: Path) -> str:
    rows = [_flat(json.loads(p.read_text())) for p in sorted(run_dir.rglob("*.metrics.json"))]
    if not rows:
        return f"no *.metrics.json under {run_dir}"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k.count("."), k))
    with open(run_dir / "report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r.get(k):.4g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                                       for k in keys) + " |")
    md = "\n".join(lines)
    (run_dir / "report.md").write_text(md + "\n")
    return md
