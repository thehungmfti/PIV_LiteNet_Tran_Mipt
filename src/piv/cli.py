"""Command line interface (spec section 14).

python -m piv.cli doctor
python -m piv.cli generate --config configs/demo_cpu.yaml
python -m piv.cli validate-data --input <dataset.h5>
python -m piv.cli infer --config <config.yaml> --input <dataset.h5> --output <result.h5>
python -m piv.cli train --config <config.yaml>
python -m piv.cli evaluate --prediction <result.h5> --reference <dataset.h5> --config <eval.yaml>
python -m piv.cli report --run-dir <path>
python -m piv.cli demo                      (one-command CPU demo)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    base = cfg.pop("extends", None)
    if base:
        parent = load_config(str(Path(path).parent / base))
        cfg = deep_merge(parent, cfg)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def cmd_doctor(args):
    from piv.doctor import run_doctor
    rep = run_doctor()
    print(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


def cmd_generate(args):
    from piv.engine.generate import generate
    cfg = load_config(args.config)
    res = generate(cfg, workers=args.workers)
    print(json.dumps(res, indent=2))
    return 0


def cmd_validate(args):
    from piv.data.schema import validate_dataset
    from piv.io.hdf5 import read_dataset
    cfg = load_config(args.config) if args.config else {}
    d = read_dataset(args.input, load_images=False)
    rep = validate_dataset(d, need_calibration=args.require_calibration, spectral=cfg.get("spectral"))
    print(json.dumps(rep.as_dict(), indent=2, default=str))
    return 0 if rep.ok else 2


def cmd_infer(args):
    from piv.engine.infer import infer
    cfg = load_config(args.config)
    prov = infer(cfg, args.input, args.output, device=args.device)
    print(json.dumps({"output": args.output, "model": prov["backbone"]["model_id"],
                      "refiner": prov.get("refiner", {}).get("kind"), "timing_s": prov["timing_s"]}, indent=2))
    return 0


def cmd_train(args):
    from piv.data.schema import validate_config
    from piv.engine.train import train
    cfg = load_config(args.config)
    rep = validate_config(cfg)
    if not rep.ok:
        print(json.dumps(rep.as_dict(), indent=2))
        return 2
    if args.resume:
        cfg.setdefault("training", {})["resume"] = True
    print(json.dumps(train(cfg), indent=2, default=float))
    return 0


def cmd_evaluate(args):
    from piv.engine.evaluate import evaluate
    cfg = load_config(args.config)
    res = evaluate(args.prediction, args.reference, cfg.get("evaluation", cfg))
    out = Path(args.output) if args.output else Path(args.prediction).with_suffix(".metrics.json")
    out.write_text(json.dumps(res, indent=2, default=float))
    summary = {k: res[k] for k in ("full", "interior", "grid") if k in res}
    print(json.dumps({"metrics_file": str(out), **summary}, indent=2, default=float))
    return 0


def cmd_report(args):
    from piv.report import make_report
    print(make_report(Path(args.run_dir)))
    return 0


def cmd_demo(args):
    from piv.demo import run_demo
    print(json.dumps(run_demo(args.config, args.out), indent=2, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="piv.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    g = sub.add_parser("generate"); g.add_argument("--config", required=True); g.add_argument("--workers", type=int, default=4)
    g.set_defaults(fn=cmd_generate)
    v = sub.add_parser("validate-data"); v.add_argument("--input", required=True); v.add_argument("--config")
    v.add_argument("--require-calibration", action="store_true"); v.set_defaults(fn=cmd_validate)
    i = sub.add_parser("infer"); i.add_argument("--config", required=True); i.add_argument("--input", required=True)
    i.add_argument("--output", required=True); i.add_argument("--device", default="cpu"); i.set_defaults(fn=cmd_infer)
    t = sub.add_parser("train"); t.add_argument("--config", required=True); t.add_argument("--resume", action="store_true")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("evaluate"); e.add_argument("--prediction", required=True); e.add_argument("--reference", required=True)
    e.add_argument("--config"); e.add_argument("--output"); e.set_defaults(fn=cmd_evaluate)
    r = sub.add_parser("report"); r.add_argument("--run-dir", required=True); r.set_defaults(fn=cmd_report)
    dm = sub.add_parser("demo"); dm.add_argument("--config", default="configs/demo_cpu.yaml")
    dm.add_argument("--out", default="outputs/demo"); dm.set_defaults(fn=cmd_demo)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
