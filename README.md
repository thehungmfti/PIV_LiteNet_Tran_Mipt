# Temporal and Spectral PIV Refinement with a PIV-LiteFlowNet-en Baseline

Research code implementing `PIV_LiteFlowNet_Codex_Spec.md` (v1.0): a reproducible CLI pipeline
from synthetic time-resolved particle images to dense displacement fields, a faithful PyTorch
execution of the original **PIV-LiteFlowNet-en** Caffe model (B0), its fine-tuned version (B1),
a temporal residual refiner (M1), spectral loss (M2), uncertainty head (M3), a parameter-matched
spatial-only ablation (S1) and two cross-correlation comparators (**PIVlab** run in GNU Octave and
**OpenPIV**), with a reliability analysis of the computation.

> Status, results and limitations: **[docs/VALIDATION.md](docs/VALIDATION.md)** ·
> baseline audit: **[docs/BASELINE_AUDIT.md](docs/BASELINE_AUDIT.md)** ·
> data conventions: **[docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md)** ·
> milestone log: [PROGRESS.md](PROGRESS.md) · continuation notes: [HANDOFF.md](HANDOFF.md)

All numbers in this repository come from **tier-A synthetic data with exact ground truth**
(analytic flows). No CFD/experimental data were provided, so nothing here is a claim about real
flows, m/s or Hz of a real experiment. Novelty of the proposed title has not been assessed.

## Layout

```
configs/      research_data, baseline (B0), finetune (B1), temporal (M1), spatial (S1),
              spectral (M2), uncertainty (M3), demo_cpu, eval
src/piv/      data/ (schema, dataset, pairing, calibration, split, synthetic, degradation)
              models/ (caffe_graph = Caffe deploy-graph interpreter, baseline_adapter, registry,
                       temporal_refiner, uncertainty_head)
              losses/ metrics/ engine/ (generate, infer, train, evaluate, checkpoint) io/ cli.py
comparators/  pivlab_runner.py + pivlab_octave/ (Octave adapter generated from PIVlab source),
              openpiv_runner.py
matlab/       run_demo, run_inference, read_piv_h5, write_piv_h5, plot_results, validate_roundtrip
scripts/      fetch_third_party.(sh|ps1), doctor, prepare_data, benchmark.py,
              reliability_analysis.py, make_figures.py, roundtrip_check.py
tests/        schema, axis_sign, pairing, analytic_motion, spectral_signal, masking,
              baseline_equivalence, checkpoint_resume, comparators
docs/         BASELINE_AUDIT.md, DATA_PROTOCOL.md, VALIDATION.md
outputs/      metrics (JSON/CSV), figures, training logs/checkpoints (small); large data excluded
```

## Setup (working directory = repository root)

Windows (PowerShell) or Linux, Python ≥ 3.10 (tested: see `environment.lock`, Python 3.13, CPU only).

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
scripts/fetch_third_party.sh          # Windows: powershell -File scripts/fetch_third_party.ps1
python -m piv.cli doctor
```

`fetch_third_party` clones PIV-LiteFlowNet-en (weights; research-use-only licence), PIVlab (MIT),
the abrosua PyTorch port (MIT, used only for cross-checks) and PIV_dataset at pinned commits and
verifies the caffemodel SHA-256. Without the weights, B0/B1 fail with a clear error and the demo
falls back to a `demo_only` backend that is never reported as a baseline.

PIVlab comparator: MATLAB (`matlab -batch`) is used if present, otherwise GNU Octave ≥ 8 with the
`image` package (`pkg install -forge image`). OpenPIV: `pip install openpiv`.

## Commands (CLI contract)

```bash
python -m piv.cli demo                                   # one-command CPU demo -> outputs/demo
python -m piv.cli generate --config configs/research_data.yaml
python -m piv.cli validate-data --input data/research/test/<run>.h5
python -m piv.cli infer --config configs/baseline.yaml --input <dataset.h5> --output <result.h5>
python -m piv.cli train --config configs/finetune.yaml   # B1, then temporal / spatial / spectral / uncertainty
python -m piv.cli evaluate --prediction <result.h5> --reference <dataset.h5> --config configs/eval.yaml
python -m piv.cli report --run-dir outputs/demo
python -m pytest
```

Full study (≈ 6 h on 4 CPU cores):

```bash
python -m piv.cli generate --config configs/research_data.yaml --workers 4
python -m piv.cli train --config configs/finetune.yaml
python -m piv.cli train --config configs/temporal.yaml
python -m piv.cli train --config configs/spatial.yaml
python -m piv.cli train --config configs/spectral.yaml
python -m piv.cli train --config configs/uncertainty.yaml
for m in B0 B1 M1 S1 M2 M3 PIVlab OpenPIV; do
  python scripts/benchmark.py predict --method $m --splits test ood_flow ood_imaging; done
python scripts/benchmark.py evaluate --methods B0 B1 M1 S1 M2 M3 PIVlab OpenPIV --splits test ood_flow ood_imaging
python scripts/benchmark.py aggregate --methods B0 B1 M1 S1 M2 M3 PIVlab OpenPIV --splits test ood_flow ood_imaging --ref B1
python scripts/reliability_analysis.py --with-b1
python scripts/make_figures.py
```

When several jobs run in parallel on a CPU, give each an explicit thread budget
(`OMP_NUM_THREADS=...`); oversubscription slowed training 4–5× in our runs.

### MATLAB R2025b

```matlab
addpath('matlab')
run_demo('C:\path\to\python.exe')
run_inference('configs/baseline.yaml', 'D:\data with spaces\run.h5', 'D:\out\pred.h5', 'python')
S = read_piv_h5('D:\out\pred.h5');      % arrays in logical order, e.g. [N,2,H,W]
validate_roundtrip('python')            % axis-order round trip; must pass before trusting MATLAB I/O
```

MATLAB was not available in the execution environment: the MATLAB bridge is **NOT RUN** (the
Python side of the round trip is tested). Run `validate_roundtrip` once on your machine.

## Key conventions

dx channel 0 (right), dy_image channel 1 (down), pixel/pair; GT = forward Lagrangian displacement
of the particle at the pixel centre at t_a; `v_phys = −sy·dy/dt_pair`; NaN outside the valid mask;
offline centred refiner with 2 samples look-ahead (not real-time). Details: docs/DATA_PROTOCOL.md.

## Licences

This repository's own code: MIT (see LICENSE). Third-party components keep their licences and are
not redistributed: PIV-LiteFlowNet-en code/weights are for research use only (commercial use needs
the authors' consent); PIVlab MIT; OpenPIV GPLv3 (installed separately). Cite Cai et al., IEEE TIM
69(6) 3538–3554 (2020), doi:10.1109/TIM.2019.2932649, and Thielicke & Sonntag, JORS 9(1) 12 (2021)
for PIVlab when using the corresponding parts.
