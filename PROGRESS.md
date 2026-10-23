# PROGRESS

Run environment: cloud container, 4 CPU cores, no GPU, no MATLAB (GNU Octave 8.4 for PIVlab).
Status legend: PASS / FAIL / NOT RUN / PARTIAL.

| Milestone | Status | What was done | Command(s) | Checks |
|---|---|---|---|---|
| M0 Audit | PASS (Caffe parity NOT RUN) | Pinned commits/licences/weights hash; read custom-layer CUDA sources; decided PyTorch interpreter of the deploy graph; found deploy/caffemodel name mismatch (harmless) and two deviations of the third-party port | `python -m pytest tests/test_baseline_equivalence.py -s`, `python -m piv.cli doctor` | independent-implementation parity 1.1e-5 px; demo AEE 0.030 / 0.081 px; docs/BASELINE_AUDIT.md |
| M1 Infrastructure | PASS (MATLAB NOT RUN) | HDF5 schema v1, analytic flows + RK4 GT, erf particle renderer, degradations, metrics, CLI, demo, MATLAB bridge | `python -m piv.cli demo`, `python -m pytest` | 49 tests pass; demo outputs in outputs/demo |
| M2 Baseline | PASS | B0 on test/OOD; PIVlab (Octave adapter, interface-only) and OpenPIV comparators; split manifest | `scripts/benchmark.py predict/evaluate` | sign/coordinate tests for comparators; summary_main.csv |
| M3 Adaptation | PASS (1 seed) | B1 fine-tuned 600 it., lr 1e-5, train/val only | `python -m piv.cli train --config configs/finetune.yaml` | val EPE 0.190 → 0.135; test −20…−35 % vs B0 |
| M4 Temporal | PASS (1 seed) | M1 5-field residual refiner + parameter-matched spatial ablation S1; edge masking; full-context metrics | `configs/temporal.yaml`, `configs/spatial.yaml` | M1 −4…−19 % vs B1 (CI excludes 0); S1 −1…−2 % |
| M5 Spectral | PARTIAL (negative result) | M2 with log-PSD loss on 128-sample segments; PSD/coherence/phase metrics, noise sweep | `configs/spectral.yaml` | EPE ≈ M1; band-power error not improved (target ≤ 10 % not met) |
| M6 Uncertainty | PASS (calibration partial) | M3 head-only NLL, variance scale fitted on val (1.43) | `configs/uncertainty.yaml` | 95 % coverage 0.94–0.98 test; 68 % over-covered (0.84–0.90) |
| M7 Package | PASS | tables (CSV/JSON), figures, reliability analysis, docs, README | `scripts/make_figures.py`, `scripts/reliability_analysis.py --with-b1` | docs/VALIDATION.md |

Tier B (CFD) and tier C (experiment): NOT DONE — no data provided (spec §16 questions open).
Seeds: 1 per model (≥ 3 required before publication tables).
