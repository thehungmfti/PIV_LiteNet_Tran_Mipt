# VALIDATION — software acceptance, benchmark and reliability analysis

Everything below was produced in one cloud run (4 CPU cores, no GPU; versions in `environment.lock`)
on **tier-A synthetic data with exact ground truth** (docs/DATA_PROTOCOL.md). One training seed per
model (CPU budget) — the spec asks for ≥ 3 seeds for research claims, so all model-vs-model
differences below are **pilot results**. Confidence intervals: moving-block bootstrap (block 16
pairs, runs resampled, 1000 replicates); pixels are never treated as independent.

Raw numbers: `outputs/benchmark/summary_main.csv` (72 rows: split × noise × method),
per-run metrics `outputs/benchmark/metrics/<method>/<run>.json`, reliability CSVs in
`outputs/reliability/`, figures in `outputs/figures/` and `outputs/reliability/`.

## 1. Software acceptance (spec §12)

| Requirement | Status | Evidence |
|---|---|---|
| CPU demo from one command → HDF5, metrics JSON/CSV, PNG | **PASS** | `python -m piv.cli demo` (11 s) → `outputs/demo/` |
| schema / axis / sign / calibration / mask / pairing tests | **PASS** | `pytest`: 49 tests pass (incl. B0 and comparator tests) |
| MATLAB ↔ Python round trip | **NOT RUN** | MATLAB not available; Octave 8 lacks h5read/h5write. Python side (`scripts/roundtrip_check.py`) tested; run `matlab/validate_roundtrip.m` |
| continuous vs double-frame, dt_pair ≠ dt_sample | **PASS** | tests/test_pairing.py; `test_double_frame` split (dt_pair 0.2 ms, dt_sample 1 ms): B0 EPE 0.032 px, spectral fs taken as 1000 Hz from velocity timestamps |
| tone in the correct FFT bin (±1), integrated PSD = variance | **PASS** | tests/test_spectral_signal.py (periodogram exact to 1e-10; Hann within 2 % on average; torch PSD == scipy) |
| checkpoint resume reproduces state; finite loss/grad smoke | **PASS** | tests/test_checkpoint_resume.py (interrupted == uninterrupted to 1e-6) |
| missing baseline weights → clear failure, no fake B0 | **PASS** | `BackendUnavailable`; demo backend must be declared `demo_only` |
| provenance, demo/research separation | **PASS** | every prediction has provenance JSON (weights sha256, cache key, timing, demo flag); manifests per run |
| Caffe exact parity of B0 | **NOT RUN** | no Caffe build; functional reproduction + independent-implementation parity 1.1e-5 px (BASELINE_AUDIT.md) |

## 2. Main benchmark — test split (6 parents × 320 pairs per noise level, 128×128 px)

Dense metrics: interior (8 px margin), full-context pairs. Grid metrics: all methods at the common
8-px vector grid (15 × 15) against the exact Lagrangian GT at the grid points; dense methods are
bilinearly sampled there. EPE in pixel/pair.

| noise | B0 | B1 | M1 | S1 (spatial abl.) | M2 | PIVlab | OpenPIV |
|---|---|---|---|---|---|---|---|
| clean — dense EPE | 0.124 | 0.081 | **0.077** | 0.079 | 0.078 | – | – |
| clean — grid EPE | 0.127 | 0.086 | **0.082** | 0.084 | 0.083 | 0.370 | 0.274 |
| low — grid EPE | 0.135 | 0.097 | **0.089** | 0.095 | 0.089 | 0.387 | 0.289 |
| medium — grid EPE | 0.158 | 0.120 | **0.105** | 0.119 | 0.105 | 0.393 | 0.305 |
| high — grid EPE | 0.222 | 0.181 | **0.146** | 0.179 | 0.146 | 0.436 | 0.361 |

Paired difference in dense EPE versus B1 (mean [95 % CI]):

| noise | B0 − B1 | M1 − B1 | S1 − B1 | M2 − B1 |
|---|---|---|---|---|
| clean | +0.044 [+0.017, +0.086] | −0.0033 [−0.0044, −0.0021] | −0.0013 [−0.0026, −0.0000] | −0.0028 [−0.0038, −0.0012] |
| low | +0.040 [+0.013, +0.082] | −0.0078 [−0.0088, −0.0067] | −0.0014 [−0.0028, −0.0001] | −0.0075 [−0.0084, −0.0061] |
| medium | +0.039 [+0.010, +0.086] | −0.0155 [−0.0171, −0.0139] | −0.0015 [−0.0029, −0.0002] | −0.0151 [−0.0168, −0.0134] |
| high | +0.043 [+0.008, +0.095] | −0.0336 [−0.0368, −0.0305] | −0.0017 [−0.0035, +0.0000] | −0.0337 [−0.0368, −0.0307] |

Spectral fidelity at the grid probes (band 15–300 Hz, Welch N = 64, Hann; energetic probes only,
i.e. GT band power ≥ 10 % of the run maximum):

| noise | metric | B0 | B1 | M1 | M2 | PIVlab | OpenPIV |
|---|---|---|---|---|---|---|---|
| clean | band-power rel. error (median) | 0.376 | **0.245** | 0.241 | 0.256 | 0.492 | 0.584 |
| clean | total band power pred/GT | 0.71 | 0.86 | 0.83 | 0.80 | 4.57 | 3.99 |
| clean | coherence at f_ref | 0.939 | 0.961 | 0.962 | 0.962 | 0.607 | 0.658 |
| clean | peak frequency within 1 bin | 0.958 | 0.993 | 0.993 | 0.993 | 0.644 | 0.655 |
| high | band-power rel. error (median) | 0.352 | **0.266** | 0.336 | 0.361 | 0.615 | 0.784 |
| high | total band power pred/GT | 1.26 | 1.22 | 0.91 | 0.87 | 5.44 | 4.74 |
| high | coherence at f_ref | 0.855 | 0.895 | 0.896 | 0.897 | 0.567 | 0.600 |
| high | peak within 1 bin (dense) | 0.856 | 0.911 | **0.984** | 0.977 | – | – |
| high | phase error at f_ref (deg, dense) | 9.1 | 4.4 | 5.0 | 5.1 | – | – |

Out-of-distribution (grid EPE): unseen flow `orbiting_vortices`, clean / high:
B0 0.033 / 0.109, B1 0.028 / 0.107, M1 0.036 / **0.090**, M2 0.039 / 0.091, PIVlab 0.101 / 0.151,
OpenPIV 0.060 / 0.141. Unseen imaging (sparse, large particles, 8 % dropout, heavy noise):
B0 0.218, B1 0.201, M1 **0.147**, M2 0.143, PIVlab 0.645, OpenPIV 0.509.

Uncertainty (M3, test, after the validation-fitted variance scale 1.43): marginal coverage of the
95 % interval 0.94–0.98 (dx), 0.94–0.96 (dy) — close to nominal; the 68 % interval covers 0.84–0.90,
i.e. the Gaussian model is too wide in the core / the error distribution is heavier-tailed. On OOD
data coverage is conservative (95 % → 0.98–0.99). M3 displacement is identical to M2 (head-only).

Timing (indicative only, single thread under concurrent load): B0/B1 ≈ 0.55–0.6 s per 128² pair,
refiner ≈ 0.15 s per pair on top of the cached backbone, PIVlab (Octave) ≈ 0.14 s, OpenPIV ≈ 0.03 s.
The refiner is offline/centred with a 2-sample look-ahead; no real-time claim.

### Interpretation
1. **B0 vs C1:** the original PIV-LiteFlowNet-en is 2–3× more accurate than PIVlab and OpenPIV at the
   same grid points on every split, with much higher coherence and correct peak frequencies. The
   gap is largest where the flow has small scales (vortex cores 5–9 px, cellular wavelengths ≥ 48 px),
   consistent with the spatial-response curves (§3).
2. **B1 (domain adaptation):** −20 to −35 % EPE vs B0 on test; it also improves band power and phase.
3. **M1 (temporal refiner):** further −4 % (clean) to −19 % (high noise) vs B1 with tight CIs. The
   parameter-matched spatial-only ablation S1 gains only 1–2 %, so the benefit comes from the
   **temporal context**, not from extra capacity.
4. **Trade-off (negative result for M2):** M1 lowers EPE mostly by removing temporal noise, and it
   *under-estimates* in-band power (ratio 0.83–0.91), so the band-power error does not improve over
   B1 (it gets worse at high noise). The spectral loss (M2) did **not** fix this: M2 ≈ M1 in EPE and
   slightly worse in band power. Diagnosis: with analytic GT whose PSD is essentially zero between
   the tones, the log-PSD loss is dominated by those empty bins and pushes towards suppressing
   fluctuations (see HANDOFF item 5). The design targets (band-power error ≤ 10 %) are **not met**.
5. On the unseen flow family at low noise, M1/M2 are slightly worse than B1 (+0.006–0.012 px);
   they help only at high noise — the refiner partly learned flow-specific dynamics.

## 3. Reliability analysis of the computation (`scripts/reliability_analysis.py`)

Uniform translation at 30°, 128² images, 6 pairs per point, "low" noise unless varied; EPE at the
common grid (B0 dense values in the CSVs).

| factor | B0 | PIVlab | OpenPIV | finding / recommendation |
|---|---|---|---|---|
| displacement 0.25–16 px | 0.027–0.044 | 0.033–0.057 | 0.030–0.106 | B0 stable up to 16 px; keep max displacement ≲ 10–12 px (dt_pair choice) |
| particle density 0.005–0.15 ppp | 0.080 → 0.024 | 0.161 → 0.049 | 0.268 → 0.030 | ≥ 0.02 ppp recommended; B0 degrades gracefully when sparse |
| particle diameter 1–6 px | 0.072 (1 px), best 0.023 (3–4 px) | 0.087 → 0.039 (4 px) | best 0.039 (3 px) | 2–4 px particle images; 1 px gives peak-locking-like errors for all methods |
| read noise 0–30 counts | 0.016 → 0.235 | 0.047 → 0.197 | 0.030 → 0.251 | CNN advantage disappears above ≈ 20 counts (8-bit) |
| particle loss between frames 0–30 % | 0.028 → 0.294 | 0.051 → 0.152 | 0.044 → 0.185 | **above ≈ 20 % loss (strong out-of-plane motion) cross-correlation is more robust** — reduce dt_pair or thicken the sheet |
| particle peak intensity 20–220 counts | 0.112 → 0.025 | 0.125 → 0.050 | 0.142 → 0.039 | use the dynamic range; no per-frame normalisation was needed |
| spatial response (gain, amp. 1 px) | λ16: 0.60, λ24: 0.95, λ32: 0.99 | λ24: 0.36, λ32: 0.56, λ48: 0.78 | λ24: 0.42, λ32: 0.76, λ48: 0.94 | grid methods undefined at λ ≤ 16 px (aliasing); dense CNN resolves ≈ 2× smaller scales |
| image size not multiple of 32 | bias dx −0.025 px at 100×150, −0.021 at 250×250 | – | – | original pipeline resizes → ~1 % scale bias. Options: crop ROI to a multiple of 32 (recommended) or `size_policy: pad` (removes the bias on uniform flow, EPE unchanged on complex flow); default kept = original |
| 16-bit (×257) / 12-bit (×16) input | max diff 0.0000 / 0.0017 px | – | – | bit-depth path verified; 12-bit differs only by the 255/4095 vs 1/16 scale |

PIVlab settings on a shear flow (γ = 0.04): 64-32-16 (3 passes) 0.060 px; 4 passes (…-16) 0.056
(best); final 32-px window (2 passes) 0.29; 2-D Gauss sub-pixel 0.062; no validation 0.060;
`*spline` deformation NOT RUN (Octave `interp2` limitation). Vector positions reported by PIVlab
(window centre at IA/2) fit the GT better than half-pixel-shifted positions (0.060 vs 0.062) — the
coordinate convention used in this benchmark is therefore correct. **Recommendation for PIVlab
comparisons: ≥ 3 passes ending at 16 px (4 passes slightly better), 3-point Gauss, linear
deformation, standard validation.**

Other checks performed while building the pipeline and fixed:
* generator wrote no images in its first version (caught by the demo; regenerated);
* vortex birth/death discontinuities made RK4 non-convergent → smooth strength ramps (converged to 1e-9 px);
* relative band-power error is undefined where GT does not fluctuate → energetic-probe definition;
* spatial-response fit at λ ≤ 2·grid spacing is aliased → reported as undefined;
* parallel CPU jobs without thread limits ran 4–5× slower → explicit `OMP_NUM_THREADS`.

## 4. Limitations

Synthetic analytic flows only (no CFD, no experiment); 128² images; one seed; CPU-sized training
(B1 600 iterations, still improving); the spectral-loss variant failed its objective; the UQ head is a
Gaussian-diagonal predictive variance, not a calibrated posterior; Caffe parity and the MATLAB
bridge are not executed here. No statement in m/s or Hz about a real flow is made.
