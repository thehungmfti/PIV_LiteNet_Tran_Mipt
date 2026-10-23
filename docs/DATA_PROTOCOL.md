# DATA PROTOCOL

## Conventions (binding)

* Image axes: x to the right, y_image downward; pixel (0,0) is the **centre** of the top-left pixel.
* Displacement `[N,2,H,W]`: channel 0 = dx (right), channel 1 = dy_image (down), **pixel per pair**.
* Physical velocity (x right, y up): `u = sx·dx/dt_pair`, `v = −sy·dy_image/dt_pair` (scalar
  calibration, axis-aligned camera). Perspective cameras: map both end points (`mode: mapping`).
  Export in m/s is refused without calibration (`CalibrationError`).
* Times: `t_a`, `t_b` per pair (s); `t_velocity = (t_a+t_b)/2`; `dt_pair = t_b − t_a` is distinct from
  `dt_sample` (spacing of velocity fields). Non-uniform `dt_sample` → spectral evaluation refused.
* Ground truth = **forward Lagrangian displacement** of the fluid particle located at the pixel
  centre at `t_a`, integrated with RK4 to `t_b` with step-doubling until the change is < 1e-4 px
  (never `u(t)·dt`). Measured maximum change in the research data: ≤ 3e-7 px.
* Masks: `/masks/valid [N,H,W]`; predictions are NaN outside the valid region; losses never see NaN.
* Continuous mode pairs `(n, n+1)`; double-frame mode pairs `(A_k, B_k)` only — never `B_k` with `A_{k+1}`.

## HDF5 schema v1

Dataset file: `/images [T,H,W]` (or `/images_a`, `/images_b [N,H,W]`), `/pairs/index_a|index_b [N]`,
`/time/t_a|t_b [N]` float64 s, `/gt/displacement [N,2,H,W]` float32, `/masks/valid [N,H,W]` uint8,
group `/calibration` (attrs `mode, sx_m_per_px, sy_m_per_px`), `/meta/*` (flow, particles,
degradation, RK4 diagnostics as JSON). Root attributes: `schema_version, case_id, run_id,
acquisition_mode, split, ood, flow_type, noise, bit_depth, black_level, white_level, gt_convention`.
Every dataset carries an `axes` attribute (logical order) because MATLAB's `h5read` shows the reversed
order; `matlab/read_piv_h5.m` reverses it back and `matlab/validate_roundtrip.m` checks it with
axis-distinct tensors.

Prediction file: `/prediction/displacement [N,2,H,W]` (or `[N,2,ny,nx]` + `/prediction/grid_x|grid_y`
for vector-grid methods), `/prediction/valid`, `/prediction/std` (M3), `/prediction/velocity`
(optional, m/s), `/prediction/t_a|t_b`; provenance JSON in the `provenance` attribute and a
`*.provenance.json` side-car.

## Tier A synthetic data (implemented, `configs/research_data.yaml`)

| Item | Value |
|---|---|
| Image size | 128 × 128 px, 8 bit, continuous high-speed, fs = 1000 Hz (dt_pair = dt_sample = 1 ms) |
| Particles | Poisson count, ppp = 0.05, diameter N(2.5, 0.5) px (e⁻² definition, σ = d/4), peak N(150, 40) counts, Gaussian laser-sheet weighting, **exact pixel integration (erf)**, background 4 counts |
| Particle motion | RK4 through the whole sequence, periodic wrap in a box with 24 px margin (no spurious appearance inside the image) |
| Flows | `vortex_street` (wake-like: free stream + Gaussian deficit + Lamb–Oseen Kármán street, f_shed ∈ [50,140] Hz, smooth strength ramps), `cellular` (ψ = A(t) sin kx sin ky, two modulation frequencies 30–120 and 120–220 Hz, drift), OOD `orbiting_vortices` (unseen family) |
| Noise variants (same parent, same split) | clean / low (read 2, Poisson gain 4) / medium (read 5, gain 2, background ramp 10, brightness drift 5 %) / high (read 10, gain 1, ramp 20, drift 10 %, blur 0.5 px) |
| OOD imaging | ppp 0.025, d = 4 px, dropout 8 %/frame (out-of-plane **proxy** only), read 8, ramp 30, drift 15 %, blur 0.8 |
| Splits | train 14 parents × 145 frames, val 4 × 145, test 6 × 321, ood_flow 2 × 321, ood_imaging 2 × 321, test_double_frame 1 × 40 pairs (dt_pair 0.2 ms ≠ dt_sample 1 ms) |
| Split rule | by parent; all noise variants of a parent in one split; manifest `data/research/split_manifest.json` (sha256 recorded) with leakage check |

Calibration in this data is **declared** (0.1 mm/px) only to exercise the m/s export path; no
physical claims are made from it. Frequencies (Hz) are exact because timestamps are synthetic.

## Tier B (CFD/DNS) and Tier C (experiment)

NOT DONE: no CFD time series or time-resolved experimental images with metadata were provided.
The pipeline accepts them through the same schema (`write_piv_h5.m` from MATLAB, or Python). A
snapshot collection must not be presented as a time series. The information requested in spec §16
(acquisition mode, timing, calibration, flow parameters, frequency band, reference data) is still
needed before any m/s or Hz statement on real data.

## Reproduction

```
scripts/fetch_third_party.sh
python -m piv.cli generate --config configs/research_data.yaml --workers 4
python -m piv.cli validate-data --input data/research/test/<file>.h5
```
