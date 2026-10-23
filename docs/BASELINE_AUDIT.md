# BASELINE AUDIT (milestone M0)

Status: **PASS for functional reproduction of B0 in PyTorch; exact Caffe parity NOT RUN** (no Caffe/CUDA
build is available in the execution environment; see "What is not verified").

## 1. Sources, commits, licences

| Component | Source | Commit | Licence / terms | Use here |
|---|---|---|---|---|
| PIV-LiteFlowNet-en code + Caffe weights | github.com/shengzesnail/PIV-LiteFlowNet-en | `128768b5` | "research purposes only, all rights reserved, commercial use requires consent" (README) | B0/B1 weights, deploy prototxt, custom-layer sources (read only) |
| caffemodel used | `caffe/models/PIV-LiteFlowNet-en/PIV-LiteFlowNet-en.caffemodel` | sha256 `8b9a0c76eb85e5a8988654fd0b165ee0fb73088d4d3d039208567e6e96fd868b` | as above | B0 |
| deploy prototxt | `PIV-LiteFlowNet-en_deploy.prototxt` | sha256 `69b0cf13…c06` | as above | graph definition |
| PyTorch port | github.com/abrosua/piv_liteflownet-pytorch | `dae39338` | MIT | cross-check only (never the baseline) |
| PIVlab | github.com/Shrediquette/PIVlab | `ec255f2c` | MIT | comparator C1 (GNU Octave 8.4) |
| OpenPIV-python | PyPI `openpiv==0.26.1` | – | GPLv3 (used as an installed tool, not redistributed) | comparator C1b |
| Authors' dataset generator | github.com/shengzesnail/PIV_dataset | `5fecd475` | see its README | not used for training (provenance only) |

Third-party code and weights are **not** committed to this repository; `scripts/fetch_third_party.sh`
(or `.ps1`) clones them at the pinned commits and checks the caffemodel hash. Their licences are unchanged.

## 2. Backend decision (evidence)

Caffe (CUDA 8 / cuDNN 5.1 / MatCaffe R2017b) cannot be built in the cloud container (no GPU) and is
not expected to work with MATLAB R2025b without effort; the spec forbids assuming compatibility.
Decision: **execute the original deploy graph with the original weights in PyTorch**, through a
layer-by-layer interpreter (`src/piv/models/caffe_graph.py`) that

* parses the deploy prototxt with the original `caffe.proto` (compiled with protoc),
* fills the template exactly like `demos/test_pivLiteflownet.m` (floor to a multiple of 32,
  `width/single(adapted)`, MATLAB `num2str` formatting),
* loads blobs from the `.caffemodel` **by layer name** (Caffe's rule) and uses the prototxt filler
  where no blob exists (Caffe's behaviour),
* re-implements each custom layer from its CUDA source.

This is labelled **"B0 (PyTorch interpreter of the Caffe deploy graph)"** everywhere, never "Caffe".

## 3. Preprocessing of the original pipeline (from `test_pivLiteflownet.m` + prototxt)

1. `double(imread(...))` – 8-bit grey values 0..255, **no per-frame normalisation**.
2. Grey image replicated into 3 identical channels (`cat(3,im,im,im)`); matcaffe permutation `[2,1]`
   is only MATLAB's column-major view (equivalent to NCHW in C++).
3. `Eltwise` × 0.00392156862745 (1/255).
4. `DataAugmentation` in TEST phase: copy + subtraction of the learned per-channel mean stored in blob 2
   (`recompute_mean: 3000`, iteration counter 932 400 > 3000 so no recomputation):
   img0 mean = (0.173935, 0.180594, 0.192608), img1 mean = (0.172978, 0.179518, 0.191300).
5. `Resample` to the adapted size (multiple of 32), `FlowUnscaling` × 5.0, `Resample` back,
   `ScaleMag_flow` 1×1 convolution with **diagonal filler** (SCALE_WIDTH, SCALE_HEIGHT) to rescale
   the vector components after resizing.
6. Output channel 0 = dx (right), channel 1 = dy (down), pixel per pair, on the image-1 grid.

For 12/16-bit images the project maps linearly to 0..255 with declared black/white levels
(`preprocessing_id = cai2019_matlab_demo+linear_to_8bit`); verified numerically identical to the
8-bit path for 16-bit×257 data (see VALIDATION.md, reliability §bit depth).

## 4. Custom layers – semantics reproduced (and quirks kept on purpose)

| Layer | Source file | Reproduced behaviour |
|---|---|---|
| Correlation | `correlation_layer.cu` | zero padding, `top = ceil((H+2p-2md)/s1)`, displacement grid step `s2`, channel index `(dy)*(2r+1)+(dx)`, **mean over kernel²·C** |
| Warp | `warp_layer.cu` | bilinear at absolute pixel coordinates, each out-of-range corner contributes 0 (= `grid_sample(align_corners=True, zeros)`) |
| Grid | `grid_layer.cpp` | channel 0 = x (column index), channel 1 = y |
| Resample (LINEAR) | `resample_layer.cu` | triangle filter, antialias only when down-sampling, **`x_in = x_out·fx + fy/2 − 0.5` (fx/fy swap of the FlowNet kernel kept)**, CUDA `round` (half away from zero) |
| Downsample | `downsample_layer.cu` | align-corner triangle filter with radius `ceil(scale)`; L2–L4 images from the full image, **L5–L6 from L4** (as wired in the prototxt) |
| ChannelNorm | `channel_norm_layer.cu` | sqrt(Σ_c x²) |
| ExpMax | `expmax_layer.cu` | exp(x − max_c x); the Caffe kernel takes the max of sample 0 only (outer_num = 1); identical for batch 1, and the normalisation that follows makes the per-sample max mathematically equivalent for batch > 1 |
| Im2col / Reduction(MEAN, axis 1, coeff −1) / Bias(axis 0) / NegSquare / Power | Caffe | standard semantics |
| Convolution with 2 bottoms | Caffe | shared weights applied to both images (Siamese NetC) |

**Name mismatch found:** the deploy prototxt names the 1×1 kernel-sum convolutions
`scaled_flow0_R_L*_x/y` and `sum_exp_kernel_R_L*`, while the caffemodel stores `scaled_flow_R_L*_x/y`.
Caffe therefore uses the **constant filler (1, bias 0)** in deploy. The stored training blobs are
also exactly 1 / 0 with `lr_mult: 0`, so there is no numerical consequence; recorded in
`backbone.info()["filler_initialised_layers"]` and `caffemodel_blobs_not_used_by_deploy`.

## 5. Checks performed (all reproducible with `python -m pytest tests/test_baseline_equivalence.py -s`)

| Check | Result |
|---|---|
| Correlation vs loop transcription of `CorrelateData` (both stride settings) | max diff < 1e-5 |
| Warp vs loop transcription of `ForwardGPU` | max diff < 1e-5 |
| Template arithmetic vs MATLAB demo (e.g. 250×300 → 224×288, `num2str(250/224)='1.1161'`) | equal |
| Every tensor of the third-party port's converted weights found **bit-exactly** in the caffemodel | 252/252 |
| Interpreter vs port, author demo `backstep_Re1000` (256×256) | max 2.98e-2 px, mean 8.2e-4 px |
| …after re-creating Caffe's image pyramid inside the port (two port deviations, below) | **max 1.1e-5 px, mean 3.1e-6 px** (float32 round-off) |
| Accuracy on authors' demo pairs with GT (`backstep_Re1000`, `DNS_turbulence`) | AEE 0.0299 px / 0.0809 px |

Port deviations identified (they make the port a *near* reproduction only):
1. image pyramid by `torch.nn.functional.interpolate(bilinear, align_corners=False)` instead of
   Caffe's align-corner `Downsample`;
2. chained pyramid (each level from the previous one) instead of L2–L4 from the full image and
   L5–L6 from L4;
3. requires CuPy/CUDA for the correlation (replaced by the faithful PyTorch correlation for the check);
4. no Resample path for sizes that are not multiples of 32.

Parity threshold policy: interpreter-vs-independent-implementation agreement is required to be at
float32 round-off level (< 1e-4 px max) once known structural differences are removed; this was met.

## 6. What is NOT verified

* **Exact parity with Caffe itself** (same image → same tensor in Caffe): NOT RUN. Residual risk:
  undocumented behaviour of the compiled Caffe fork (e.g. cuDNN algorithm differences, ~1e-6 level).
  Procedure for a machine with Caffe: run `demos/test_pivLiteflownet.m` on the three demo pairs, save
  `res_all{1}`, compare with `python -m piv.cli infer --config configs/baseline.yaml` (tolerance 1e-4 px).
* MATLAB R2025b compatibility of MatCaffe: not assessed (not needed by this pipeline).

## 7. Consequences for the study

* B0 = original weights + original preprocessing via the interpreter (functionally reproduced,
  Caffe parity pending). B1 = the same graph fine-tuned in PyTorch (gradients flow through all ops;
  Downsample/Resample are linear operators). The original multi-level training loss of the authors
  is **not** reproduced: B1 is fine-tuned with a final-output Charbonnier EPE loss (documented).
