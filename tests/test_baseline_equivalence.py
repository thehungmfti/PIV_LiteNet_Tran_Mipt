"""B0 audit tests.

1. custom-layer semantics vs direct loops transcribed from the Caffe CUDA kernels;
2. weights: every tensor of the third-party PyTorch port exists bit-exactly in the caffemodel;
3. interpreter vs the port (independent implementation) on author demo images;
4. accuracy on the authors' demo cases with GT (functional reproduction, NOT Caffe parity);
5. missing weights -> clear failure, never a fake B0 output.
Caffe itself cannot be run here, so exact Caffe parity is reported as NOT RUN in docs/BASELINE_AUDIT.md.
"""
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from piv.models import caffe_graph as cg
from piv.models.baseline_adapter import BackendUnavailable, LiteFlowNetBackend
from tests.conftest import B0_PROTO, B0_WEIGHTS, ROOT, needs_b0

DEMO = ROOT / "third_party/PIV-LiteFlowNet-en/demos/testedData"
PORT = ROOT / "third_party/piv_liteflownet-pytorch"


def read_flo(p):
    with open(p, "rb") as f:
        assert np.fromfile(f, np.float32, 1)[0] == 202021.25
        w, h = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float32, 2 * w * h).reshape(h, w, 2).transpose(2, 0, 1)


def ref_correlation(f1, f2, pad, md, s1, s2):
    """Loop transcription of CorrelateData (kernel_size 1)."""
    n, c, h, w = f1.shape
    p1 = np.pad(f1, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    p2 = np.pad(f2, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    th = math.ceil((h + 2 * pad - 2 * md) / s1); tw = math.ceil((w + 2 * pad - 2 * md) / s1)
    r = md // s2; gw = 2 * r + 1
    out = np.zeros((n, gw * gw, th, tw))
    for tc in range(gw * gw):
        s2o = (tc % gw - r) * s2; s2p = (tc // gw - r) * s2
        for y in range(th):
            for x in range(tw):
                x1, y1 = x * s1 + md, y * s1 + md
                out[:, tc, y, x] = (p1[:, :, y1, x1] * p2[:, :, y1 + s2p, x1 + s2o]).sum(1) / c
    return out


@pytest.mark.parametrize("pad,md,s1,s2", [(3, 3, 1, 1), (6, 6, 2, 2)])
def test_correlation_matches_kernel(pad, md, s1, s2):
    rng = np.random.default_rng(0)
    f1 = rng.normal(size=(1, 5, 11, 13)).astype(np.float32)
    f2 = rng.normal(size=(1, 5, 11, 13)).astype(np.float32)
    got = cg.correlation(torch.from_numpy(f1), torch.from_numpy(f2), pad, md, s1, s2).numpy()
    assert np.allclose(got, ref_correlation(f1, f2, pad, md, s1, s2), atol=1e-5)


def test_warp_matches_kernel():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(1, 2, 6, 7)).astype(np.float32)
    coords = np.stack(np.meshgrid(np.arange(7), np.arange(6)), 0)[None].astype(np.float32)
    coords = coords + rng.uniform(-2, 2, coords.shape).astype(np.float32)
    got = cg.warp(torch.from_numpy(u), torch.from_numpy(coords)).numpy()
    ref = np.zeros_like(u)
    for c in range(2):
        for h in range(6):
            for w in range(7):
                x, y = coords[0, 0, h, w], coords[0, 1, h, w]
                for px, py, wt in ((math.floor(x), math.floor(y), (1 - (x - math.floor(x))) * (1 - (y - math.floor(y)))),
                                   (math.floor(x) + 1, math.floor(y), (1 - (math.floor(x) + 1 - x)) * (1 - (y - math.floor(y)))),
                                   (math.floor(x), math.floor(y) + 1, (1 - (x - math.floor(x))) * (1 - (math.floor(y) + 1 - y))),
                                   (math.floor(x) + 1, math.floor(y) + 1, (1 - (math.floor(x) + 1 - x)) * (1 - (math.floor(y) + 1 - y)))):
                    if 0 <= px < 7 and 0 <= py < 6:
                        ref[0, c, h, w] += wt * u[0, c, py, px]
    assert np.allclose(got, ref, atol=1e-5)


def test_resample_identity_and_downsample_weights():
    W = cg.resample_matrix(64, 64, 1.0, 1.0, False)
    assert np.allclose(W, np.eye(64), atol=1e-6)
    D = cg.downsample_matrix(64, 32)
    assert np.allclose(D.sum(1), 1) and D.shape == (32, 64)
    # align-corner property of caffe Downsample: first/last output equal first/last input
    x = np.random.default_rng(0).normal(size=64)
    y = D @ x
    assert np.isclose(y[0], (D[0] @ x)) and D[0, 0] > 0.5 and D[-1, -1] > 0.5


def test_template_values_match_matlab_demo():
    v = cg.template_values(250, 300)
    assert v["$ADAPTED_WIDTH"] == 224 and v["$ADAPTED_HEIGHT"] == 288
    assert cg.matlab_num2str(250 / 224) == "1.1161"


@needs_b0
def test_port_weights_are_bit_exact_subset_of_caffemodel():
    port_w = PORT / "models/pretrain_torch/PIV-LiteFlowNet-en.paramOnly"
    if not port_w.is_file():
        pytest.skip("port not fetched")
    sd = torch.load(port_w, map_location="cpu", weights_only=False)
    caffe = cg.load_caffemodel(B0_WEIGHTS)
    pool = {}
    for arrs in caffe.values():
        for a in arrs:
            pool.setdefault(a.size, []).append(a.reshape(-1))
    missing = [k for k, v in sd.items()
               if not any(np.array_equal(v.numpy().reshape(-1), c) for c in pool.get(v.numel(), []))]
    assert missing == []


@needs_b0
@pytest.mark.parametrize("case,max_aee", [("backstep_Re1000", 0.05), ("DNS_turbulence", 0.12)])
def test_demo_accuracy(b0, case, max_aee):
    from skimage.io import imread
    a = imread(DEMO / f"{case}_img1.tif"); b = imread(DEMO / f"{case}_img2.tif")
    d = b0.predict(a, b, bit_depth=8)
    gt = read_flo(DEMO / f"{case}_gt.flo")
    aee = np.sqrt(((d - gt) ** 2).sum(0)).mean()
    assert aee < max_aee


def _load_port():
    fake = types.ModuleType("src.correlation")
    fake.FunctionCorrelation = lambda tensorFirst, tensorSecond, intStride: cg.correlation(
        tensorFirst, tensorSecond, 3 * intStride, 3 * intStride, intStride, intStride)
    sys.modules.setdefault("src", types.ModuleType("src"))
    sys.modules["src.correlation"] = fake
    mod = types.ModuleType("port_models")
    src = (PORT / "src/models.py").read_text().replace(".cuda()", "")
    exec(compile(src, "port_models", "exec"), mod.__dict__)
    sd = torch.load(PORT / "models/pretrain_torch/PIV-LiteFlowNet-en.paramOnly", map_location="cpu", weights_only=False)
    return mod, mod.piv_liteflownet(sd).eval()


@needs_b0
def test_interpreter_vs_port(b0):
    if not (PORT / "src/models.py").is_file():
        pytest.skip("port not fetched")
    from skimage.io import imread
    mod, net = _load_port()
    a = imread(DEMO / "backstep_Re1000_img1.tif").astype(np.float32)
    b = imread(DEMO / "backstep_Re1000_img2.tif").astype(np.float32)
    t = lambda x: torch.from_numpy(np.repeat(x[None, None], 3, 1).copy())
    with torch.no_grad():
        ours = b0(t(a), t(b))[0].numpy()
        port = net(t(a) / 255.0, t(b) / 255.0)[0].numpy()
    diff = np.abs(ours - port)
    assert diff.mean() < 5e-3 and diff.max() < 0.1     # known difference: image Downsample (see audit)
    # Two documented differences of the port: (i) torch bilinear instead of Caffe Downsample,
    # (ii) chained pyramid, whereas Caffe builds L2-L4 from the full image and L5-L6 from L4.
    # Re-creating Caffe's pyramid inside the port must close the gap to float32 round-off.
    orig = torch.nn.functional.interpolate
    store = {}

    def caffe_like(input, size, mode, align_corners):
        stream = "a" if store.get("next", "a") == "a" else "b"
        store["next"] = "b" if stream == "a" else "a"
        if f"full_{stream}" not in store:
            store[f"full_{stream}"] = input
        lvl = store.get(f"lvl_{stream}", 1) + 1
        store[f"lvl_{stream}"] = lvl
        src = store[f"full_{stream}"] if lvl <= 4 else store[f"L4_{stream}"]
        h, w = src.shape[-2:]
        wy = torch.from_numpy(cg.downsample_matrix(h, size[0])); wx = torch.from_numpy(cg.downsample_matrix(w, size[1]))
        out = cg.separable_apply(src, wy, wx)
        if lvl == 4:
            store[f"L4_{stream}"] = out
        return out
    mod.torch.nn.functional.interpolate = caffe_like
    try:
        with torch.no_grad():
            port2 = net(t(a) / 255.0, t(b) / 255.0)[0].numpy()
    finally:
        mod.torch.nn.functional.interpolate = orig
    d2 = np.abs(ours - port2)
    print("interp-vs-port: original max %.2e mean %.2e | caffe pyramid max %.2e mean %.2e"
          % (diff.max(), diff.mean(), d2.max(), d2.mean()))
    assert d2.max() < 1e-4 and d2.mean() < 1e-5


def test_missing_weights_fail_clearly(tmp_path):
    with pytest.raises(BackendUnavailable):
        LiteFlowNetBackend(B0_PROTO, tmp_path / "nope.caffemodel")


def test_demo_backend_requires_flag():
    from piv.models.registry import build_backbone
    with pytest.raises(ValueError):
        build_backbone({"name": "demo"})
    assert build_backbone({"name": "demo", "demo_only": True}).info()["demo_only"] is True


@needs_b0
def test_size_policies_non_multiple_of_32():
    from piv.data.synthetic import ParticleConfig, ParticleField, UniformFlow
    pf = ParticleField(70, 100, ParticleConfig(), np.random.default_rng(0))
    a = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
    pf.advect(UniformFlow(u=2.0, v=1.0), 0, 1)
    b = np.clip(np.round(pf.render()), 0, 255).astype(np.uint8)
    for pol in ("resize", "pad"):
        bk = LiteFlowNetBackend(B0_PROTO, B0_WEIGHTS, size_policy=pol)
        d = bk.predict(a, b)
        assert d.shape == (2, 70, 100)
        c = d[:, 8:-8, 8:-8]
        assert abs(c[0].mean() - 2.0) < 0.15 and abs(c[1].mean() - 1.0) < 0.15
