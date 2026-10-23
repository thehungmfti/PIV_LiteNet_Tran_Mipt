"""Faithful PyTorch interpreter for the PIV-LiteFlowNet(-en) Caffe deploy graph.

The deploy prototxt is parsed with the original ``caffe.proto`` and executed
layer by layer. Every custom layer (Correlation, Warp, Resample, Downsample,
Grid, ChannelNorm, ExpMax, Im2col, Reduction, Bias, NegSquare,
DataAugmentation) is re-implemented from the CUDA/C++ sources shipped in
https://github.com/shengzesnail/PIV-LiteFlowNet-en (see docs/BASELINE_AUDIT.md
for the line-by-line mapping and the known quirks that are reproduced, e.g. the
fx/fy swap in the FlowNet Resample kernel).

Weights are read directly from the original ``.caffemodel`` by layer name, the
same rule Caffe uses; layers without stored blobs get their prototxt filler,
again as Caffe does. All operations are differentiable torch ops, so the graph
can also be fine-tuned (B1).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from piv.models._caffe import caffe_pb2  # resolved with a protobuf-version fallback
from google.protobuf import text_format

TEMPLATE_VARS = ("$TARGET_WIDTH", "$TARGET_HEIGHT", "$ADAPTED_WIDTH",
                 "$ADAPTED_HEIGHT", "$SCALE_WIDTH", "$SCALE_HEIGHT")


def matlab_num2str(x: float) -> str:
    """Mimic MATLAB num2str for a scalar, as used by the original demo to fill the template."""
    if float(x).is_integer():
        return str(int(x))
    digits = max(math.ceil(math.log10(abs(x))), 1) + 4
    return f"{x:.{digits}g}"


def template_values(width: int, height: int, divisor: int = 32) -> dict:
    """Same arithmetic as demos/test_pivLiteflownet.m (floor to multiple of 32, single precision ratio)."""
    aw = (width // divisor) * divisor
    ah = (height // divisor) * divisor
    if aw == 0 or ah == 0:
        raise ValueError(f"image {width}x{height} smaller than divisor {divisor}")
    sw = float(np.float32(width) / np.float32(aw))
    sh = float(np.float32(height) / np.float32(ah))
    return dict(zip(TEMPLATE_VARS, (width, height, aw, ah, sw, sh)))


def load_net_param(prototxt: str | Path, width: int, height: int) -> caffe_pb2.NetParameter:
    txt = Path(prototxt).read_text()
    for k, v in template_values(width, height).items():
        txt = txt.replace(k, matlab_num2str(v))
    net = caffe_pb2.NetParameter()
    text_format.Parse(txt, net)
    return net


def load_caffemodel(path: str | Path) -> dict[str, list[np.ndarray]]:
    net = caffe_pb2.NetParameter()
    net.ParseFromString(Path(path).read_bytes())
    out = {}
    for layer in list(net.layer) + list(net.layers):
        if not layer.blobs:
            continue
        arrs = []
        for b in layer.blobs:
            shape = tuple(b.shape.dim) if len(b.shape.dim) else (b.num, b.channels, b.height, b.width)
            arrs.append(np.asarray(b.data, dtype=np.float32).reshape(shape))
        out[layer.name] = arrs
    return out


# --------------------------------------------------------------------------- custom ops
def _round_away(x: np.ndarray) -> np.ndarray:
    """CUDA round(): half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def resample_matrix(n_in: int, n_out: int, f_this: float, f_other: float, antialias: bool) -> np.ndarray:
    """1-D weights of caffe ResampleLayer (LINEAR -> triangle filter, kernel_width=2).

    Reproduces ``x_in = x_out * fx + fy / 2 - 0.5`` (note the fx/fy swap in the
    original kernel; ``f_other`` is the *other* axis' factor).
    """
    ax = 1.0 / f_this if antialias else 1.0
    r = 2 if f_this < 1.0 else int(math.ceil(2.0 / ax))
    W = np.zeros((n_out, n_in), dtype=np.float64)
    xo = np.arange(n_out, dtype=np.float32)
    x_in = (xo * np.float32(f_this) + np.float32(f_other) / 2.0 - 0.5).astype(np.float32)
    x_round = _round_away(x_in).astype(int)
    for o in range(n_out):
        for x in range(x_round[o] - r, x_round[o] + r + 1):
            if x < 0 or x >= n_in:
                continue
            d = ax * (x_in[o] - x)
            w = ax * max(0.0, 1.0 - abs(d)) if -1 <= d <= 1 else 0.0
            W[o, x] += w
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (W / s).astype(np.float32)


def downsample_matrix(n_in: int, n_out: int) -> np.ndarray:
    """1-D weights of caffe DownsampleLayer (align-corner triangle filter)."""
    if n_in == n_out:
        return np.eye(n_in, dtype=np.float32)
    scale = (n_in - 1) / (n_out - 1)
    rad = int(math.ceil(scale))
    W = np.zeros((n_out, n_in), dtype=np.float64)
    for o in range(n_out):
        b = np.float32(o) / np.float32(n_out - 1) * np.float32(n_in - 1)
        ib = int(_round_away(np.array(b)))
        for k in range(ib - rad, ib + rad + 1):
            if 0 <= k < n_in:
                W[o, k] = max(0.0, 1.0 - abs(k - b) / scale)
    W /= W.sum(1, keepdims=True)
    return W.astype(np.float32)


def separable_apply(x: torch.Tensor, wy: torch.Tensor, wx: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oh,nchw,pw->ncop", wy, x, wx)


def correlation(f1: torch.Tensor, f2: torch.Tensor, pad: int, max_disp: int,
                stride1: int, stride2: int, kernel_size: int = 1) -> torch.Tensor:
    """caffe CorrelationLayer (MULTIPLY), mean over kernel*kernel*C, zero padding."""
    if kernel_size != 1:
        raise NotImplementedError("only kernel_size=1 is used by PIV-LiteFlowNet")
    n, c, h, w = f1.shape
    f1p = F.pad(f1, (pad, pad, pad, pad))
    f2p = F.pad(f2, (pad, pad, pad, pad))
    hp, wp = h + 2 * pad, w + 2 * pad
    th = math.ceil((hp - 2 * max_disp) / stride1)
    tw = math.ceil((wp - 2 * max_disp) / stride1)
    rad = max_disp // stride2
    y0 = max_disp
    x0 = max_disp
    a = f1p[:, :, y0:y0 + (th - 1) * stride1 + 1:stride1, x0:x0 + (tw - 1) * stride1 + 1:stride1]
    outs = []
    for j in range(-rad, rad + 1):          # top_channel / grid_width -> y
        for i in range(-rad, rad + 1):      # top_channel % grid_width -> x
            dy, dx = j * stride2, i * stride2
            b = f2p[:, :, y0 + dy:y0 + dy + (th - 1) * stride1 + 1:stride1,
                    x0 + dx:x0 + dx + (tw - 1) * stride1 + 1:stride1]
            outs.append((a * b).sum(1))
    return torch.stack(outs, 1) / float(c)


def warp(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """caffe WarpLayer: bilinear sampling at absolute pixel coords, zero outside."""
    n, c, h, w = u.shape
    gx = 2.0 * coords[:, 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * coords[:, 1] / max(h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(u, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def pixel_grid(ref: torch.Tensor) -> torch.Tensor:
    n, _, h, w = ref.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=ref.dtype, device=ref.device),
                            torch.arange(w, dtype=ref.dtype, device=ref.device), indexing="ij")
    return torch.stack([xs, ys], 0).unsqueeze(0).expand(n, -1, -1, -1)


# --------------------------------------------------------------------------- interpreter
class CaffeGraph(torch.nn.Module):
    """Executes a parsed deploy NetParameter. Built for one (H, W) input size."""

    def __init__(self, net: caffe_pb2.NetParameter, weights: dict[str, list[np.ndarray]],
                 strict_weights: bool = True):
        super().__init__()
        self.layers = [l for l in net.layer]
        self.inputs = list(net.input)
        self.input_shapes = [tuple(s.dim) for s in net.input_shape]
        self.params = torch.nn.ParameterDict()
        self.buffers_np: dict[str, np.ndarray] = {}
        self.filled_from_filler: list[str] = []
        self.unused_weights = set(weights)
        for l in self.layers:
            key = l.name.replace(".", "_")
            if l.type in ("Convolution", "Deconvolution"):
                w, b = self._conv_params(l, weights)
                if w is None:
                    continue
                self.params[key + "__w"] = torch.nn.Parameter(torch.from_numpy(w))
                if b is not None:
                    self.params[key + "__b"] = torch.nn.Parameter(torch.from_numpy(b))
            elif l.type == "DataAugmentation":
                if l.name not in weights:
                    raise KeyError(f"mean blobs for {l.name} missing from caffemodel")
                self.register_buffer(key + "__mean", torch.from_numpy(weights[l.name][2].reshape(1, -1, 1, 1).copy()))
                self.unused_weights.discard(l.name)
        if strict_weights and not any(k.endswith("__w") for k in self.params):
            raise RuntimeError("no convolution weights loaded")
        # last use of every blob, to free memory during forward
        self.last_use: dict[str, int] = {}
        for i, l in enumerate(self.layers):
            for b in l.bottom:
                self.last_use[b] = i
        self._mats: dict = {}
        self._fill_cache: dict = {}

    def _conv_params(self, l, weights):
        p = l.convolution_param
        group = p.group or 1
        if l.name in weights:
            blobs = weights[l.name]
            self.unused_weights.discard(l.name)
            w = blobs[0]
            b = blobs[1].reshape(-1) if len(blobs) > 1 else None
            if p.bias_term and b is None:
                raise ValueError(f"{l.name}: bias expected")
            return w.copy(), (b.copy() if b is not None else None)
        # Filler (Caffe behaviour when no blob with this name exists). The input
        # channel count is only known at the first forward, so creation is deferred.
        if l.type != "Convolution":
            raise KeyError(f"{l.name}: Deconvolution weights missing in caffemodel")
        if p.weight_filler.type not in ("constant", "diagonal"):
            raise KeyError(f"{l.name}: no weights and non-deterministic filler {p.weight_filler.type}")
        self.filled_from_filler.append(l.name)
        return None, None

    def _filler_weight(self, l, cin: int, ref: torch.Tensor):
        p = l.convolution_param
        group = p.group or 1
        kh = p.kernel_h or p.kernel_size[0]
        kw = p.kernel_w or p.kernel_size[0]
        cout = p.num_output
        wf = p.weight_filler
        if wf.type == "constant":
            w = np.full((cout, cin // group, kh, kw), wf.value, np.float32)
        else:  # diagonal
            w = np.zeros((cout, cin // group, kh, kw), np.float32)
            for i, v in enumerate(wf.diag_val):
                w[i, i, kh // 2, kw // 2] = v
        b = np.full((cout,), p.bias_filler.value, np.float32) if p.bias_term else None
        conv = lambda a: None if a is None else torch.from_numpy(a).to(ref.device, ref.dtype)
        return conv(w), conv(b)

    def _mat(self, kind, *args, device=None, dtype=None):
        key = (kind,) + args + (str(device), str(dtype))
        if key not in self._mats:
            fn = resample_matrix if kind == "resample" else downsample_matrix
            self._mats[key] = torch.from_numpy(fn(*args)).to(device=device, dtype=dtype)
        return self._mats[key]

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, return_blobs: bool = False):
        blobs = {self.inputs[0]: img0, self.inputs[1]: img1}
        keep = {}
        for i, l in enumerate(self.layers):
            bot = [blobs[b] for b in l.bottom]
            out = self._run(l, bot)
            if not isinstance(out, (list, tuple)):
                out = [out]
            for name, t in zip(l.top, out):
                blobs[name] = t
            if return_blobs:
                keep.update({n: t for n, t in zip(l.top, out)})
            else:
                for b in set(l.bottom):
                    if self.last_use.get(b) == i and b not in l.top:
                        blobs.pop(b, None)
        final = blobs[self.layers[-1].top[0]]
        return (final, keep) if return_blobs else final

    def _run(self, l, bot):
        t = l.type
        key = l.name.replace(".", "_")
        if t == "Eltwise":
            p = l.eltwise_param
            op = p.operation
            if op == caffe_pb2.EltwiseParameter.SUM:
                coeff = list(p.coeff) or [1.0] * len(bot)
                out = coeff[0] * bot[0]
                for c, b in zip(coeff[1:], bot[1:]):
                    out = out + c * b
                return out
            if op == caffe_pb2.EltwiseParameter.PROD:
                out = bot[0]
                for b in bot[1:]:
                    out = out * b
                return out
            if op == caffe_pb2.EltwiseParameter.MAX:
                out = bot[0]
                for b in bot[1:]:
                    out = torch.maximum(out, b)
                return out
        if t == "DataAugmentation":
            return bot[0] - getattr(self, key + "__mean")
        if t == "Resample":
            p = l.resample_param
            x = bot[0]
            h_in, w_in = x.shape[-2:]
            h_out, w_out = p.height, p.width
            if p.type != caffe_pb2.ResampleParameter.LINEAR:
                raise NotImplementedError("only LINEAR resample used")
            fx, fy = w_in / w_out, h_in / h_out
            fx, fy = float(np.float32(fx)), float(np.float32(fy))
            aa = ((fx > 1) or (fy > 1)) and p.antialias
            wx = self._mat("resample", w_in, w_out, fx, fy, aa, device=x.device, dtype=x.dtype)
            wy = self._mat("resample", h_in, h_out, fy, fx, aa, device=x.device, dtype=x.dtype)
            return separable_apply(x, wy, wx)
        if t == "Downsample":
            x, ref = bot
            h_out, w_out = ref.shape[-2:]
            h_in, w_in = x.shape[-2:]
            if (h_in, w_in) == (h_out, w_out):
                return x
            wx = self._mat("down", w_in, w_out, device=x.device, dtype=x.dtype)
            wy = self._mat("down", h_in, h_out, device=x.device, dtype=x.dtype)
            return separable_apply(x, wy, wx)
        if t == "Convolution":
            p = l.convolution_param
            ph = p.pad_h if p.HasField("pad_h") else (p.pad[0] if p.pad else 0)
            pw = p.pad_w if p.HasField("pad_w") else (p.pad[0] if p.pad else 0)
            s = p.stride[0] if p.stride else 1
            d = p.dilation[0] if p.dilation else 1
            if key + "__w" in self.params:
                w, b = self.params[key + "__w"], self.params.get(key + "__b")
            else:  # filler weights: constant, never trained (lr_mult 0 in the original)
                if key not in self._fill_cache:
                    self._fill_cache[key] = self._filler_weight(l, bot[0].shape[1], bot[0])
                w, b = self._fill_cache[key]
            # several bottoms -> several tops with shared weights (Siamese NetC)
            outs = [F.conv2d(x, w, b, stride=s, padding=(ph, pw), dilation=d, groups=p.group or 1)
                    for x in bot]
            return outs if len(outs) > 1 else outs[0]
        if t == "Deconvolution":
            p = l.convolution_param
            pad = p.pad[0] if p.pad else 0
            s = p.stride[0] if p.stride else 1
            outs = [F.conv_transpose2d(x, self.params[key + "__w"], self.params.get(key + "__b"),
                                       stride=s, padding=pad, groups=p.group or 1) for x in bot]
            return outs if len(outs) > 1 else outs[0]
        if t == "ReLU":
            ns = l.relu_param.negative_slope
            return F.leaky_relu(bot[0], ns) if ns else F.relu(bot[0])
        if t == "Concat":
            axis = l.concat_param.axis if l.concat_param.HasField("axis") else 1
            return torch.cat(bot, axis)
        if t == "Slice":
            p = l.slice_param
            axis = p.axis if p.HasField("axis") else 1
            pts = list(p.slice_point)
            n = bot[0].shape[axis]
            if not pts:
                k = len(l.top)
                pts = [n // k * (i + 1) for i in range(k - 1)]
            edges = [0] + pts + [n]
            return [bot[0].narrow(axis, a, b - a) for a, b in zip(edges[:-1], edges[1:])]
        if t == "Correlation":
            p = l.correlation_param
            return correlation(bot[0], bot[1], p.pad, p.max_displacement,
                               p.stride_1 or 1, p.stride_2 or 1, p.kernel_size)
        if t == "Grid":
            return pixel_grid(bot[0])
        if t == "Warp":
            return warp(bot[0], bot[1])
        if t == "ChannelNorm":
            return torch.sqrt((bot[0] ** 2).sum(1, keepdim=True))
        if t == "NegSquare":
            return -(bot[0] ** 2)
        if t == "ExpMax":
            x = bot[0]
            return torch.exp(x - x.max(1, keepdim=True).values)
        if t == "Im2col":
            p = l.convolution_param
            k = p.kernel_size[0]
            pad = p.pad[0] if p.pad else 0
            s = p.stride[0] if p.stride else 1
            x = bot[0]
            n, c, h, w = x.shape
            cols = F.unfold(x, k, padding=pad, stride=s)
            ho = (h + 2 * pad - k) // s + 1
            wo = (w + 2 * pad - k) // s + 1
            return cols.view(n, c * k * k, ho, wo)
        if t == "Reduction":
            p = l.reduction_param
            axis = p.axis if p.HasField("axis") else 0
            if p.operation != caffe_pb2.ReductionParameter.MEAN:
                raise NotImplementedError
            coeff = p.coeff if p.HasField("coeff") else 1.0
            dims = tuple(range(axis, bot[0].dim()))
            return coeff * bot[0].mean(dim=dims)
        if t == "Bias":
            x, b = bot
            axis = l.bias_param.axis if l.bias_param.HasField("axis") else 1
            shape = [1] * x.dim()
            for i, s in enumerate(b.shape):
                shape[axis + i] = s
            return x + b.view(shape)
        if t == "Power":
            p = l.power_param
            power = p.power if p.HasField("power") else 1.0
            scale = p.scale if p.HasField("scale") else 1.0
            shift = p.shift if p.HasField("shift") else 0.0
            return (shift + scale * bot[0]) ** power
        raise NotImplementedError(f"layer type {t} ({l.name})")
