"""Losses use only valid elements, never produce NaN from invalid regions, all-invalid batch is finite."""
import torch

from piv.losses.supervised import charbonnier, gradient_loss
from piv.losses.temporal import temporal_derivative_loss
from piv.losses.uncertainty import gaussian_nll
from piv.metrics.velocity import vector_metrics
import numpy as np


def test_nan_in_invalid_region_is_ignored():
    pred = torch.zeros(2, 2, 8, 8, requires_grad=True)
    gt = torch.ones(2, 2, 8, 8)
    mask = torch.ones(2, 8, 8, dtype=torch.bool)
    gt[:, :, :2] = float("nan")
    mask[:, :2] = False
    for fn in (charbonnier, gradient_loss):
        l = fn(pred, gt, mask)
        l.backward()
        assert torch.isfinite(l) and torch.isfinite(pred.grad).all()
        pred.grad = None
    l = gaussian_nll(pred, torch.zeros_like(pred), gt, mask)
    assert torch.isfinite(l)


def test_reduction_by_valid_count():
    pred = torch.zeros(1, 2, 4, 4)
    gt = torch.zeros(1, 2, 4, 4); gt[:, 0, 0, 0] = 3.0
    m = torch.zeros(1, 4, 4, dtype=torch.bool); m[0, 0, 0] = True
    assert abs(charbonnier(pred, gt, m, eps=1e-6).item() - 3.0) < 1e-4


def test_all_invalid_batch():
    pred = torch.randn(1, 2, 4, 4, requires_grad=True)
    l = charbonnier(pred, torch.full_like(pred, float("nan")), torch.zeros(1, 4, 4, dtype=torch.bool))
    l.backward()
    assert l.item() == 0.0 and torch.isfinite(pred.grad).all()


def test_gradient_loss_targets_gradient_difference_not_magnitude():
    x = torch.arange(8.0).view(1, 1, 1, 8).expand(1, 2, 8, 8).clone()
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    assert gradient_loss(x + 5.0, x, m).item() < 1e-6        # same gradients, offset -> 0
    assert gradient_loss(x, x, m).item() < 1e-6


def test_temporal_loss_not_freezing():
    gt = torch.arange(5.0).view(1, 5, 1, 1, 1).expand(1, 5, 2, 4, 4).clone()
    frozen = torch.zeros_like(gt)
    m = torch.ones(1, 5, 4, 4, dtype=torch.bool)
    dtp = torch.full((1, 5), 1e-3)
    assert temporal_derivative_loss(gt, gt, m, dtp, 1e-3).item() < 1e-6
    assert temporal_derivative_loss(frozen, gt, m, dtp, 1e-3).item() > 0.5


def test_metrics_respect_mask():
    p = np.zeros((1, 2, 4, 4)); g = np.zeros((1, 2, 4, 4)); g[0, 0, 0, 0] = 10
    m = np.ones((1, 4, 4), bool); m[0, 0, 0] = False
    assert vector_metrics(p, g, m)["epe_mean"] == 0.0
