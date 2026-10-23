"""Resume restores model/optimizer/scheduler/RNG/sampler: interrupted run == uninterrupted run."""
import numpy as np
import torch

from piv.engine import checkpoint as ck
from piv.models.temporal_refiner import TemporalRefiner, n_params, matched_spatial
from piv.losses.supervised import charbonnier


def _run(steps, model, opt, sched, rng):
    losses = []
    for _ in range(steps):
        F = torch.from_numpy(rng.normal(size=(2, 5, 2, 16, 16))).float()
        A = torch.from_numpy(rng.uniform(0, 255, (2, 1, 16, 16))).float()
        G = F[:, 2] + 0.1
        M = torch.ones(2, 5)
        loss = charbonnier(model(F, A, A, M), G, torch.ones(2, 16, 16, dtype=torch.bool)) + 1e-3 * torch.rand(1).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    return losses


def _fresh(seed=0):
    torch.manual_seed(seed)
    m = TemporalRefiner(5, 8)
    o = torch.optim.Adam(m.parameters(), 1e-3)
    s = torch.optim.lr_scheduler.StepLR(o, 2, 0.5)
    return m, o, s


def test_resume_bitwise(tmp_path):
    m, o, s = _fresh(); rng = np.random.default_rng(5); torch.manual_seed(1)
    full = _run(6, m, o, s, rng)
    m, o, s = _fresh(); rng = np.random.default_rng(5); torch.manual_seed(1)
    first = _run(3, m, o, s, rng)
    ck.save(tmp_path / "c.pt", model_state=m.state_dict(), optimizer=o, scheduler=s, step=3, sampler_rng=rng)
    m2, o2, s2 = _fresh(99); rng2 = np.random.default_rng(123)
    st = ck.load(tmp_path / "c.pt", model=m2, optimizer=o2, scheduler=s2, sampler_rng=rng2)
    assert st["step"] == 3
    second = _run(3, m2, o2, s2, rng2)
    assert np.allclose(first + second, full, rtol=0, atol=1e-6)


def test_refiner_identity_at_init_and_finite_grad():
    m = TemporalRefiner(5, 8)
    F = torch.randn(2, 5, 2, 16, 16)
    A = torch.rand(2, 1, 16, 16) * 255
    out = m(F, A, A, torch.ones(2, 5))
    assert torch.allclose(out, F[:, 2])              # zero-initialised residual head
    loss = (out - 1).abs().mean(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_spatial_ablation_param_budget():
    t = TemporalRefiner(5, 32)
    s = matched_spatial(t, 32)
    assert n_params(s) >= n_params(t) and n_params(s) < 1.15 * n_params(t)
