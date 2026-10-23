"""Analytic flows: divergence-free, RK4 convergence, Lagrangian GT != u(t)*dt for strong unsteadiness."""
import math

import numpy as np

from piv.data.synthetic import (CellularFlow, OrbitingVortexFlow, ShearFlow, VortexStreetFlow, converged_advect,
                                gt_displacement)


def divergence(flow, t=0.013, h=1e-3):
    ys, xs = np.mgrid[5:60:3, 5:60:3].astype(float)
    u1, _ = flow.velocity(xs + h, ys, t); u0, _ = flow.velocity(xs - h, ys, t)
    _, v1 = flow.velocity(xs, ys + h, t); _, v0 = flow.velocity(xs, ys - h, t)
    return (u1 - u0) / (2 * h) + (v1 - v0) / (2 * h), np.abs(flow.velocity(xs, ys, t)[0]).max()


def test_flows_are_divergence_free():
    for f in (CellularFlow(A0=40000, Lx=64, Ly=80, amps=(0.4,), freqs=(60,), phases=(0,)),
              VortexStreetFlow(y_c=32, x_end=200), OrbitingVortexFlow()):
        div, umax = divergence(f)
        assert np.abs(div).max() < 1e-5 * max(umax, 1)


def test_shear_exact_displacement():
    d, n, err = gt_displacement(ShearFlow(gamma=0.5, y0=8), 16, 16, 0, 2.0)
    ys = np.arange(16)[:, None]
    assert np.allclose(d[0], 0.5 * (ys - 8) * 2.0, atol=1e-5) and np.allclose(d[1], 0)


def test_rk4_converges_on_vortex_street():
    f = VortexStreetFlow(y_c=32, x_end=200)
    xs, ys = np.meshgrid(np.arange(0, 64, 4.0), np.arange(0, 64, 4.0))
    x1, y1, n, err = converged_advect(f, xs, ys, 0.010, 0.011)
    assert err < 1e-4


def test_lagrangian_vs_eulerian_estimate():
    """For a strongly unsteady flow the midpoint u(t_mid)*dt differs from the trajectory integral;
    the generator uses the integral."""
    f = CellularFlow(A0=200000, Lx=64, Ly=64, amps=(0.9,), freqs=(250,), phases=(0,))
    d, _, _ = gt_displacement(f, 32, 32, 0.0, 2e-3)
    ys, xs = np.mgrid[0:32, 0:32].astype(float)
    u, v = f.velocity(xs, ys, 1e-3)
    diff = np.hypot(d[0] - u * 2e-3, d[1] - v * 2e-3)
    assert diff.max() > 0.05        # non-negligible: u*dt would be a biased GT here
