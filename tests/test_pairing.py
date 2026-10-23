import numpy as np
import pytest

from piv.data import pairing
from piv.data.synthetic import SequenceSpec, UniformFlow, render_sequence
from piv.data.dataset import context_indices


def test_continuous_pairs():
    t = np.arange(6) * 2e-3
    ia, ib, ta, tb = pairing.continuous_pairs(t)
    assert list(ia) == [0, 1, 2, 3, 4] and list(ib) == [1, 2, 3, 4, 5]
    assert np.allclose(pairing.dt_pair(ta, tb), 2e-3)
    assert np.isclose(pairing.dt_sample(ta, tb), 2e-3)


def test_double_frame_dt_pair_differs_from_dt_sample():
    ta = np.arange(5) * 1e-2
    tb = ta + 1e-4
    ia, ib, a, b = pairing.double_frame_pairs(ta, tb)
    assert np.allclose(pairing.dt_pair(a, b), 1e-4)
    assert np.isclose(pairing.dt_sample(a, b), 1e-2)
    assert np.allclose(pairing.velocity_times(a, b), ta + 5e-5)


def test_no_cross_pairs_between_pulses():
    ta = np.array([0.0, 1e-3]); tb = np.array([1.2e-3, 2e-3])       # B0 after A1 -> overlap
    with pytest.raises(ValueError):
        pairing.double_frame_pairs(ta, tb)


def test_nonincreasing_timestamps_rejected():
    with pytest.raises(ValueError):
        pairing.continuous_pairs(np.array([0, 1, 1, 2.0]))


def test_double_frame_render_gt_uses_dt_pair():
    spec = SequenceSpec(h=32, w=32, n_frames=3, fs=100.0, mode="double_frame", dt_pair=1e-3)
    r = render_sequence(UniformFlow(u=2000.0, v=0.0), spec, seed=0)
    assert np.allclose(r["gt"][:, 0], 2.0, atol=1e-5)           # 2000 px/s * 1 ms
    assert np.allclose(np.diff(r["t_a"]), 1e-2)
    assert r["images"][0].shape == (3, 32, 32)


def test_context_window_edges():
    idx, m = context_indices(0, 10, 5, "mask")
    assert list(m) == [0, 0, 1, 1, 1] and list(idx) == [0, 0, 0, 1, 2]
    idx, m = context_indices(5, 10, 5, "mask")
    assert m.all() and list(idx) == [3, 4, 5, 6, 7]
    idx, m = context_indices(9, 10, 5, "replicate")
    assert list(idx) == [7, 8, 9, 9, 9] and m.all()
