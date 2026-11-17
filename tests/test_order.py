"""Shot-order diagnostics (plan A2): each test is calibrated on i.i.d. data and has power on planted structure."""
import numpy as np

from qeccal.diagnostics.order import autocorr, block_heterogeneity, block_variance_ratio, ljung_box, runs_test, trend


def test_block_heterogeneity_is_calibrated_on_iid():
    rng = np.random.default_rng([20261117, 100])
    p = [block_heterogeneity(rng.random(5000) < 0.1, 10, 200, rng) for _ in range(200)]
    assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09


def test_variance_ratio_is_one_on_iid():
    rng = np.random.default_rng([20261117, 101])
    r = [block_variance_ratio(rng.random(5000) < 0.1, 10) for _ in range(200)]
    assert 0.9 <= np.mean(r) <= 1.1


def test_ramp_is_detected_by_blocks_and_trend():
    rng = np.random.default_rng([20261117, 102])
    x = rng.random(50000) < np.linspace(0.10, 0.15, 50000)
    assert block_heterogeneity(x, 50, 2000, rng) < 0.001
    assert trend(x)[1] < 0.001
    assert block_variance_ratio(x, 50) > 2


def test_burst_is_detected_by_blocks():
    rng = np.random.default_rng([20261117, 103])
    x = rng.random(50000) < 0.1
    x[20000:20500] = rng.random(500) < 0.5
    assert block_heterogeneity(x, 50, 2000, rng) < 0.001


def test_runs_test_rejects_alternation():
    assert runs_test(np.arange(1000) % 2 == 0) < 0.001


def test_runs_test_degenerate_sequence():
    assert runs_test(np.zeros(100, bool)) == 1.0


def test_autocorr_recovers_ar1_and_ljung_box_rejects_it():
    rng = np.random.default_rng([20261117, 104])
    e = rng.standard_normal(50000)
    x = np.empty_like(e)
    x[0] = e[0]
    for t in range(1, x.size):
        x[t] = 0.5 * x[t - 1] + e[t]
    r = autocorr(x, 10)
    assert r.shape == (10,)
    assert abs(r[0] - 0.5) < 0.02 and abs(r[1] - 0.25) < 0.02
    assert ljung_box(r, x.size) < 1e-6
    iid = autocorr(rng.standard_normal(50000), 10)
    assert np.all(np.abs(iid) < 4 / np.sqrt(50000))
