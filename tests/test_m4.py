"""Calibrators recover known parameters; Learn-then-Test keeps its (1 - delta) promise in simulation."""
import numpy as np
import pytest
from scipy.special import expit

from qeccal.calibration import CALIBRATORS, ece, score_from_q
from qeccal.guarantees import KEEP_GRID, learn_then_test, p_value, start_keep, thresholds


def synthetic(n, T=2.0, b=0.0, seed=0):
    """Scores s with true P(wrong | s) = sigmoid(-(s / T) + b): raw sigmoid(-s) is overconfident when T > 1."""
    rng = np.random.default_rng(seed)
    s = rng.gamma(2.0, 3.0, n)
    return s, rng.random(n) < expit(-s / T + b)


def test_calibrators_recover_parameters():
    s, y = synthetic(200_000, T=2.0)
    assert CALIBRATORS["temperature"]().fit(s, y).T == pytest.approx(2.0, rel=0.05)
    pl = CALIBRATORS["platt"]().fit(s, y)
    assert pl.a == pytest.approx(0.5, rel=0.05) and abs(pl.b) < 0.05
    iso = CALIBRATORS["isotonic"]().fit(s, y)
    grid = np.linspace(0, 30, 50)
    assert np.all(np.diff(iso.predict(grid)) <= 1e-12)          # non-increasing in s
    s2, y2 = synthetic(100_000, T=2.0, seed=1)
    raw = ece(CALIBRATORS["raw"]().predict(s2), y2)
    for name in ("temperature", "platt", "isotonic"):
        assert ece(CALIBRATORS[name]().fit(s, y).predict(s2), y2) < raw / 3


def test_score_from_q_inverts_sigmoid():
    s = np.array([0.0, 1.0, 5.0, 30.0])
    assert np.allclose(score_from_q(expit(-s)), s)


def test_p_value_and_start():
    assert p_value(0, 0, 0.01) == 1.0
    assert p_value(1000, 0, 0.01) == pytest.approx(0.99 ** 1000)
    assert start_keep(15_000, 1e-3, 0.05) == 0.4                # n k >= ln(20) / -ln(1 - 5e-4) ~ 5990
    assert start_keep(15_000, 1e-5, 0.05) is None               # not certifiable at all with 15,000 shots


def test_ltt_coverage_in_simulation():
    """Over repeated Calibrate draws, the deployed threshold's true risk exceeds alpha at most ~delta of the time."""
    alpha, delta, reps = 0.02, 0.1, 400
    s_tr, _ = synthetic(20_000, T=1.0, b=1.5, seed=11)
    lams = thresholds(s_tr)
    s_big, y_big = synthetic(1_000_000, T=1.0, b=1.5, seed=12)   # ground truth R(lambda) for each threshold
    true_risk = np.array([y_big[s_big >= lam].mean() for lam in lams])
    violations = certified = 0
    for r in range(reps):
        s_cal, y_cal = synthetic(3_000, T=1.0, b=1.5, seed=1000 + r)
        i, _ = learn_then_test(s_cal, y_cal, lams, alpha, delta)
        if i is not None:
            certified += 1
            violations += true_risk[i] > alpha
    assert certified > reps * 0.8
    assert violations / reps <= delta + 0.03


def test_keep_grid_and_thresholds():
    s = np.arange(100.0)
    lams = thresholds(s)
    assert len(lams) == len(KEEP_GRID) and lams[-1] == -np.inf
    assert np.mean(s >= lams[list(KEEP_GRID).index(0.3)]) == pytest.approx(0.3, abs=0.02)
