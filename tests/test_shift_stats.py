"""Shift statistics that respect dependence between pairs (plan A6)."""
import numpy as np
from scipy.stats import binom

from qeccal.guarantees import KEEP_GRID
from qeccal.shift.stats import bh, cluster_bootstrap_share, exchangeability_test, pair_p_value


def sample(rng, n, scale=1.0):
    s = np.round(rng.gamma(2.0, 3.0, n), 2)
    p = np.minimum(1.0, scale / (1 + np.exp(s / 2 - 1)))
    return s, rng.random(n) < p


def test_exchangeability_test_is_calibrated():
    rng = np.random.default_rng([20261117, 50])
    p = [exchangeability_test(*sample(rng, 2000), *sample(rng, 2000), KEEP_GRID, 200, rng) for _ in range(200)]
    assert 0.02 <= np.mean(np.array(p) < 0.05) <= 0.09


def test_exchangeability_test_detects_a_higher_error_rate():
    rng = np.random.default_rng([20261117, 51])
    p = [exchangeability_test(*sample(rng, 2000), *sample(rng, 2000, 1.5), KEEP_GRID, 200, rng) for _ in range(200)]
    assert np.mean(np.array(p) < 0.05) > 0.9


def test_pair_p_value_is_the_binomial_upper_tail():
    assert pair_p_value(1000, 10, 0.01) == binom.sf(9, 1000, 0.01)
    assert pair_p_value(1000, 0, 0.01) == 1.0
    assert pair_p_value(0, 0, 0.01) == 1.0


def test_bh_on_a_known_case():
    assert bh(np.array([0.01, 0.02, 0.03, 0.5]), 0.05).tolist() == [True, True, True, False]
    assert bh(np.array([0.04, 0.5, 0.03, 0.2]), 0.05).tolist() == [False, False, False, False]
    assert bh(np.array([]), 0.05).size == 0


def test_cluster_bootstrap_share():
    rng = np.random.default_rng([20261117, 52])
    src = np.repeat(np.arange(6), 5)
    tgt = np.tile(np.arange(5), 6)
    fail = rng.random(30) < 0.3
    share, lo, hi = cluster_bootstrap_share(fail, src, tgt, 500, rng)
    assert share == fail.mean() and lo <= share <= hi and lo < hi
    same = np.arange(30) % 7                                          # source cluster = target cluster: one-way
    share1, lo1, hi1 = cluster_bootstrap_share(fail, same, same, 500, rng)
    assert share1 == fail.mean() and lo1 <= share1 <= hi1
