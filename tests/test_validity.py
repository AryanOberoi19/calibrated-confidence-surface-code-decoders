"""Validity of threshold rules over repeated calibration draws (plan A4).

Populations are given as band counts: band 0 holds the shots kept by the strictest threshold, band j the shots
first kept at threshold j. Each population below has 20 thresholds keeping 5%, 10%, ..., 100% of 30,000 shots,
with a chosen risk (error rate among kept shots) at each threshold."""
import numpy as np

from qeccal.guarantees import KEEP_GRID, evaluate, learn_then_test, loosest_where, thresholds
from qeccal.guarantees.validity import (band_counts, ltt_counts, plugin_counts, population_risk,
                                        violation_rate)

N_POP, N_CAL, ALPHA, DELTA, DRAWS = 30000, 15000, 0.01, 0.05, 2000


def population(risk):
    """Band counts whose cumulative risk at threshold i is risk[i] (rounded to whole errors)."""
    n_cum = np.round(KEEP_GRID * N_POP).astype(int)
    e_cum = np.round(np.asarray(risk) * n_cum).astype(int)
    n_band = np.diff(np.concatenate([[0], n_cum, [N_POP]]))
    e_band = np.diff(np.concatenate([[0], e_cum, [e_cum[-1]]]))
    assert (e_band >= 0).all() and (e_band <= n_band).all()
    return n_band, e_band


def rate(res, key):
    return res[key] / res["draws"]


def test_ltt_holds_on_a_rising_risk_curve():
    n, e = population(np.linspace(0.002, 0.05, 20))
    res = violation_rate(n, e, ALPHA, DELTA, N_CAL, DRAWS, np.random.default_rng([20261117, 30]), rule="ltt")
    assert res["draws"] == DRAWS and res["certified"] > 0
    assert rate(res, "violations") <= 0.065


def test_harness_catches_an_invalid_rule():
    risk = np.where(np.arange(20) < 9, 0.5 * ALPHA, 2 * ALPHA)
    risk[9] = 1.02 * ALPHA
    n, e = population(risk)
    rng = np.random.default_rng([20261117, 31])
    assert rate(violation_rate(n, e, ALPHA, DELTA, N_CAL, DRAWS, rng, rule="plugin"), "violations") > 0.25
    assert rate(violation_rate(n, e, ALPHA, DELTA, N_CAL, DRAWS, rng, rule="ltt"), "violations") <= 0.065


def test_ltt_rarely_certifies_when_every_threshold_is_bad():
    n, e = population(np.full(20, 1.1 * ALPHA))
    res = violation_rate(n, e, ALPHA, DELTA, N_CAL, DRAWS, np.random.default_rng([20261117, 32]), rule="ltt")
    assert rate(res, "certified") <= 0.065


def test_band_counts_reproduce_evaluate():
    rng = np.random.default_rng([20261117, 33])
    s = np.round(rng.gamma(2.0, 3.0, 20000), 1)                  # rounded: ties at thresholds
    wrong = rng.random(20000) < 1 / (1 + np.exp(s / 2))
    lams = thresholds(s[:8000])
    nb, eb = band_counts(s[8000:], wrong[8000:], lams)
    assert nb.size == len(lams) + 1 and nb.sum() == 12000
    n_cum, e_cum = np.cumsum(nb)[: len(lams)], np.cumsum(eb)[: len(lams)]
    for i, lam in enumerate(lams):
        ev = evaluate(lam, s[8000:], wrong[8000:], ALPHA)
        assert (ev["n"], ev["errors"]) == (n_cum[i], e_cum[i])
    r = population_risk(nb, eb)
    assert np.allclose(r[n_cum > 0], e_cum[n_cum > 0] / n_cum[n_cum > 0])


def test_count_rules_match_shot_rules():
    rng = np.random.default_rng([20261117, 34])
    for t in range(40):
        s = np.round(rng.gamma(2.0, 3.0, 35000), 1)
        wrong = rng.random(35000) < 1 / (1 + np.exp(s / 2 - 1))
        lams = thresholds(s[:20000])
        s_ca, w_ca = s[20000:], wrong[20000:]
        nb, eb = band_counts(s_ca, w_ca, lams)
        n, e = np.cumsum(nb)[None, : len(lams)], np.cumsum(eb)[None, : len(lams)]
        for alpha in (1e-3, 1e-2, 3e-2, 1e-1):
            i = learn_then_test(s_ca, w_ca, lams, alpha, DELTA)[0]
            assert ltt_counts(n, e, s_ca.size, alpha, DELTA)[0] == (-1 if i is None else i)
            err = [w_ca[s_ca >= lam].mean() if (s_ca >= lam).any() else np.inf for lam in lams]
            j = loosest_where(err, lambda v: v <= alpha)
            assert plugin_counts(n, e, s_ca.size, alpha, DELTA)[0] == (-1 if j is None else j)
