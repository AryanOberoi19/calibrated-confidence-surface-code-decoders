"""Statistical baselines for certified thresholds (plan A5), checked on hand-computed cases.

Three thresholds keeping 20%, 60% and 100% of 1,000 Calibrate shots; alpha = 0.1, delta = 0.1, so the start
rule admits all three (m = 3), Bonferroni tests each at 0.1 / 3 and SGR runs ceil(log2 3) = 2 steps at 0.1 / 2.
Exact binomial p-values P(Bin(n, 0.1) <= e) and Hoeffding-Bentkus p-values, rounded:
  case A, errors (12, 47, 100): binomial (0.032, 0.041, 0.53), HB (0.087, 0.111, 1)
  case B, errors (13, 44, 100): binomial (0.057, 0.015, 0.53), HB (0.154, 0.040, 1)
  case C, errors (11, 48, 100): binomial (0.017, 0.055, 0.53), HB (0.046, 0.151, 1)
"""
import numpy as np
import pytest

from qeccal.guarantees import bonferroni, hb_p_value, learn_then_test, ltt_hb, p_value, sgr

GRID = np.array([0.2, 0.6, 1.0])
LAMS = np.array([3.0, 2.0, -np.inf])
ALPHA, DELTA = 0.1, 0.1


def shots(e0, e1, e2):
    """1,000 shots: 200 at score 3.5, 400 at 2.5, 400 at 1.0, with cumulative errors e0, e1, e2."""
    s = np.repeat([3.5, 2.5, 1.0], [200, 400, 400])
    w = np.zeros(1000, bool)
    w[:e0] = True
    w[200:200 + e1 - e0] = True
    w[600:600 + e2 - e1] = True
    return s, w


@pytest.mark.parametrize("errors, expect", [
    ((12, 47, 100), {"ltt": 1, "bonferroni": 0, "sgr": 1, "ltt_hb": 0}),
    ((13, 44, 100), {"ltt": 1, "bonferroni": 1, "sgr": 1, "ltt_hb": None}),
    ((11, 48, 100), {"ltt": 1, "bonferroni": 0, "sgr": 0, "ltt_hb": 0}),
])
def test_rules_on_hand_computed_cases(errors, expect):
    s, w = shots(*errors)
    got = {"ltt": learn_then_test(s, w, LAMS, ALPHA, DELTA, GRID)[0],
           "bonferroni": bonferroni(s, w, LAMS, ALPHA, DELTA, GRID),
           "sgr": sgr(s, w, LAMS, ALPHA, DELTA, GRID),
           "ltt_hb": ltt_hb(s, w, LAMS, ALPHA, DELTA, GRID)}
    assert got == expect


def test_hb_p_value_is_never_below_the_exact_binomial():
    for n in (1, 10, 100, 1000, 15000):
        for alpha in (1e-3, 1e-2, 0.1, 0.3):
            for e in sorted({0, 1, int(alpha * n / 2), int(alpha * n), int(2 * alpha * n), n}):
                assert hb_p_value(n, e, alpha) >= p_value(n, e, alpha) - 1e-12
    assert hb_p_value(0, 0, 0.1) == 1.0


def test_every_rule_returns_none_below_the_start_rule():
    rng = np.random.default_rng([20261117, 40])
    s, w = rng.random(1000), np.zeros(1000, bool)                 # alpha = 1e-3 needs ~6,000 shots
    lams = np.quantile(s, 1 - np.arange(0.05, 1.0001, 0.05))
    for rule in (bonferroni, sgr, ltt_hb):
        assert rule(s, w, lams, 1e-3, 0.05) is None
    assert learn_then_test(s, w, lams, 1e-3, 0.05)[0] is None
