"""Calibration budget (arXiv v1 plan, task A8): how many labelled shots a certificate needs.

A threshold whose true risk among kept shots is R is certified at level alpha when the binomial p-value
P(Bin(m, alpha) <= e) is at most delta, with m kept calibration shots and e errors among them. The functions
give the number of kept shots m; divide by the keep fraction for the calibration-set size. They consider one
threshold on its own: fixed-sequence Learn-then-Test must also pass the stricter thresholds from its start rule,
so for it these numbers are lower bounds.
"""
from __future__ import annotations

import math

from scipy.stats import binom


def zero_error_kept(alpha: float, delta: float) -> int:
    """Smallest m with (1 - alpha)^m <= delta: kept shots needed when no kept shot is an error."""
    return math.ceil(math.log(1 / delta) / -math.log1p(-alpha) - 1e-9)


def max_certifiable_errors(m: int, alpha: float, delta: float) -> int:
    """Largest e with P(Bin(m, alpha) <= e) <= delta, or -1 when even e = 0 does not certify."""
    if m <= 0:
        return -1
    e = int(binom.ppf(delta, m, alpha))
    while e >= 0 and binom.cdf(e, m, alpha) > delta:
        e -= 1
    while binom.cdf(e + 1, m, alpha) <= delta:
        e += 1
    return e


def _ok(m, risk, alpha, delta, prob):
    e = max_certifiable_errors(m, alpha, delta)
    return e >= 0 and binom.cdf(e, m, risk) >= prob


def kept_needed(risk: float, alpha: float, delta: float, prob: float = 0.5, cap: float = 1e13) -> int | None:
    """Smallest m (found by doubling then bisection; the binomial sawtooth can make it a few shots off) such that a
    threshold with true risk `risk` is certified with probability at least `prob`. None if risk >= alpha or the
    answer exceeds `cap`."""
    if risk >= alpha:
        return None
    lo = zero_error_kept(alpha, delta)
    if risk <= 0:
        return lo
    hi = lo
    while not _ok(hi, risk, alpha, delta, prob):
        lo, hi = hi, hi * 2
        if hi > cap:
            return None
    while hi - lo > max(1, hi // 10000):
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if _ok(mid, risk, alpha, delta, prob) else (mid, hi)
    return hi
