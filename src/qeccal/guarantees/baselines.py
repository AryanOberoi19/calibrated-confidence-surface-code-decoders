"""Certified-threshold baselines for comparison with fixed-sequence Learn-then-Test (arXiv v1 plan, task A5).

All share Learn-then-Test's start rule (ltt.start_keep): thresholds stricter than the start are never tested.
  bonferroni  test every threshold from the start at delta / m (m = thresholds from the start on); deploy the
              loosest one certified. Needs no ordering.
  sgr         Geifman and El-Yaniv's selection with guaranteed risk: binary search over the thresholds from the
              start, ceil(log2 m) steps, each an exact binomial test at delta / ceil(log2 m); deploy the loosest
              threshold certified on the way.
  ltt_hb      fixed-sequence testing as Learn-then-Test, with the Hoeffding-Bentkus p-value (the default of the
              Learn-then-Test paper) in place of the exact binomial tail.
Shot-level functions return an index into keep_grid or None, like learn_then_test(...)[0]. The *_counts forms
work on cumulative counts n[d, i], e[d, i] and return -1 for none; importing this module registers them in
validity.RULES so the A4 harness can check them.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from .ltt import KEEP_GRID, start_keep
from .validity import RULES, band_counts


def hb_p_value(n: int, e: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H: R > alpha from e errors in n kept shots (1 if nothing is kept):
    min(exp(-n h(min(e/n, alpha), alpha)), e_const * P(Bin(n, alpha) <= e)), h the Bernoulli KL divergence."""
    return float(hb_p_values(np.array([n]), np.array([e]), alpha)[0])


def hb_p_values(n, e, alpha):
    n, e = np.asarray(n, np.float64), np.asarray(e, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(np.where(n > 0, e / np.maximum(n, 1), alpha), alpha)
        h = np.where(r > 0, r * np.log(np.maximum(r, 1e-300) / alpha), 0.0) + (1 - r) * np.log((1 - r) / (1 - alpha))
    p = np.minimum(np.exp(-n * h), np.e * binom.cdf(e, n, alpha))
    return np.where(n > 0, np.minimum(p, 1.0), 1.0)


def _start(n_cal, alpha, delta, keep_grid):
    k0 = start_keep(n_cal, alpha, delta, keep_grid)
    return None if k0 is None else int(np.argmax(np.asarray(keep_grid) >= k0 - 1e-9))


def _exact(n, e, alpha):
    return np.where(n == 0, 1.0, binom.cdf(e, n, alpha))


def bonferroni_counts(n, e, n_cal, alpha, delta, keep_grid=KEEP_GRID):
    n, e = np.atleast_2d(n), np.atleast_2d(e)
    i0 = _start(n_cal, alpha, delta, keep_grid)
    if i0 is None:
        return np.full(n.shape[0], -1)
    m = n.shape[1] - i0
    ok = _exact(n[:, i0:], e[:, i0:], alpha) <= delta / m
    last = ok.shape[1] - 1 - ok[:, ::-1].argmax(axis=1)
    return np.where(ok.any(axis=1), i0 + last, -1)


def sgr_counts(n, e, n_cal, alpha, delta, keep_grid=KEEP_GRID):
    n, e = np.atleast_2d(n), np.atleast_2d(e)
    d = n.shape[0]
    i0 = _start(n_cal, alpha, delta, keep_grid)
    if i0 is None:
        return np.full(d, -1)
    m = n.shape[1] - i0
    steps = max(1, math.ceil(math.log2(m)))
    lo, hi = np.full(d, i0 - 1), np.full(d, n.shape[1])   # lo: loosest certified so far; hi: failed
    rows = np.arange(d)
    for _ in range(steps):
        live = hi - lo > 1
        mid = (lo + hi) // 2
        j = np.clip(mid, i0, n.shape[1] - 1)
        ok = _exact(n[rows, j], e[rows, j], alpha) <= delta / steps
        lo = np.where(live & ok, mid, lo)
        hi = np.where(live & ~ok, mid, hi)
    return np.where(lo >= i0, lo, -1)


def ltt_hb_counts(n, e, n_cal, alpha, delta, keep_grid=KEEP_GRID):
    n, e = np.atleast_2d(n), np.atleast_2d(e)
    out = np.full(n.shape[0], -1)
    i0 = _start(n_cal, alpha, delta, keep_grid)
    if i0 is None:
        return out
    fail = hb_p_values(n[:, i0:], e[:, i0:], alpha) > delta
    first = np.where(fail.any(axis=1), fail.argmax(axis=1), fail.shape[1])
    return np.where(first > 0, i0 + first - 1, -1)


BASELINE_RULES = {"bonferroni": bonferroni_counts, "sgr": sgr_counts, "ltt_hb": ltt_hb_counts}
RULES.update(BASELINE_RULES)


def _shot(rule):
    def f(s_cal, wrong_cal, lams, alpha, delta, keep_grid=KEEP_GRID):
        nb, eb = band_counts(s_cal, wrong_cal, lams)
        m = len(lams)
        i = int(rule(np.cumsum(nb)[None, :m], np.cumsum(eb)[None, :m], len(s_cal), alpha, delta, keep_grid)[0])
        return None if i < 0 else i
    f.__name__ = rule.__name__.removesuffix("_counts")
    f.__doc__ = f"Shot-level {f.__name__}: index into keep_grid of the deployed threshold, or None."
    return f


bonferroni = _shot(bonferroni_counts)
sgr = _shot(sgr_counts)
ltt_hb = _shot(ltt_hb_counts)
