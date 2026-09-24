"""Calibration and selective-risk metrics for q = a method's probability that its own prediction is wrong
(spec sections 8.3 and 8.5). `wrong` is 1[prediction != actual observable flip]."""
from __future__ import annotations

import numpy as np


def _order_most_doubtful_first(q, rng):
    """Indices sorted by q descending; ties broken at random (spec 8.5: committed seed via rng)."""
    q = np.asarray(q, dtype=np.float64)
    return np.lexsort((rng.random(q.size), -q))


def error_vs_discard(q, wrong, discard_fracs, rng):
    """Error rate among kept shots after discarding the given fractions of most doubtful shots.
    Returns (kept counts, errors among kept) as int arrays."""
    wrong = np.asarray(wrong, dtype=bool)
    order = _order_most_doubtful_first(q, rng)
    n = wrong.size
    csum = np.concatenate([[0], np.cumsum(wrong[order][::-1])])   # errors among the k least doubtful
    kept = np.array([n - int(round(f * n)) for f in discard_fracs])
    return kept, csum[kept]


def reliability(q, wrong, n_bins=10, rng=None):
    """Equal-mass bins by q: (mean q, observed error rate, count) per bin."""
    rng = rng or np.random.default_rng(0)
    q = np.asarray(q, dtype=np.float64)
    wrong = np.asarray(wrong, dtype=bool)
    order = _order_most_doubtful_first(q, rng)[::-1]
    bins = np.array_split(order, n_bins)
    return (np.array([q[b].mean() for b in bins]), np.array([wrong[b].mean() for b in bins]),
            np.array([b.size for b in bins]))


def ece(q, wrong, n_bins=10, rng=None):
    mq, mw, cnt = reliability(q, wrong, n_bins, rng)
    return float(np.sum(cnt * np.abs(mq - mw)) / cnt.sum())


def brier(q, wrong):
    return float(np.mean((np.asarray(q, dtype=np.float64) - np.asarray(wrong, dtype=np.float64)) ** 2))


def nll(q, wrong, eps=1e-12):
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1 - eps)
    w = np.asarray(wrong, dtype=bool)
    return float(-np.mean(np.where(w, np.log(q), np.log1p(-q))))
