"""Validity of threshold rules over repeated calibration draws (arXiv v1 plan, task A4).

The guarantee of Learn-then-Test is a statement over the draw of the Calibrate set: P(R(lambda_hat) <= alpha)
>= 1 - delta. To check it, a finite population of shots (with frozen scores, labels and Train thresholds) is
summarised by band counts, calibration sets are drawn from it, and every rule is applied to each draw. Because
the population risk at every threshold is known exactly, the share of draws whose selected threshold has risk
above alpha is the violation probability itself.

Rules work on cumulative counts so that thousands of draws are cheap: n[d, i], e[d, i] are the kept shots and
errors at threshold i in draw d. Each returns, per draw, the index of the selected threshold or -1 for none.
They are tested for equality with the shot-level rules (ltt.learn_then_test, ltt.loosest_where).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from .ltt import KEEP_GRID, start_keep


def band_counts(s, wrong, lams) -> tuple[np.ndarray, np.ndarray]:
    """Shots and errors per score band (length len(lams) + 1). Thresholds run from strict to loose; a shot is
    kept at threshold i if s >= lams[i]; band b holds the shots first kept at threshold b (band len(lams): never)."""
    s = np.asarray(s, np.float64)
    neg = -np.asarray(lams, np.float64)                    # non-decreasing
    band = np.searchsorted(neg, -s, side="left")           # number of thresholds with s < lam
    m = len(lams) + 1
    return (np.bincount(band, minlength=m).astype(np.int64),
            np.bincount(band, weights=np.asarray(wrong, np.float64), minlength=m).round().astype(np.int64))


def population_risk(n_band, e_band) -> np.ndarray:
    """Error rate among kept shots at each threshold (NaN where nothing is kept)."""
    m = len(n_band) - 1
    n, e = np.cumsum(n_band)[:m], np.cumsum(e_band)[:m]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, e / np.maximum(n, 1), np.nan)


def ltt_counts(n, e, n_cal, alpha, delta, keep_grid=KEEP_GRID) -> np.ndarray:
    """Fixed-sequence Learn-then-Test on cumulative counts; identical to ltt.learn_then_test."""
    n, e = np.atleast_2d(n), np.atleast_2d(e)
    out = np.full(n.shape[0], -1)
    k0 = start_keep(n_cal, alpha, delta, keep_grid)
    if k0 is None:
        return out
    i0 = int(np.argmax(np.asarray(keep_grid) >= k0 - 1e-9))
    p = np.where(n[:, i0:] == 0, 1.0, binom.cdf(e[:, i0:], n[:, i0:], alpha))
    fail = p > delta
    first = np.where(fail.any(axis=1), fail.argmax(axis=1), fail.shape[1])
    return np.where(first > 0, i0 + first - 1, -1)


def plugin_counts(n, e, n_cal, alpha, delta, keep_grid=KEEP_GRID) -> np.ndarray:
    """Loosest threshold whose Calibrate error among kept shots is <= alpha; identical to M4's plug-in rule."""
    n, e = np.atleast_2d(n), np.atleast_2d(e)
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = (n > 0) & (e / np.maximum(n, 1) <= alpha)
    last = ok.shape[1] - 1 - ok[:, ::-1].argmax(axis=1)
    return np.where(ok.any(axis=1), last, -1)


RULES = {"ltt": ltt_counts, "plugin": plugin_counts}


def violation_rate(n_band, e_band, alpha, delta, n_cal, n_draws, rng, rule="ltt", keep_grid=KEEP_GRID) -> dict:
    """Draw n_draws calibration sets of n_cal shots i.i.d. from the population given by band counts, apply the rule,
    and count draws that select a threshold (certified) and draws whose selected threshold has population risk
    above alpha (violations)."""
    n_band, e_band = np.asarray(n_band, np.int64), np.asarray(e_band, np.int64)
    b = n_band.size
    m = b - 1
    risk = population_risk(n_band, e_band)
    pv = np.concatenate([n_band - e_band, e_band]).astype(np.float64)
    c = rng.multinomial(n_cal, pv / pv.sum(), size=n_draws)
    nb, eb = c[:, :b] + c[:, b:], c[:, b:]
    n, e = np.cumsum(nb, axis=1)[:, :m], np.cumsum(eb, axis=1)[:, :m]
    idx = RULES[rule](n, e, n_cal, alpha, delta, keep_grid)
    cert = idx >= 0
    viol = cert & (risk[np.clip(idx, 0, m - 1)] > alpha)
    return {"draws": int(n_draws), "certified": int(cert.sum()), "violations": int(viol.sum())}
