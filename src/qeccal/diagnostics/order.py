"""Tests for structure in the order of shots within one experiment file (plan A2).

The binomial p-value of Learn-then-Test assumes the shots are independent. These functions look for the
two ways file order could break that: slow drift (block means vary more than a random order would give,
or a monotone trend) and short-range dependence (autocorrelation, runs). All take a 1-D sequence in file
order. P-values from permutations use the (1 + count) / (1 + n_perm) convention, so they are never 0.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm, spearmanr


def _blocks(x, n_blocks):
    x = np.asarray(x, np.float64)
    m = x.size // n_blocks
    if m < 1:
        raise ValueError(f"{x.size} values cannot fill {n_blocks} blocks")
    return x[: m * n_blocks], m


def block_variance_ratio(x, n_blocks: int) -> float:
    """Variance of the block means divided by its expectation under a uniformly random order (1 = no excess).

    Under a random permutation of N = n_blocks * m values with population variance s2, each block mean has
    variance s2 / m * (N - m) / (N - 1); the block means average to the overall mean, so that is also the
    expected (ddof = 0) variance across blocks."""
    x, m = _blocks(x, n_blocks)
    n = x.size
    s2 = x.var()
    if s2 == 0 or n == m:
        return float("nan")
    expect = s2 / m * (n - m) / (n - 1)
    return float(x.reshape(n_blocks, m).mean(axis=1).var() / expect)


def block_heterogeneity(x, n_blocks: int, n_perm: int, rng, batch: int = 100) -> float:
    """Permutation p-value for "block means of x vary more than a random ordering would give".
    Statistic: variance of the n_blocks block means (consecutive blocks of equal size, remainder dropped)."""
    x, m = _blocks(x, n_blocks)
    obs = x.reshape(n_blocks, m).mean(axis=1).var()
    tol = 1e-12 * max(obs, 1e-300)
    hits, done = 0, 0
    while done < n_perm:
        b = min(batch, n_perm - done)
        perm = rng.permuted(np.broadcast_to(x, (b, x.size)), axis=1)
        stat = perm.reshape(b, n_blocks, m).mean(axis=2).var(axis=1)
        hits += int((stat >= obs - tol).sum())
        done += b
    return (1 + hits) / (1 + n_perm)


def trend(x) -> tuple[float, float]:
    """Spearman correlation of x with shot index, and its two-sided p-value."""
    x = np.asarray(x, np.float64)
    if x.var() == 0:
        return float("nan"), 1.0
    r = spearmanr(np.arange(x.size), x)
    return float(r.statistic), float(r.pvalue)


def autocorr(x, max_lag: int) -> np.ndarray:
    """Sample autocorrelation at lags 1..max_lag (denominator: the lag-0 sum of squares)."""
    x = np.asarray(x, np.float64)
    d = x - x.mean()
    den = float(d @ d)
    if den == 0:
        return np.full(max_lag, np.nan)
    return np.array([float(d[:-k] @ d[k:]) / den for k in range(1, max_lag + 1)])


def ljung_box(r, n: int) -> float:
    """Ljung-Box p-value for "no autocorrelation at lags 1..len(r)", from sample autocorrelations r of n values."""
    r = np.asarray(r, np.float64)
    if np.isnan(r).any():
        return 1.0
    k = np.arange(1, r.size + 1)
    q = n * (n + 2) * float(np.sum(r**2 / (n - k)))
    return float(chi2.sf(q, r.size))


def runs_test(b) -> float:
    """Wald-Wolfowitz two-sided p-value (normal approximation) for a binary sequence; 1.0 if it is constant."""
    b = np.asarray(b, bool)
    n1 = int(b.sum())
    n2 = b.size - n1
    if n1 == 0 or n2 == 0:
        return 1.0
    n = n1 + n2
    runs = 1 + int(np.count_nonzero(b[1:] != b[:-1]))
    mu = 2.0 * n1 * n2 / n + 1
    var = 2.0 * n1 * n2 * (2.0 * n1 * n2 - n) / (n * n * (n - 1))
    return float(2 * norm.sf(abs(runs - mu) / np.sqrt(var)))
