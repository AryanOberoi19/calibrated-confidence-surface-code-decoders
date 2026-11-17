"""Shift statistics that respect dependence between source-target pairs (arXiv v1 plan, task A6).

The M5 drift table counts pairs, but pairs that share a source (or a target) patch are not independent, so a
share of failing pairs is not a binomial rate. These functions give each pair a proper p-value, control the
false discovery rate across pairs, put a cluster-bootstrap interval on a share, and test the exchangeability
assumption directly, without any threshold."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom

from ..guarantees.ltt import thresholds


def pair_p_value(n: int, e: int, alpha: float) -> float:
    """P(Bin(n, alpha) >= e): p-value for H: the target's risk at the transferred threshold is at most alpha."""
    return 1.0 if n == 0 else float(binom.sf(e - 1, n, alpha))


def bh(pvals, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg rejections at level q, in the order of pvals."""
    p = np.asarray(pvals, np.float64)
    out = np.zeros(p.size, bool)
    if p.size == 0:
        return out
    order = np.argsort(p, kind="stable")
    ok = np.flatnonzero(p[order] <= q * np.arange(1, p.size + 1) / p.size)
    if ok.size:
        out[order[: ok[-1] + 1]] = True
    return out


def cluster_bootstrap_share(fail, source_patch, target_patch, n_boot: int, rng) -> tuple[float, float, float]:
    """Share of failing pairs and a 90% interval from resampling source patches and target patches with
    replacement (two-way: a pair's weight is the product of its source's and its target's resample counts).
    When every pair's source cluster equals its target cluster, it resamples those clusters once (one-way)."""
    fail = np.asarray(fail, np.float64)
    src, tgt = np.asarray(source_patch), np.asarray(target_patch)
    share = float(fail.mean())

    def counts(labels):
        u, inv = np.unique(labels, return_inverse=True)
        c = rng.multinomial(u.size, np.full(u.size, 1 / u.size), size=n_boot)
        return c[:, inv]

    w = counts(src) if np.array_equal(src, tgt) else counts(src) * counts(tgt)
    tot = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (w * fail).sum(axis=1) / tot
    b = b[tot > 0]
    return share, float(np.percentile(b, 5)), float(np.percentile(b, 95))


def _max_z(ca, tot):
    """Max over thresholds of |rate_a - rate_b| / pooled standard error, from per-(band, wrong) counts of A."""
    na, ea = ca[:, 0::2] + ca[:, 1::2], ca[:, 1::2]
    tn, te = tot[0::2] + tot[1::2], tot[1::2]
    m = na.shape[1] - 1
    na, ea = np.cumsum(na, axis=1)[:, :m], np.cumsum(ea, axis=1)[:, :m]
    tn, te = np.cumsum(tn)[:m], np.cumsum(te)[:m]
    nb, eb = tn - na, te - ea
    with np.errstate(invalid="ignore", divide="ignore"):
        r = te / tn
        se = np.sqrt(r * (1 - r) * (1 / na + 1 / nb))
        z = np.abs(ea / na - eb / nb) / se
    z = np.where((na > 0) & (nb > 0) & (se > 0), z, 0.0)
    return z.max(axis=1)


def exchangeability_test(s_a, wrong_a, s_b, wrong_b, keep_grid, n_perm: int, rng) -> float:
    """Permutation p-value for "the risk-versus-keep-fraction curves of A and B are the same". Thresholds keep the
    grid's fractions of the pooled scores; the statistic is the largest standardised difference in error rate among
    kept shots. Permuting the A/B labels is done exactly by drawing A's counts per (score band, error) cell from the
    multivariate hypergeometric distribution."""
    s = np.concatenate([np.asarray(s_a, np.float64), np.asarray(s_b, np.float64)])
    w = np.concatenate([np.asarray(wrong_a, bool), np.asarray(wrong_b, bool)])
    lams = thresholds(s, keep_grid)
    band = np.searchsorted(-lams, -s, side="left")
    cell = 2 * band + w
    k = 2 * (len(lams) + 1)
    na = len(s_a)
    tot = np.bincount(cell, minlength=k)
    obs = _max_z(np.bincount(cell[:na], minlength=k)[None], tot)[0]
    perm = _max_z(rng.multivariate_hypergeometric(tot, na, size=n_perm), tot)
    return (1 + int((perm >= obs - 1e-12 * max(obs, 1)).sum())) / (1 + n_perm)
