"""Learn-then-Test threshold selection with fixed-sequence testing (spec section 8.6).

Target: P( R(lambda_hat) <= alpha ) >= 1 - delta over the draw of the Calibrate split, where R(lambda) is the
logical error rate among shots kept at threshold lambda (keep a shot if its score s >= lambda) for one
experiment.

1. Thresholds come from Train scores: lambda_k keeps a fraction k of Train shots, for k on a fixed grid.
2. The sequence runs from strict (small k) to loose (large k) and starts at the smallest k whose Calibrate
   count could certify alpha at all (n_k >= ln(1/delta) / -ln(1 - alpha / 2)); a start with no power would
   stop the sequence at once. The start depends only on alpha, delta and the Calibrate size, not on labels.
3. For each k in order: p_k = P(Bin(n_k, alpha) <= e_k), with n_k kept Calibrate shots and e_k errors
   among them. Stop at the first p_k > delta; every earlier k is certified. Deploy the loosest certified k.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, binom

KEEP_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def thresholds(s_train, keep_grid=KEEP_GRID) -> np.ndarray:
    """lambda_k such that s >= lambda_k keeps about a fraction k of Train shots (k = 1 keeps everything)."""
    s_train = np.asarray(s_train, np.float64)
    return np.array([-np.inf if k >= 1 else np.quantile(s_train, 1 - k) for k in keep_grid])


def p_value(n: int, e: int, alpha: float) -> float:
    """Binomial tail for H: R > alpha given e errors in n kept shots; 1 if nothing is kept."""
    return 1.0 if n == 0 else float(binom.cdf(e, n, alpha))


def start_keep(n_cal: int, alpha: float, delta: float, keep_grid=KEEP_GRID, margin: float = 2.0):
    """Smallest grid k with enough Calibrate shots to certify alpha with a few errors to spare, or None."""
    need = np.log(1 / delta) / -np.log1p(-alpha / margin)
    ok = np.asarray(keep_grid) * n_cal >= need
    return float(np.asarray(keep_grid)[np.argmax(ok)]) if ok.any() else None


def learn_then_test(s_cal, wrong_cal, lams, alpha, delta, keep_grid=KEEP_GRID):
    """Index into keep_grid of the loosest certified threshold (None if none), and the p-values tested."""
    s_cal = np.asarray(s_cal, np.float64)
    wrong_cal = np.asarray(wrong_cal, bool)
    k0 = start_keep(s_cal.size, alpha, delta, keep_grid)
    certified, pvals = None, {}
    if k0 is None:
        return None, pvals
    for i, k in enumerate(keep_grid):
        if k < k0 - 1e-9:
            continue
        kept = s_cal >= lams[i]
        p = p_value(int(kept.sum()), int(wrong_cal[kept].sum()), alpha)
        pvals[i] = p
        if p > delta:
            break
        certified = i
    return certified, pvals


def loosest_where(values, ok) -> int | None:
    """Largest index whose condition holds (for the uncertified baselines)."""
    idx = [i for i, v in enumerate(values) if ok(v)]
    return idx[-1] if idx else None


def cp_lower(e: int, n: int, level: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on a rate from e errors in n shots."""
    return 0.0 if e == 0 or n == 0 else float(beta.ppf(1 - level, e, n - e + 1))
