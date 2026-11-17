"""Splits added for the arXiv v1 robustness work; the frozen split of splits.py is not changed.

blocked_indices (plan A3): Train / Calibrate / Test as consecutive blocks in file order, so a threshold is
calibrated on earlier shots and evaluated on later ones.
"""
from __future__ import annotations

import numpy as np

from .splits import SHARES


def blocked_indices(shots: int, shares: dict = SHARES) -> dict[str, np.ndarray]:
    """First 40% of shot indices -> train, next 30% -> calibrate, the rest -> test (same rounding as splits.py)."""
    n_train = round(shares["train"] * shots)
    n_cal = round(shares["calibrate"] * shots)
    return {"train": np.arange(0, n_train), "calibrate": np.arange(n_train, n_train + n_cal),
            "test": np.arange(n_train + n_cal, shots)}
