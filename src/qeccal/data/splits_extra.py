"""Splits added for the arXiv v1 robustness work; the frozen split of splits.py is not changed.

blocked_indices (plan A3): Train / Calibrate / Test as consecutive blocks in file order, so a threshold is
calibrated on earlier shots and evaluated on later ones.
confirmatory_indices (plan A7): a 40 / 30 / 30 split of the held-out r = 13 experiments, built like
splits.role_codes but with its own seed, for the preregistered confirmatory run (docs/prereg_r13.md).
"""
from __future__ import annotations

import numpy as np

from .splits import SHARES, _key_seed

CONFIRMATORY_SEED = 20261117


def blocked_indices(shots: int, shares: dict = SHARES) -> dict[str, np.ndarray]:
    """First 40% of shot indices -> train, next 30% -> calibrate, the rest -> test (same rounding as splits.py)."""
    n_train = round(shares["train"] * shots)
    n_cal = round(shares["calibrate"] * shots)
    return {"train": np.arange(0, n_train), "calibrate": np.arange(n_train, n_train + n_cal),
            "test": np.arange(n_train + n_cal, shots)}


def confirmatory_indices(key: str, shots: int, seed: int = CONFIRMATORY_SEED) -> dict[str, np.ndarray]:
    """Sorted shot indices per role: numpy default_rng([seed, first 8 bytes of sha256(key)]).permutation(shots);
    first 40% train, next 30% calibrate, rest test (the construction of splits.role_codes, with a new seed)."""
    perm = np.random.default_rng([seed, _key_seed(key)]).permutation(shots)
    n_train = round(SHARES["train"] * shots)
    n_cal = round(SHARES["calibrate"] * shots)
    return {"train": np.sort(perm[:n_train]), "calibrate": np.sort(perm[n_train:n_train + n_cal]),
            "test": np.sort(perm[n_train + n_cal:])}


def confirmatory_manifest(experiments) -> dict:
    """Shots, role counts and a SHA-256 of the role codes per experiment, as in splits_v1.json."""
    import hashlib
    out = {}
    for e in experiments:
        idx = confirmatory_indices(e.key, e.shots)
        codes = np.empty(e.shots, np.uint8)
        for i, r in enumerate(("train", "calibrate", "test")):
            codes[idx[r]] = i
        out[e.key] = {"shots": e.shots, **{r: int(v.size) for r, v in idx.items()},
                      "sha256": hashlib.sha256(codes.tobytes()).hexdigest()}
    return {"version": "r13_v1", "seed": CONFIRMATORY_SEED, "shares": SHARES,
            "method": "per experiment: numpy default_rng([seed, first 8 bytes of sha256(key) as little-endian int])"
                      ".permutation(shots); first 40% train, next 30% calibrate, rest test",
            "experiments": out}
