"""Exact posterior P(observable flip | detection events) under a DEM, for small instances (spec section 8.3).

Keeps a probability table over every (detector bits, observable bit) pattern and folds in the error
mechanisms one at a time: with probability p a mechanism fires and XORs its pattern into the state.
Mechanisms with the same full pattern are merged first. A DEM's "^" decomposition is only a hint for
matching, so each mechanism's pattern is the XOR of all its components.

The table is viewed as a (2,)*bits array, so XOR with a mask is a flip along that mask's axes (a view,
no index arrays). Cost O(mechanisms * 2^bits); memory 8 * 2^bits bytes, 268 MB at 25 bits.
"""
from __future__ import annotations

import numpy as np
import stim

MAX_BITS = 26


def mechanisms(dem: stim.DetectorErrorModel) -> dict[int, float]:
    """{pattern mask: probability}, bit k = detector k, bit num_detectors = observable 0."""
    n = dem.num_detectors
    out: dict[int, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                if t.val != 0:
                    raise ValueError("only observable 0 is supported")
                mask ^= 1 << n
        if mask == 0 or p == 0:
            continue
        q = out.get(mask, 0.0)
        out[mask] = q * (1 - p) + p * (1 - q)          # two independent mechanisms with the same effect
    return out


class ExactPosterior:
    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_detectors = n = dem.num_detectors
        bits = n + 1
        if bits > MAX_BITS:
            raise ValueError(f"{bits} bits exceeds the exact-posterior limit of {MAX_BITS}")
        T = np.zeros(1 << bits)
        T[0] = 1.0
        T = T.reshape((2,) * bits)                      # axis 0 is the most significant bit
        self.num_mechanisms = 0
        for mask, p in mechanisms(dem).items():
            axes = tuple(bits - 1 - b for b in range(bits) if mask >> b & 1)
            new = T * (1 - p)
            new += p * np.flip(T, axis=axes)
            T = new
            self.num_mechanisms += 1
        self.table = T.reshape(-1)

    def p_flip(self, det: np.ndarray) -> np.ndarray:
        """P(observable flipped | detection events) per shot; det is (shots, num_detectors) bool."""
        n = self.num_detectors
        key = np.asarray(det, dtype=np.int64) @ (np.int64(1) << np.arange(n, dtype=np.int64))
        p0 = self.table[key]
        p1 = self.table[key | (1 << n)]
        tot = p0 + p1
        with np.errstate(invalid="ignore", divide="ignore"):
            post = np.where(tot > 0, p1 / tot, 0.5)     # a pattern the DEM cannot produce gets 0.5
        return post

    def decode(self, det: np.ndarray):
        """(prediction bool, q = P(prediction wrong)) per shot."""
        post = self.p_flip(det)
        return post > 0.5, np.minimum(post, 1 - post)
