"""Logical gap for minimum-weight perfect matching (spec section 8.3).

Construction: the logical observable becomes one extra detector node N. Every DEM component that flips the
observable gets N added to its detectors. Decoding a shot twice, with N's bit forced to 0 and to 1, gives
the minimum correction weight with even (w0) and odd (w1) observable parity. The prediction is the
lighter class and the gap is |w1 - w0|. PyMatching's edge weights are ln((1 - p) / p), so the gap is an
approximate log-likelihood ratio between the two classes and q = 1 / (1 + e^gap) is the implied
probability that the prediction is wrong.

Precondition: every observable-flipping component touches at most one detector; otherwise adding N
creates a hyperedge that matching cannot use. It holds on all of Google's DEMs (docs/data_checks.md).
"""
from __future__ import annotations

import numpy as np
import pymatching
import stim


class PreconditionError(ValueError):
    """A DEM component flips the observable and touches two or more detectors."""


def _components(inst):
    comps, cur = [], []
    for t in inst.targets_copy():
        if t.is_separator():
            comps.append(cur)
            cur = []
        else:
            cur.append(t)
    comps.append(cur)
    return comps


def gap_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """The DEM with observable 0 turned into the extra detector D<num_detectors> (the observable is kept too)."""
    if dem.num_observables != 1:
        raise ValueError(f"expected one observable, got {dem.num_observables}")
    n = dem.num_detectors
    extra = stim.target_relative_detector_id(n)
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type != "error":
            out.append(inst)
            continue
        targets = []
        for comp in _components(inst):
            dets = [t for t in comp if t.is_relative_detector_id()]
            obs = [t for t in comp if t.is_logical_observable_id()]
            if obs:
                if len(dets) >= 2:
                    raise PreconditionError(f"component {comp} flips the observable and touches {len(dets)} detectors")
                dets = dets + [extra]
            if targets:
                targets.append(stim.target_separator())
            targets += dets + obs
        out.append("error", inst.args_copy(), targets)
    return out


def q_from_gap(gap: np.ndarray) -> np.ndarray:
    """Implied probability that the lighter class is wrong: 1 / (1 + e^gap), in (0, 0.5]."""
    return 1.0 / (1.0 + np.exp(np.asarray(gap, dtype=np.float64)))


class LogicalGap:
    """MWPM prediction and logical gap for each shot, under one DEM."""

    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_detectors = dem.num_detectors
        self.matching = pymatching.Matching.from_detector_error_model(gap_dem(dem))

    def decode(self, det: np.ndarray, *, packed: bool = False, chunk: int = 10_000):
        """det: (shots, num_detectors) bool, or bit-packed uint8 rows if packed=True.
        Returns (prediction bool, gap float64); gap is 0 on ties, where the prediction is class 0."""
        n = det.shape[0]
        pred = np.empty(n, dtype=bool)
        gap = np.empty(n, dtype=np.float64)
        D = self.num_detectors
        for s in range(0, n, chunk):
            block = det[s:s + chunk]
            if packed:
                block = np.unpackbits(block, axis=1, bitorder="little")[:, :D]
            x = np.zeros((block.shape[0], D + 1), dtype=np.uint8)
            x[:, :D] = block
            _, w0 = self.matching.decode_batch(x, return_weights=True)
            x[:, D] = 1
            _, w1 = self.matching.decode_batch(x, return_weights=True)
            pred[s:s + chunk] = w1 < w0
            gap[s:s + chunk] = np.abs(w1 - w0)
        return pred, gap


def sample_gap(dem: stim.DetectorErrorModel, shots: int, seed: int):
    """Simulate shots from the DEM itself and decode them: (gap, wrong) as if the DEM were the true noise.
    Used as the source side of the simulation-to-hardware shift (spec section 8.7)."""
    det, obs, _ = dem.compile_sampler(seed=seed).sample(shots)
    pred, gap = LogicalGap(dem).decode(det)
    return gap, pred != obs[:, 0]
