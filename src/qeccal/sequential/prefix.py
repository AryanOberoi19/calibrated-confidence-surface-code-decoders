"""Prefix logical gap: the decoder's confidence partway through a run.

After t rounds, only the detectors whose time coordinate is below t have been measured (Google's circuits give
each detector a coordinate (x, y, t), with t = 0..r, and number detectors in time order). The prefix DEM keeps,
for every error component of the full DEM, only its measured detectors:

  * a component with no measured detector is dropped: its evidence lies in the future;
  * a component that straddles the cut keeps its measured detectors, so a defect next to the cut can be
    explained by an error whose other end has not been measured yet (an open time boundary);
  * observable flags are kept.

The prefix gap is the logical gap (qeccal.soft.gap) of the measured detection events under the prefix DEM. With
every detector measured it is the ordinary logical gap.
"""
from __future__ import annotations

import numpy as np
import stim

from ..soft.gap import LogicalGap, _components


def detector_steps(circuit: stim.Circuit) -> np.ndarray:
    """(num_detectors,) int: the time step t of each detector."""
    coords = circuit.get_detector_coordinates()
    return np.array([int(round(coords[i][2])) for i in range(circuit.num_detectors)], dtype=np.int64)


def checkpoints_for(rounds: int, every: int = 5) -> list[int]:
    """Rounds at which an abort decision is taken: every `every` rounds, strictly before the end."""
    return list(range(every, rounds, every))


def prefix_dem(dem: stim.DetectorErrorModel, n_measured: int) -> stim.DetectorErrorModel:
    """The DEM restricted to detectors 0..n_measured-1 (detectors must be numbered in time order)."""
    out = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        targets = []
        for comp in _components(inst):
            dets = [t for t in comp if t.is_relative_detector_id() and t.val < n_measured]
            if not dets:
                continue
            obs = [t for t in comp if t.is_logical_observable_id()]
            if targets:
                targets.append(stim.target_separator())
            targets += dets + obs
        if targets:
            out.append("error", inst.args_copy(), targets)
    for k in range(n_measured):                      # fixes num_detectors even if a detector has no error
        out.append("detector", [], [stim.target_relative_detector_id(k)])
    out.append("logical_observable", [], [stim.target_logical_observable_id(0)])
    return out


def prefix_gaps(dem: stim.DetectorErrorModel, steps: np.ndarray, det: np.ndarray, checkpoints) -> tuple:
    """Prefix gap and number of detection events after each checkpoint round.

    dem: the full DEM; steps: detector_steps of the circuit (non-decreasing); det: (shots, num_detectors) bool.
    Returns (gaps float32 (shots, K), counts uint16 (shots, K)).
    """
    if np.any(np.diff(steps) < 0):
        raise ValueError("detectors are not numbered in time order")
    gaps = np.empty((det.shape[0], len(checkpoints)), dtype=np.float32)
    counts = np.empty((det.shape[0], len(checkpoints)), dtype=np.uint16)
    for j, t in enumerate(checkpoints):
        n = int(np.searchsorted(steps, t, side="left"))          # detectors with step < t
        _, g = LogicalGap(prefix_dem(dem, n)).decode(det[:, :n])
        gaps[:, j] = g
        counts[:, j] = det[:, :n].sum(axis=1)
    return gaps, counts
