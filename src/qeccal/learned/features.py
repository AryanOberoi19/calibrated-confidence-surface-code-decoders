"""Detection events as a sequence over time for the recurrent decoder.

Google's circuits give each detector a coordinate whose third entry is its time step t = 0..r. At d = 3 a
middle step has 8 detectors (one per stabilizer); the first and last steps have 4 (only one stabilizer type
is deterministic there, and the last step compares with the data-qubit measurements). Within a step the
detectors are kept in index order, which is the same stabilizer order in every round and, because the
patches are translates of each other, in every patch of the same basis. The first and last steps fill the
first slots and are marked by flags, so the network can give them their own meaning.
"""
from __future__ import annotations

import numpy as np
import stim


class Layout:
    def __init__(self, circuit: stim.Circuit, slots: int | None = None):
        """slots: width of a step (default: the widest step here); pass the common width to mix round counts."""
        coords = circuit.get_detector_coordinates()
        n = circuit.num_detectors
        t = np.array([int(round(coords[i][2])) for i in range(n)])
        self.steps = int(t.max()) + 1
        slot = np.empty(n, dtype=np.int64)
        for s in range(self.steps):
            members = np.flatnonzero(t == s)                   # index order
            slot[members] = np.arange(members.size)
        widest = int(slot.max()) + 1
        if slots is not None and slots < widest:
            raise ValueError(f"a step has {widest} detectors, more than slots={slots}")
        self.slots = slots or widest
        self.num_detectors = n
        self.position = t * self.slots + slot                  # flat index into (steps, slots)

    def sequence(self, det: np.ndarray, *, packed: bool = False) -> np.ndarray:
        """(shots, steps, slots) uint8 from detection events (shots, detectors), bit-packed if packed=True."""
        if packed:
            det = np.unpackbits(det, axis=1, count=self.num_detectors, bitorder="little")
        out = np.zeros((det.shape[0], self.steps * self.slots), dtype=np.uint8)
        out[:, self.position] = det
        return out.reshape(det.shape[0], self.steps, self.slots)
