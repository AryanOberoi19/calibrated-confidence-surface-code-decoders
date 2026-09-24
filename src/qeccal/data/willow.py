"""Reader for Google's Willow 105-qubit surface-code archive (Zenodo 13273331), extracted under data/raw/.

Layout (confirmed from the archive README, see docs/data.md):

    google_105Q_surface_code_d3_d5_d7/d<d>_at_q<r>_<c>/<X|Z>/r<NN>/
        circuit_ideal.stim  circuit_noisy_si1000.stim  metadata.json
        detection_events.b8  obs_flips_actual.b8  measurements.b8  sweep_bits.b8
        decoding_results/<pathway>/{error_model.dem, obs_flips_predicted.b8}

b8 files pack each shot into ceil(bits/8) bytes, least significant bit first. There is one observable,
so observable files hold one byte per shot with the flip in bit 0.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import pathlib
import re
from collections.abc import Iterable

import numpy as np
import stim

REPO = pathlib.Path(__file__).resolve().parents[3]
# Override with the QECCAL_DATA_ROOT environment variable if the data lives elsewhere.
DEFAULT_ROOT = pathlib.Path(os.environ.get("QECCAL_DATA_ROOT", REPO / "data" / "raw" / "google_105Q_surface_code_d3_d5_d7"))

EXP_RE = re.compile(r"^d(?P<d>\d+)_at_(?P<patch>q\d+_\d+)/(?P<basis>[XZ])/r(?P<rounds>\d+)$")

PATHWAYS = (
    "correlated_matching_decoder_with_si1000_prior",
    "correlated_matching_decoder_with_rl_optimized_prior",
    "harmony_decoder_with_si1000_prior",
    "harmony_decoder_with_rl_optimized_prior",
    "libra_decoder_with_rl_optimized_prior",
)
# Priors shipped with the data. Every pathway with the same prior ships the same DEM file,
# so each prior is read from its correlated-matching pathway.
PRIOR_PATHWAY = {
    "si1000": "correlated_matching_decoder_with_si1000_prior",
    "rl": "correlated_matching_decoder_with_rl_optimized_prior",
}


def _as_set(x):
    if x is None:
        return None
    if isinstance(x, (str, int)):
        return {x}
    return set(x)


def read_b8(path: pathlib.Path, bits_per_shot: int, *, packed: bool = False) -> np.ndarray:
    """Shots from a b8 file: (n, bits) bool, or (n, ceil(bits/8)) uint8 if packed (the layout PyMatching's
    bit_packed_shots expects)."""
    nbytes = (bits_per_shot + 7) // 8
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % nbytes:
        raise ValueError(f"{path}: {raw.size} bytes is not a multiple of {nbytes} bytes per shot")
    raw = raw.reshape(-1, nbytes)
    if packed:
        return raw
    return np.unpackbits(raw, axis=1, bitorder="little")[:, :bits_per_shot].astype(bool)


@dataclasses.dataclass(frozen=True)
class Experiment:
    """One memory experiment: a patch, a basis and a round count; 50,000 shots in this archive."""

    key: str            # path relative to the archive root, e.g. "d3_at_q4_9/Z/r10"
    distance: int
    patch: str
    basis: str
    rounds: int
    path: pathlib.Path

    def __repr__(self):
        return f"Experiment({self.key})"

    # --- small files ---------------------------------------------------------
    def metadata(self) -> dict:
        return json.loads((self.path / "metadata.json").read_text())

    @functools.cached_property
    def shots(self) -> int:
        return int(self.metadata()["shots"])

    def circuit(self, noisy: bool = False) -> stim.Circuit:
        name = "circuit_noisy_si1000.stim" if noisy else "circuit_ideal.stim"
        return stim.Circuit.from_file(str(self.path / name))

    @functools.cached_property
    def num_detectors(self) -> int:
        return self.circuit().num_detectors

    def pathways(self) -> list[str]:
        d = self.path / "decoding_results"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def dem(self, prior: str = "si1000") -> stim.DetectorErrorModel:
        """Detector error model to decode with.

        "si1000" / "rl": the DEMs shipped with Google's decoding results (SI1000 prior, and the
        RL-optimized prior fitted on the r=13 data). "si1000_circuit": built by Stim from
        circuit_noisy_si1000.stim; it differs from the shipped SI1000 DEM.
        """
        if prior == "si1000_circuit":
            return self.circuit(noisy=True).detector_error_model(decompose_errors=True)
        if prior not in PRIOR_PATHWAY:
            raise ValueError(f"unknown prior {prior!r}; use one of {[*PRIOR_PATHWAY, 'si1000_circuit']}")
        return stim.DetectorErrorModel.from_file(str(self.path / "decoding_results" / PRIOR_PATHWAY[prior] / "error_model.dem"))

    # --- shot data -----------------------------------------------------------
    def detection_events(self, *, packed: bool = False) -> np.ndarray:
        """(shots, num_detectors) bool, or bit-packed uint8 rows if packed=True."""
        out = read_b8(self.path / "detection_events.b8", self.num_detectors, packed=packed)
        self._check_rows(out, "detection_events.b8")
        return out

    def observable_flips(self) -> np.ndarray:
        """(shots,) bool: the ground truth the decoders try to predict."""
        out = read_b8(self.path / "obs_flips_actual.b8", 1)[:, 0]
        self._check_rows(out, "obs_flips_actual.b8")
        return out

    def google_predictions(self, pathway: str) -> np.ndarray | None:
        """(shots,) bool predictions of one of Google's pathways, or None if this experiment lacks it."""
        f = self.path / "decoding_results" / pathway / "obs_flips_predicted.b8"
        if not f.exists():
            return None
        out = read_b8(f, 1)[:, 0]
        self._check_rows(out, f"{pathway}/obs_flips_predicted.b8")
        return out

    def _check_rows(self, arr, name):
        if arr.shape[0] != self.shots:
            raise ValueError(f"{self.key}/{name}: {arr.shape[0]} shots, metadata says {self.shots}")


def list_experiments(root: pathlib.Path | str = DEFAULT_ROOT, *, distance=None, basis=None, rounds=None,
                     patch=None) -> list[Experiment]:
    """All experiments under root, filtered by any of distance, basis, rounds, patch (a value or an iterable).
    Sorted by distance, patch, basis, rounds."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found: extract the Zenodo archive into data/raw/ (see docs/data.md)")
    want = {"distance": _as_set(distance), "basis": _as_set(basis), "rounds": _as_set(rounds), "patch": _as_set(patch)}
    out = []
    for meta in root.glob("d*_at_q*/[XZ]/r*/metadata.json"):
        key = meta.parent.relative_to(root).as_posix()
        m = EXP_RE.match(key)
        if not m:
            continue
        e = Experiment(key=key, distance=int(m["d"]), patch=m["patch"], basis=m["basis"], rounds=int(m["rounds"]),
                       path=meta.parent)
        if all(v is None or getattr(e, k) in v for k, v in want.items()):
            out.append(e)
    return sorted(out, key=lambda e: (e.distance, e.patch, e.basis, e.rounds))


def get_experiment(key: str, root: pathlib.Path | str = DEFAULT_ROOT) -> Experiment:
    """Experiment by key, e.g. "d5_at_q6_5/X/r50". "r1" is accepted for the archive's "r01"."""
    root = pathlib.Path(root)
    key = re.sub(r"/r(\d)$", r"/r0\1", key)
    m = EXP_RE.match(key)
    if not m or not (root / key / "metadata.json").exists():
        raise KeyError(f"no experiment {key!r} under {root}")
    return Experiment(key=key, distance=int(m["d"]), patch=m["patch"], basis=m["basis"], rounds=int(m["rounds"]),
                      path=root / key)
