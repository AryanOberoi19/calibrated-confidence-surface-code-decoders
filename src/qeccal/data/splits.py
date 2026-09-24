"""Train / Calibrate / Test split of the shots in each experiment (spec section 8.4), fixed once.

Decisions (docs/data.md, 24 Sep 2026):
  * 40 / 30 / 30 per experiment, by a seeded random permutation of shot indices.
  * r=13 experiments are held out: every shot gets the role "sanity" (decoder sanity checks only),
    because Google's RL-optimized prior was fitted on the 13-cycle data.

The permutation is seeded by (SEED, sha256 of the experiment key), so each experiment's split is
reproducible on its own. splits/splits_v1.json records a checksum per experiment; verify_manifest()
recomputes them, so a change in numpy's generator or in the code shows up as a test failure.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from .willow import REPO, Experiment

VERSION = "v1"
SEED = 20260924
SHARES = {"train": 0.4, "calibrate": 0.3, "test": 0.3}
HOLDOUT_ROUNDS = (13,)
ROLES = ("train", "calibrate", "test", "sanity")
MANIFEST = REPO / "splits" / f"splits_{VERSION}.json"


def _key_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def role_codes(key: str, rounds: int, shots: int) -> np.ndarray:
    """(shots,) uint8 array of indices into ROLES."""
    if rounds in HOLDOUT_ROUNDS:
        return np.full(shots, ROLES.index("sanity"), dtype=np.uint8)
    perm = np.random.default_rng([SEED, _key_seed(key)]).permutation(shots)
    codes = np.empty(shots, dtype=np.uint8)
    n_train = round(SHARES["train"] * shots)
    n_cal = round(SHARES["calibrate"] * shots)
    codes[perm[:n_train]] = ROLES.index("train")
    codes[perm[n_train:n_train + n_cal]] = ROLES.index("calibrate")
    codes[perm[n_train + n_cal:]] = ROLES.index("test")
    return codes


def split_indices(exp: Experiment) -> dict[str, np.ndarray]:
    """Sorted shot indices per role for one experiment. Roles with no shots are omitted."""
    codes = role_codes(exp.key, exp.rounds, exp.shots)
    return {r: np.flatnonzero(codes == i) for i, r in enumerate(ROLES) if (codes == i).any()}


def _entry(key, rounds, shots):
    codes = role_codes(key, rounds, shots)
    counts = {r: int((codes == i).sum()) for i, r in enumerate(ROLES)}
    return {"shots": shots, **{r: c for r, c in counts.items() if c}, "sha256": hashlib.sha256(codes.tobytes()).hexdigest()}


def build_manifest(experiments: list[Experiment]) -> dict:
    return {
        "version": VERSION,
        "seed": SEED,
        "shares": SHARES,
        "holdout_rounds": list(HOLDOUT_ROUNDS),
        "method": "per experiment: numpy default_rng([seed, first 8 bytes of sha256(key) as little-endian int])"
                  ".permutation(shots); first 40% train, next 30% calibrate, rest test; held-out rounds -> sanity",
        "experiments": {e.key: _entry(e.key, e.rounds, e.shots) for e in experiments},
    }


def write_manifest(experiments: list[Experiment], path: pathlib.Path = MANIFEST) -> dict:
    m = build_manifest(experiments)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=1) + "\n")
    return m


def verify_manifest(path: pathlib.Path = MANIFEST) -> list[str]:
    """Keys whose recomputed split differs from the committed manifest (empty list = all match).
    Needs only the manifest, not the data."""
    m = json.loads(pathlib.Path(path).read_text())
    if m["seed"] != SEED or m["shares"] != SHARES or m["holdout_rounds"] != list(HOLDOUT_ROUNDS):
        return ["<settings differ from code>"]
    bad = []
    for key, entry in m["experiments"].items():
        rounds = int(key.rsplit("/r", 1)[1])
        if _entry(key, rounds, entry["shots"]) != entry:
            bad.append(key)
    return bad
