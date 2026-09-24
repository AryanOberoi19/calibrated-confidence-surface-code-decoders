"""Per-shot soft outputs cached under results/cache/soft/<method>/<prior>/<key>.npz (git-ignored).
Written by scripts/m2_soft_outputs.py; read by the summary and calibration scripts."""
from __future__ import annotations

import os
import pathlib

import numpy as np

from ..data.willow import REPO

CACHE = pathlib.Path(os.environ.get("QECCAL_SOFT_CACHE", REPO / "results" / "cache" / "soft"))


def cache_path(method: str, prior: str, key: str) -> pathlib.Path:
    return CACHE / method / prior / (key.replace("/", "__") + ".npz")


def save(method: str, prior: str, key: str, pred: np.ndarray, **arrays) -> pathlib.Path:
    path = cache_path(method, prior, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pred=np.packbits(pred, bitorder="little"), shots=pred.size, **arrays)
    tmp.rename(path)
    return path


def load(method: str, prior: str, key: str) -> dict:
    """{'pred': bool, 'q': P(prediction wrong), ...}; 'gap' for mwpm_gap, 'p_flip' for exact."""
    z = np.load(cache_path(method, prior, key))
    out = {k: z[k] for k in z.files}
    out["pred"] = np.unpackbits(out["pred"], count=int(out["shots"]), bitorder="little").astype(bool)
    if "gap" in out:
        out["q"] = 1.0 / (1.0 + np.exp(out["gap"].astype(np.float64)))
    elif "p_flip" in out:
        out["q"] = np.minimum(out["p_flip"], 1 - out["p_flip"])
    return out
