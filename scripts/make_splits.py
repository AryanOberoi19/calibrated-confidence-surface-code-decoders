"""Write splits/splits_v1.json: the fixed Train / Calibrate / Test assignment for every experiment.

Run once after extracting the archive; commit the manifest. Tests re-derive every checksum from it.

    python scripts/make_splits.py
"""
import collections

from qeccal.data import list_experiments, verify_manifest
from qeccal.data.splits import MANIFEST, write_manifest


def main():
    exps = list_experiments()
    m = write_manifest(exps)
    totals = collections.Counter()
    for entry in m["experiments"].values():
        totals.update({k: v for k, v in entry.items() if k not in ("shots", "sha256")})
    print(f"{len(exps)} experiments -> {MANIFEST}")
    print("shots per role (all experiments):", dict(totals))
    assert verify_manifest() == [], "manifest does not reproduce"
    print("manifest verified")


if __name__ == "__main__":
    main()
