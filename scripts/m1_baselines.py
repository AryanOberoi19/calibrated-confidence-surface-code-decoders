"""M1 validation: decode every experiment with PyMatching and set it beside Google's released predictions.

For each experiment and each shipped prior (SI1000, RL-optimized) we run plain MWPM and correlated MWPM
(PyMatching enable_correlations) on all 50,000 shots, and also score Google's five pathways on the same
shots. Per-shot agreement with Google's correlated matching (same prior) checks the loader end to end:
same shots, same labels, same DEM, and a closely related algorithm.

    python scripts/m1_baselines.py                  # all 420 experiments, 8 worker processes
    python scripts/m1_baselines.py --distance 3 --rounds 10 50

Output: results/m1_baselines.csv, one row per (experiment, decoder, prior). Resumable: experiments
already in the CSV are skipped. Error counts are given on all shots and on the Test split.
"""
import argparse
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np
import pymatching

from qeccal.data import PATHWAYS, PRIOR_PATHWAY, get_experiment, list_experiments
from qeccal.data.splits import ROLES, role_codes

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m1_baselines.csv"
FIELDS = ["key", "distance", "patch", "basis", "rounds", "source", "decoder", "prior", "shots", "errors",
          "test_shots", "test_errors", "agree_with_google_cm", "seconds"]
OURS = [("si1000", False), ("si1000", True), ("rl", False), ("rl", True)]
TEST = ROLES.index("test")


def decode_experiment(key):
    e = get_experiment(key)
    det = e.detection_events(packed=True)
    obs = e.observable_flips()
    test = role_codes(e.key, e.rounds, e.shots) == TEST
    base = {"key": e.key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds,
            "shots": e.shots, "test_shots": int(test.sum())}
    rows = []

    def score(pred):
        wrong = pred != obs
        return {"errors": int(wrong.sum()), "test_errors": int(wrong[test].sum())}

    for prior, corr in OURS:
        t0 = time.time()
        m = pymatching.Matching.from_detector_error_model(e.dem(prior), enable_correlations=corr)
        pred = m.decode_batch(det, bit_packed_shots=True, enable_correlations=corr)[:, 0].astype(bool)
        g = e.google_predictions(PRIOR_PATHWAY[prior])
        rows.append({**base, "source": "ours", "decoder": "pymatching_correlated" if corr else "pymatching",
                     "prior": prior, **score(pred), "agree_with_google_cm": int((pred == g).sum()),
                     "seconds": round(time.time() - t0, 2)})
    for pw in PATHWAYS:
        g = e.google_predictions(pw)
        if g is None:
            continue
        decoder, prior = pw.split("_decoder_with_")
        rows.append({**base, "source": "google", "decoder": decoder,
                     "prior": "rl" if prior.startswith("rl") else "si1000", **score(g),
                     "agree_with_google_cm": "", "seconds": ""})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--distance", type=int, nargs="*")
    ap.add_argument("--rounds", type=int, nargs="*")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    out = args.out

    exps = list_experiments(distance=args.distance or None, rounds=args.rounds or None)
    done = set()
    if out.exists():
        with out.open() as f:
            done = {r["key"] for r in csv.DictReader(f)}
    todo = sorted((e for e in exps if e.key not in done), key=lambda e: -(e.distance ** 2 * e.rounds))
    print(f"{len(exps)} experiments selected, {len(exps) - len(todo)} already in {out.name}, {len(todo)} to run",
          flush=True)
    if not todo:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    t0 = time.time()
    with out.open("a", newline="") as f, cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futures = {pool.submit(decode_experiment, e.key): e.key for e in todo}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            rows = fut.result()
            w.writerows(rows)
            f.flush()
            ours = {(r["decoder"], r["prior"]): r["errors"] for r in rows if r["source"] == "ours"}
            print(f"[{i}/{len(todo)}] {futures[fut]}  corr/rl errors {ours[('pymatching_correlated', 'rl')]}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
