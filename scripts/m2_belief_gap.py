"""M2b: belief-matching prediction and gap (qeccal.soft.belief) on hardware experiments.

Belief-matching costs about 100x the plain MWPM gap per shot (20 BP iterations over every DEM mechanism, then
a matching graph rebuilt per shot), so it runs on a subset: the RL prior and r in {1, 10, 30, 50} by default,
all patches and bases. Each experiment is split into chunks of shots spread over worker processes.

Per-shot outputs go to results/cache/soft/bm_gap/<prior>/<key>.npz (same layout as the MWPM gap, so the M4 and
M5 scripts read them unchanged); per-experiment counts to results/m2_belief_gap.csv. Resumable per experiment.

    python scripts/m2_belief_gap.py --probe 200          # time 200 shots per experiment, write nothing
    python scripts/m2_belief_gap.py                      # the default subset, 8 workers
    python scripts/m2_belief_gap.py --distance 3 --rounds 1 10 --workers 10
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                        # one thread per worker process

import argparse
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np

from qeccal.data import get_experiment, list_experiments
from qeccal.soft.belief import BeliefMatchingGap
from qeccal.soft.cache import cache_path, save

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m2_belief_gap.csv"
METHOD = "bm_gap"
CHUNK = 2_500
FIELDS = ["key", "distance", "patch", "basis", "rounds", "method", "prior", "shots", "errors", "ties",
          "cpu_seconds"]
_decoders = {}


def _decoder(key, prior):
    if (key, prior) not in _decoders:
        _decoders.clear()
        _decoders[key, prior] = BeliefMatchingGap(get_experiment(key).dem(prior))
    return _decoders[key, prior]


def run_chunk(key, prior, start, stop):
    t0 = time.process_time()
    det = get_experiment(key).detection_events(packed=True)[start:stop]
    pred, gap = _decoder(key, prior).decode(det, packed=True)
    return key, start, pred, gap, time.process_time() - t0


def probe(exps, prior, n):
    print(f"{'experiment':22s} {'dets':>5s} {'mechs':>6s} {'ms/shot':>8s} {'core-h for all shots':>21s}")
    total = 0.0
    for e in exps:
        t0 = time.process_time()
        bm = BeliefMatchingGap(e.dem(prior))
        build = time.process_time() - t0
        det = e.detection_events(packed=True)[:n]
        t0 = time.process_time()
        bm.decode(det, packed=True)
        ms = (time.process_time() - t0) / n * 1e3
        hours = (ms * e.shots / 1e3 + build * e.shots / CHUNK) / 3600
        total += hours
        print(f"{e.key:22s} {e.num_detectors:5d} {bm.check.shape[1]:6d} {ms:8.2f} {hours:21.2f}", flush=True)
    print(f"total {total:.1f} core-hours")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--prior", default="rl", choices=("rl", "si1000"))
    ap.add_argument("--distance", type=int, nargs="*")
    ap.add_argument("--rounds", type=int, nargs="*", default=[1, 10, 30, 50])
    ap.add_argument("--probe", type=int, default=0, help="time this many shots per experiment and exit")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    exps = list_experiments(distance=args.distance or None, rounds=args.rounds)
    if args.probe:
        probe(exps, args.prior, args.probe)
        return
    todo = [e for e in exps if not cache_path(METHOD, args.prior, e.key).exists()]
    todo.sort(key=lambda e: -e.num_detectors)
    print(f"{len(exps)} experiments, {len(exps) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    tasks = [(e.key, s, min(s + CHUNK, e.shots)) for e in todo for s in range(0, e.shots, CHUNK)]
    parts = {e.key: {} for e in todo}
    cpu = {e.key: 0.0 for e in todo}
    need = {e.key: len(range(0, e.shots, CHUNK)) for e in todo}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    t0 = time.time()
    done = 0
    with args.out.open("a", newline="") as f, cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futs = [pool.submit(run_chunk, key, args.prior, s, t) for key, s, t in tasks]
        for fut in cf.as_completed(futs):
            key, start, pred, gap, secs = fut.result()
            parts[key][start] = (pred, gap)
            cpu[key] += secs
            if len(parts[key]) < need[key]:
                continue
            e = get_experiment(key)
            pred = np.concatenate([parts[key][s][0] for s in sorted(parts[key])])
            gap = np.concatenate([parts[key][s][1] for s in sorted(parts[key])])
            del parts[key]
            save(METHOD, args.prior, key, pred, gap=gap.astype(np.float32))
            errors = int((pred != e.observable_flips()).sum())
            w.writerow({"key": key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds,
                        "method": METHOD, "prior": args.prior, "shots": e.shots, "errors": errors,
                        "ties": int((gap == 0).sum()), "cpu_seconds": round(cpu[key], 1)})
            f.flush()
            done += 1
            print(f"[{done}/{len(todo)}] {key}  {errors} err  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
