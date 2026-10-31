"""M6, step 1: prefix gaps for early abort (spec section 8.8).

For every experiment with r in 10, 30 and 50, decode the detection events measured after each checkpoint round
t = 5, 10, ..., r - 5 under the prefix of the RL-prior DEM (qeccal.sequential.prefix), and store the prefix gap and
the number of detection events so far. The final gap and prediction come from the M2 cache (mwpm_gap, rl).

    python scripts/m6_prefix_gaps.py [--workers 10] [--force]
Writes results/cache/prefix/rl/<key>.npz (git-ignored).
"""
import argparse
import concurrent.futures as cf
import pathlib
import time

import numpy as np

from qeccal.data import list_experiments
from qeccal.data.willow import get_experiment
from qeccal.sequential import checkpoints_for, detector_steps, prefix_gaps

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "cache" / "prefix" / "rl"
ROUNDS = (10, 30, 50)


def path_for(key: str) -> pathlib.Path:
    return OUT / (key.replace("/", "__") + ".npz")


def run(key: str):
    t0 = time.time()
    e = get_experiment(key)
    cps = checkpoints_for(e.rounds)
    gaps, counts = prefix_gaps(e.dem("rl"), detector_steps(e.circuit()), e.detection_events(), cps)
    p = path_for(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, gaps=gaps, counts=counts, checkpoints=np.array(cps), rounds=e.rounds)
    tmp.rename(p)
    return key, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    exps = [e for e in list_experiments(rounds=ROUNDS) if args.force or not path_for(e.key).exists()]
    exps.sort(key=lambda e: -(e.distance ** 2 * e.rounds))             # largest first
    print(f"{len(exps)} experiments", flush=True)
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for fut in cf.as_completed([ex.submit(run, e.key) for e in exps]):
            key, sec = fut.result()
            print(f"{key}  {sec:.0f} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
    print("done", flush=True)
