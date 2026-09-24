"""M2: per-shot soft outputs on every hardware experiment, under both shipped DEMs.

  * mwpm_gap  - MWPM prediction and logical gap (qeccal.soft.LogicalGap), all experiments
  * exact     - exact posterior P(flip | detections) (qeccal.exact), r = 1 at d = 3 and d = 5

Per-shot outputs go to results/cache/soft/<method>/<prior>/<key>.npz (git-ignored, about 170 MB).
Per-experiment counts go to results/m2_soft_outputs.csv (committed): errors of each method on all shots, and
for the gap, the number of tied shots (gap = 0). Resumable: finished (method, prior, key) are skipped.

    python scripts/m2_soft_outputs.py              # everything, 8 workers
    python scripts/m2_soft_outputs.py --distance 3 --rounds 1 10
"""
import argparse
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np

from qeccal.data import get_experiment, list_experiments
from qeccal.exact import ExactPosterior
from qeccal.soft import LogicalGap
from qeccal.soft.cache import cache_path, save

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "m2_soft_outputs.csv"
PRIORS = ("si1000", "rl")
FIELDS = ["key", "distance", "patch", "basis", "rounds", "method", "prior", "shots", "errors", "ties", "seconds"]


def run(key, method):
    e = get_experiment(key)
    obs = e.observable_flips()
    rows = []
    for prior in PRIORS:
        path = cache_path(method, prior, key)
        if path.exists():
            continue
        t0 = time.time()
        if method == "mwpm_gap":
            pred, gap = LogicalGap(e.dem(prior)).decode(e.detection_events(packed=True), packed=True)
            extra = {"gap": gap.astype(np.float32)}
            ties = int((gap == 0).sum())
        else:
            p_flip = ExactPosterior(e.dem(prior)).p_flip(e.detection_events())
            pred = p_flip > 0.5
            extra = {"p_flip": p_flip}
            ties = int((p_flip == 0.5).sum())
        save(method, prior, key, pred, **extra)
        rows.append({"key": key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds,
                     "method": method, "prior": prior, "shots": e.shots, "errors": int((pred != obs).sum()),
                     "ties": ties, "seconds": round(time.time() - t0, 2)})
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
    tasks = [(e, "mwpm_gap") for e in exps]
    tasks += [(e, "exact") for e in exps if e.rounds == 1 and e.distance in (3, 5)]
    todo = [(e, m) for e, m in tasks if not all(cache_path(m, p, e.key).exists() for p in PRIORS)]
    todo.sort(key=lambda t: -(t[0].distance ** 2 * t[0].rounds + (10_000 if t[1] == "exact" and t[0].distance == 5 else 0)))
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} cached, {len(todo)} to run", flush=True)
    if not todo:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    t0 = time.time()
    with out.open("a", newline="") as f, cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        futs = {pool.submit(run, e.key, m): (e.key, m) for e, m in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rows = fut.result()
            w.writerows(rows)
            f.flush()
            key, m = futs[fut]
            print(f"[{i}/{len(todo)}] {m:8s} {key}  " + "  ".join(f"{r['prior']}: {r['errors']} err" for r in rows)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
