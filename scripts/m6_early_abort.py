"""M6, step 2: early abort learned by Q-iteration and certified by Learn-then-Test (spec section 8.8).

Per experiment (r in 10, 30, 50; MWPM gap, RL prior; prefix gaps from scripts/m6_prefix_gaps.py):
  * state bins from Train quantiles: 20 per checkpoint (prefix gap), 40 for the final gap;
  * policy family from Train: for each keep threshold j on the final-gap bins, the abort policy with the most
    kept shots per round (Dinkelbach on the per-round cost rho, Q-iteration for each rho);
  * Learn-then-Test on Calibrate over the family, strict to loose, delta = 0.05;
  * evaluation on Test.
Rules compared at each alpha:
  static       the M4 rule: Learn-then-Test on the final gap, 5% keep grid, no early abort
  final_only   the policy family with rho = 0: the same keep thresholds, never aborts (ablation)
  early_abort  the policy family with early abort
Uncertified experiments keep nothing (throughput 0), as in M4.

Check: for the keep threshold at the Train median of the final gap, the batch Q-table is recomputed by
sample-by-sample Q-learning (q_learning_online).

    python scripts/m6_early_abort.py [--workers 10]
Writes results/m6_early_abort.csv, results/m6_policies.csv and results/m6_qcheck.csv.
"""
import argparse
import concurrent.futures as cf
import csv
import json
import pathlib
import time

import numpy as np

from qeccal.data import list_experiments, split_indices
from qeccal.data.willow import get_experiment
from qeccal.guarantees import certify, cp_lower
from qeccal.sequential import (Binner, abort_below, certify_family, evaluate, policy_family, q_learning_online, run,
                               solve_rate)
from qeccal.soft.cache import load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
PREFIX = RES / "cache" / "prefix" / "rl"
ROUNDS = (10, 30, 50)
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DELTA = 0.05
NBINS_PREFIX, NBINS_FINAL = 20, 40
FIELDS = ["key", "distance", "patch", "basis", "rounds", "alpha", "rule", "certified", "keep_train", "keep_from", "rho",
          "cal_n", "cal_errors", "test_shots", "test_n", "test_errors", "test_rate", "exceed", "sig_exceed", "keep",
          "rounds_mean", "aborted", "throughput"]
P_FIELDS = ["key", "distance", "patch", "basis", "rounds", "alpha", "keep_from", "rho", "keep_train", "checkpoints",
            "abort_below", "abort_share_test"]
Q_FIELDS = ["key", "distance", "rounds", "keep_from", "rho", "max_abs_diff", "same_decisions"]


def prefix_path(key):
    return PREFIX / (key.replace("/", "__") + ".npz")


def run_one(key):
    t0 = time.time()
    e = get_experiment(key)
    idx = split_indices(e)
    tr, ca, te = idx["train"], idx["calibrate"], idx["test"]
    o = load("mwpm_gap", "rl", key)
    wrong = o["pred"] != e.observable_flips()
    s = o["gap"].astype(np.float64)
    z = np.load(prefix_path(key))
    X = np.column_stack([z["gaps"].astype(np.float64), s])
    rounds = np.array([*z["checkpoints"].tolist(), e.rounds], dtype=np.float64)
    K = rounds.size - 1
    bn = Binner.fit(X[tr], [NBINS_PREFIX] * K + [NBINS_FINAL])
    B = bn.transform(X)
    fams = {"early_abort": policy_family(B[tr], rounds, bn.sizes),
            "final_only": policy_family(B[tr], rounds, bn.sizes, abort=False)}
    base = {"key": key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds}
    rows, pols = [], []
    for alpha in ALPHAS:
        row = {**base, "alpha": alpha, "rule": "static", "test_shots": te.size}
        res = certify(s[tr], s[ca], wrong[ca], alpha, DELTA)
        if res is None:
            row.update(certified=0, keep=0.0, throughput=0.0)
        else:
            lam, k = res
            kc, kt = s[ca] >= lam, s[te] >= lam
            n, err = int(kt.sum()), int(wrong[te][kt].sum())
            rate = err / n if n else float("nan")
            row.update(certified=1, keep_train=k, cal_n=int(kc.sum()), cal_errors=int(wrong[ca][kc].sum()), test_n=n,
                       test_errors=err, test_rate=rate, exceed=int(n > 0 and rate > alpha),
                       sig_exceed=int(cp_lower(err, n) > alpha), keep=n / te.size, rounds_mean=float(e.rounds),
                       aborted=0.0, throughput=n / (e.rounds * te.size))
        rows.append(row)
        for rule, fam in fams.items():
            row = {**base, "alpha": alpha, "rule": rule, "test_shots": te.size}
            i = certify_family(fam, B[ca], wrong[ca], alpha, DELTA)
            if i is None:
                row.update(certified=0, keep=0.0, throughput=0.0)
                rows.append(row)
                continue
            kf, pol = fam[i]
            ev, cal = evaluate(pol, B[te], wrong[te], alpha), evaluate(pol, B[ca], wrong[ca], alpha)
            row.update(certified=1, keep_train=kf, keep_from=pol.keep_from, rho=pol.rho, cal_n=cal["n"], cal_errors=cal["errors"],
                       test_n=ev["n"], test_errors=ev["errors"], test_rate=ev["rate"], exceed=ev["exceed"],
                       sig_exceed=ev["sig_exceed"], keep=ev["keep"], rounds_mean=ev["rounds_mean"],
                       aborted=ev["aborted"], throughput=ev["throughput"])
            rows.append(row)
            if rule == "early_abort":
                _, stop = run(pol, B[te])
                share = [float((stop == t).mean()) for t in rounds[:K]]
                pols.append({**base, "alpha": alpha, "keep_from": pol.keep_from, "rho": pol.rho, "keep_train": kf,
                             "checkpoints": json.dumps(rounds[:K].astype(int).tolist()),
                             "abort_below": json.dumps([round(x, 4) if np.isfinite(x) else None
                                                        for x in abort_below(pol, bn)]),
                             "abort_share_test": json.dumps([round(x, 5) for x in share])})
    j = int(np.median(B[tr, K]))
    batch = solve_rate(B[tr], rounds, j, bn.sizes)
    online = q_learning_online(B[tr], rounds, j, batch.rho, bn.sizes)
    diff = max(float(np.max(np.abs(a - b))) for a, b in zip(batch.q_cont, online.q_cont)) if K else 0.0
    q = {"key": key, "distance": e.distance, "rounds": e.rounds, "keep_from": j, "rho": batch.rho, "max_abs_diff": diff,
         "same_decisions": int(batch.decisions() == online.decisions())}
    return rows, pols, q, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    keys = [e.key for e in list_experiments(rounds=ROUNDS) if prefix_path(e.key).exists()]
    print(f"{len(keys)} experiments", flush=True)
    rows, pols, qs = [], [], []
    t0 = time.time()
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for fut in cf.as_completed([ex.submit(run_one, k) for k in keys]):
            r, p, q, sec = fut.result()
            rows += r
            pols += p
            qs.append(q)
            print(f"{q['key']}  {sec:.0f} s  (elapsed {time.time() - t0:.0f} s)", flush=True)
    order = lambda d: (d["distance"], d["rounds"], d.get("patch", ""), d.get("basis", ""), d.get("alpha", 0), d["key"])
    for name, fields, data in [("m6_early_abort.csv", FIELDS, rows), ("m6_policies.csv", P_FIELDS, pols),
                               ("m6_qcheck.csv", Q_FIELDS, qs)]:
        with open(RES / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for d in sorted(data, key=order):
                w.writerow(d)
    print("done", flush=True)
