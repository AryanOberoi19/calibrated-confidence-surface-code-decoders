"""M4: post-hoc calibration (RQ1) and certified abort thresholds (RQ2), per experiment.

Split discipline (spec 8.4): calibrators and the threshold grid are fitted on Train; Calibrate is used only
to select thresholds; Test only to evaluate. r = 13 has no Calibrate or Test shots and is skipped.

Score s per shot: the MWPM gap, the belief-matching gap (RL prior, r in 1, 10, 30, 50; scripts/m2_belief_gap.py),
or log((1 - q) / q) for the exact posterior (r = 1) and the learned decoder (d = 3; scripts/m3_learned.py; its
Train split excludes the shots it was fine-tuned on). Combinations without a cache are skipped. Threshold rules, each
choosing a keep fraction from the same Train-defined grid (qeccal.guarantees.KEEP_GRID):
  ltt          Learn-then-Test, fixed sequence, delta = 0.05: certified
  plugin       loosest threshold whose Calibrate error among kept is <= alpha (no margin)
  naive_raw    loosest threshold whose mean raw q among kept Calibrate shots is <= alpha (trusts the model)
  naive_platt  the same with Platt-calibrated q (fitted on Train)

    python scripts/m4_calibrate_certify.py
Writes results/m4_calibration.csv and results/m4_guarantees.csv.
"""
import argparse
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np

from qeccal.calibration import CALIBRATORS, brier, ece, nll, score_from_q
from qeccal.data import list_experiments, split_indices
from qeccal.data.willow import get_experiment
from qeccal.guarantees import KEEP_GRID, cp_lower, learn_then_test, loosest_where, thresholds
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DELTA = 0.05
COMBOS = [("mwpm_gap", "si1000"), ("mwpm_gap", "rl"), ("exact", "si1000"), ("exact", "rl"), ("bm_gap", "rl"), ("nn", "rl")]
CAL_FIELDS = ["key", "distance", "patch", "basis", "rounds", "method", "prior", "calibrator", "test_n",
              "test_errors", "mean_q", "ece", "brier", "nll", "params"]
G_FIELDS = ["key", "distance", "patch", "basis", "rounds", "method", "prior", "alpha", "rule", "keep",
            "cal_n", "cal_errors", "test_n", "test_errors", "test_rate", "exceed", "sig_exceed"]


def run(key):
    e = get_experiment(key)
    idx = split_indices(e)
    obs = e.observable_flips()
    base = {"key": key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds}
    cal_rows, g_rows = [], []
    for method, prior in COMBOS:
        if not cache_path(method, prior, key).exists():
            continue
        o = load(method, prior, key)
        s = o["gap"].astype(np.float64) if "gap" in o else score_from_q(o["q"])
        wrong = o["pred"] != obs
        tr, ca, te = idx["train"], idx["calibrate"], idx["test"]
        if "fit_mask" in o:                                # learned decoder: drop the Train shots it was trained on
            tr = tr[~o["fit_mask"][tr]]
        fitted = {name: C().fit(s[tr], wrong[tr]) for name, C in CALIBRATORS.items()}
        for name, c in fitted.items():
            q = c.predict(s[te])
            cal_rows.append({**base, "method": method, "prior": prior, "calibrator": name, "test_n": te.size,
                             "test_errors": int(wrong[te].sum()), "mean_q": float(q.mean()),
                             "ece": ece(q, wrong[te]), "brier": brier(q, wrong[te]), "nll": nll(q, wrong[te]),
                             "params": ";".join(f"{k}={v:.4g}" for k, v in c.params().items())})

        lams = thresholds(s[tr])
        s_ca, w_ca, s_te, w_te = s[ca], wrong[ca], s[te], wrong[te]
        kept_ca = [s_ca >= lam for lam in lams]
        cal_err = [w_ca[k].mean() if k.any() else np.inf for k in kept_ca]
        q_raw = fitted["raw"].predict(s_ca)
        q_platt = fitted["platt"].predict(s_ca)
        mean_raw = [q_raw[k].mean() if k.any() else np.inf for k in kept_ca]
        mean_platt = [q_platt[k].mean() if k.any() else np.inf for k in kept_ca]
        for alpha in ALPHAS:
            choice = {
                "ltt": learn_then_test(s_ca, w_ca, lams, alpha, DELTA)[0],
                "plugin": loosest_where(cal_err, lambda v: v <= alpha),
                "naive_raw": loosest_where(mean_raw, lambda v: v <= alpha),
                "naive_platt": loosest_where(mean_platt, lambda v: v <= alpha),
            }
            for rule, i in choice.items():
                row = {**base, "method": method, "prior": prior, "alpha": alpha, "rule": rule}
                if i is None:
                    g_rows.append({**row, "keep": "", "cal_n": "", "cal_errors": "", "test_n": "", "test_errors": "",
                                   "test_rate": "", "exceed": "", "sig_exceed": ""})
                    continue
                kt = s_te >= lams[i]
                n, er = int(kt.sum()), int(w_te[kt].sum())
                g_rows.append({**row, "keep": KEEP_GRID[i], "cal_n": int(kept_ca[i].sum()),
                               "cal_errors": int(w_ca[kept_ca[i]].sum()), "test_n": n, "test_errors": er,
                               "test_rate": er / n if n else "", "exceed": int(n > 0 and er / n > alpha),
                               "sig_exceed": int(cp_lower(er, n) > alpha)})
    return cal_rows, g_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    keys = [e.key for e in list_experiments() if "test" in split_indices(e)]
    t0 = time.time()
    with (RES / "m4_calibration.csv").open("w", newline="") as fc, (RES / "m4_guarantees.csv").open("w", newline="") as fg, \
            cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        wc, wg = csv.DictWriter(fc, fieldnames=CAL_FIELDS), csv.DictWriter(fg, fieldnames=G_FIELDS)
        wc.writeheader()
        wg.writeheader()
        results = {}
        for i, (key, res) in enumerate(zip(keys, pool.map(run, keys, chunksize=4)), 1):
            results[key] = res
            if i % 50 == 0:
                print(f"{i}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
        for key in keys:                                   # deterministic row order
            wc.writerows(results[key][0])
            wg.writerows(results[key][1])
    print(f"done in {time.time() - t0:.0f}s: {len(keys)} experiments -> results/m4_calibration.csv, m4_guarantees.csv")


if __name__ == "__main__":
    main()
