"""M5 / RQ3: does a certified abort threshold (and a fitted calibrator) survive a shift from source to target?

Scores: the MWPM gap under both priors; under the RL prior also the belief-matching gap and, at d = 3, the
learned decoder's |logit| (on the experiments their caches cover; scripts/m2_belief_gap.py, scripts/m3_learned.py). For every (source, target) pair: fit calibrators on the source Train split,
select the threshold with Learn-then-Test on the source Calibrate split (delta = 0.05), then measure it on the
target's Test split. Shift axes (spec 8.7):

  same      source = target (the in-distribution reference, as in M4)
  patch     another patch, same distance, basis and round count (all ordered pairs)
  basis     the same patch and round count, other basis
  pooled    leave-one-patch-out: all other patches of that distance, basis and round count pooled as the source
  sim       shots simulated from the experiment's own DEM (the prior), decoded with that DEM

Round counts 1, 10, 30, 50 (beyond 50 almost nothing is certifiable, see M4). The sim axis is run for the MWPM
gap only (belief-matching on simulated shots would cost another ~10 core-hours).

Two ways to move a certified threshold to the target (column `transfer`):
  threshold      use the source's gap threshold lambda as is
  keep_fraction  keep the same fraction of the target's shots as was certified on the source: the threshold is
                 re-set at the (1 - keep) quantile of the target's own Train scores, which needs no labels

    python scripts/m5_drift.py
Writes results/m5_drift.csv and results/m5_calibration.csv. Simulated shots are cached in results/cache/sim/.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np

from qeccal.calibration import Isotonic, Platt, ece, nll, score_from_q
from qeccal.data import get_experiment, list_experiments, split_indices
from qeccal.data.splits import SEED, _key_seed
from qeccal.guarantees import certify, evaluate, threshold_for_keep
from qeccal.soft import sample_gap
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
SIM_CACHE = RES / "cache" / "sim"
ROUNDS = (1, 10, 30, 50)
ALPHAS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DELTA = 0.05
PRIORS = ("si1000", "rl")
COMBOS = (("mwpm_gap", "si1000"), ("mwpm_gap", "rl"), ("bm_gap", "rl"), ("nn", "rl"))
TRANSFERS = ("threshold", "keep_fraction")
N_TRAIN, N_CAL = 20_000, 15_000
G_FIELDS = ["axis", "source", "target", "distance", "basis", "rounds", "patch", "method", "prior", "alpha", "transfer",
            "certified", "keep", "src_cal_n", "test_n", "test_errors", "test_rate", "exceed", "sig_exceed"]
C_FIELDS = ["axis", "source", "target", "distance", "basis", "rounds", "patch", "method", "prior", "calibrator", "ece",
            "nll"]


def sim_path(prior, key):
    return SIM_CACHE / prior / (key.replace("/", "__") + ".npz")


def simulate(key, prior):
    path = sim_path(prior, key)
    if not path.exists():
        seed = int(np.random.default_rng([SEED, _key_seed(key), PRIORS.index(prior)]).integers(2 ** 31))
        gap, wrong = sample_gap(get_experiment(key).dem(prior), N_TRAIN + N_CAL, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, gap=gap.astype(np.float32), wrong=np.packbits(wrong, bitorder="little"))
    return key, prior


def load_sim(prior, key):
    z = np.load(sim_path(prior, key))
    s = z["gap"].astype(np.float64)
    w = np.unpackbits(z["wrong"], count=s.size, bitorder="little").astype(bool)
    return {"train": (s[:N_TRAIN], w[:N_TRAIN]), "calibrate": (s[N_TRAIN:], w[N_TRAIN:])}


def load_hw(e, method, prior):
    idx = split_indices(e)
    o = load(method, prior, e.key)
    s = o["gap"].astype(np.float64) if "gap" in o else score_from_q(o["q"])
    w = o["pred"] != e.observable_flips()
    if "fit_mask" in o:                                    # learned decoder: drop the Train shots it was trained on
        idx["train"] = idx["train"][~o["fit_mask"][idx["train"]]]
    return {r: (s[idx[r]], w[idx[r]]) for r in ("train", "calibrate", "test")}


def pool(parts):
    return {r: (np.concatenate([p[r][0] for p in parts]), np.concatenate([p[r][1] for p in parts]))
            for r in ("train", "calibrate")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    exps = list_experiments(rounds=ROUNDS)
    t0 = time.time()

    todo = [(e.key, p) for e in exps for p in PRIORS if not sim_path(p, e.key).exists()]
    print(f"{len(exps)} experiments; simulating {len(todo)} (experiment, prior) sources", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, _ in enumerate(ex.map(simulate, *zip(*todo)) if todo else [], 1):
            if i % 40 == 0:
                print(f"  simulated {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)

    g_rows, c_rows = [], []
    for method, prior in COMBOS:
        cov = [e for e in exps if cache_path(method, prior, e.key).exists()]
        print(f"{method}/{prior}: {len(cov)} of {len(exps)} experiments in the soft-output cache", flush=True)
        if not cov:
            continue
        by_key = {e.key: e for e in cov}
        groups = collections.defaultdict(list)      # (d, basis, r) -> experiments over patches
        for e in cov:
            groups[(e.distance, e.basis, e.rounds)].append(e)
        hw = {e.key: load_hw(e, method, prior) for e in cov}
        pairs = []                                   # (axis, source label, source arrays, target experiment)
        for e in cov:
            pairs.append(("same", e.key, hw[e.key], e))
            if method == "mwpm_gap":
                pairs.append(("sim", f"sim:{e.key}", load_sim(prior, e.key), e))
            other = e.key.replace(f"/{e.basis}/", "/Z/" if e.basis == "X" else "/X/")
            if other in by_key:
                pairs.append(("basis", other, hw[other], e))
            peers = [p for p in groups[(e.distance, e.basis, e.rounds)] if p.key != e.key]
            for p in peers:
                pairs.append(("patch", p.key, hw[p.key], e))
            if len(peers) >= 2:
                pairs.append(("pooled", f"pool:d{e.distance}/{e.basis}/r{e.rounds} without {e.patch} ({len(peers)} patches)",
                              pool([hw[p.key] for p in peers]), e))

        fits, certs = {}, {}                          # per source label: calibrators, certified thresholds
        for axis, label, src, tgt in pairs:
            base = {"axis": axis, "source": label, "target": tgt.key, "distance": tgt.distance, "basis": tgt.basis,
                    "rounds": tgt.rounds, "patch": tgt.patch, "method": method, "prior": prior}
            s_te, w_te = hw[tgt.key]["test"]
            s_unlabelled = hw[tgt.key]["train"][0]
            if label not in fits:
                fits[label] = {c.name: c().fit(*src["train"]) for c in (Platt, Isotonic)}
            for name, c in fits[label].items():
                q = c.predict(s_te)
                c_rows.append({**base, "calibrator": name, "ece": ece(q, w_te), "nll": nll(q, w_te)})
            for alpha in ALPHAS:
                if (label, alpha) not in certs:
                    certs[(label, alpha)] = certify(src["train"][0], src["calibrate"][0], src["calibrate"][1], alpha, DELTA)
                res = certs[(label, alpha)]
                for transfer in TRANSFERS:
                    row = {**base, "alpha": alpha, "transfer": transfer, "src_cal_n": src["calibrate"][0].size}
                    if res is None:
                        g_rows.append({**row, "certified": 0, "keep": "", "test_n": "", "test_errors": "",
                                       "test_rate": "", "exceed": "", "sig_exceed": ""})
                        continue
                    lam, keep = res
                    if transfer == "keep_fraction":
                        lam = threshold_for_keep(s_unlabelled, keep)
                    ev = evaluate(lam, s_te, w_te, alpha)
                    g_rows.append({**row, "certified": 1, "keep": keep, "test_n": ev["n"], "test_errors": ev["errors"],
                                   "test_rate": ev["rate"], "exceed": ev["exceed"], "sig_exceed": ev["sig_exceed"]})
        print(f"{method}/{prior}: {len(pairs)} source-target pairs ({time.time() - t0:.0f}s)", flush=True)

    for name, rows, fields in (("m5_drift.csv", g_rows, G_FIELDS), ("m5_calibration.csv", c_rows, C_FIELDS)):
        with (RES / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s: {len(g_rows)} guarantee rows, {len(c_rows)} calibration rows")


if __name__ == "__main__":
    main()
