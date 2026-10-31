"""M6 drift: does a certified early-abort policy survive a shift from source to target? (spec 8.8; the RQ3 test
of M5, repeated for policies)

For every (source, target) pair with r in 10, 30 and 50 (MWPM gap and prefix gap, RL prior): fit the bins and learn
the policy family on the source Train split, certify one policy with Learn-then-Test on the source Calibrate split
(delta = 0.05), then play it unchanged on the target's Test split. The target's prefix and final gaps are binned with
the source's bin edges, so the abort thresholds and the keep threshold move to the target as gap values: the
analogue of M5's threshold transfer. Axes as in M5:

  same    source = target (the in-distribution reference; the same numbers as scripts/m6_early_abort.py)
  patch   another patch, same distance, basis and round count (all ordered pairs)
  basis   the same patch and round count, other basis
  pooled  all other patches of that distance, basis and round count, pooled (needs at least two)
  sim     35,000 shots simulated from the experiment's own RL-prior DEM (20,000 Train, 15,000 Calibrate)

    python scripts/m6_drift.py [--workers 9]
Writes results/m6_drift.csv; simulated prefix gaps are cached in results/cache/sim_prefix/rl/.
"""
import argparse
import collections
import concurrent.futures as cf
import csv
import pathlib
import time

import numpy as np

from qeccal.data import list_experiments, split_indices
from qeccal.data.splits import SEED, _key_seed
from qeccal.data.willow import get_experiment
from qeccal.sequential import (Binner, certify_family, checkpoints_for, detector_steps, evaluate, policy_family,
                               prefix_gaps)
from qeccal.soft.cache import load
from qeccal.soft.gap import LogicalGap

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
PREFIX = RES / "cache" / "prefix" / "rl"
SIM = RES / "cache" / "sim_prefix" / "rl"
ROUNDS = (10, 30, 50)
ALPHAS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DELTA = 0.05
NBINS_PREFIX, NBINS_FINAL = 20, 40          # as in scripts/m6_early_abort.py
N_TRAIN, N_CAL = 20_000, 15_000
FIELDS = ["axis", "source", "target", "distance", "basis", "rounds", "patch", "alpha", "certified", "src_cal_n",
          "keep_from", "test_n", "test_errors", "test_rate", "exceed", "sig_exceed", "keep", "rounds_mean", "aborted",
          "throughput"]


def _name(key):
    return key.replace("/", "__") + ".npz"


def simulate(key):
    path = SIM / _name(key)
    if path.exists():
        return key
    e = get_experiment(key)
    dem = e.dem("rl")
    seed = int(np.random.default_rng([SEED, _key_seed(key), 7]).integers(2 ** 31))
    det, obs, _ = dem.compile_sampler(seed=seed).sample(N_TRAIN + N_CAL)
    gaps, _ = prefix_gaps(dem, detector_steps(e.circuit()), det, checkpoints_for(e.rounds))
    pred, gap = LogicalGap(dem).decode(det)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=np.column_stack([gaps, gap.astype(np.float32)]),
                        wrong=np.packbits(pred != obs[:, 0], bitorder="little"))
    tmp.rename(path)
    return key


def load_hw(e):
    idx = split_indices(e)
    o = load("mwpm_gap", "rl", e.key)
    w = o["pred"] != e.observable_flips()
    X = np.column_stack([np.load(PREFIX / _name(e.key))["gaps"].astype(np.float64), o["gap"].astype(np.float64)])
    return {r: (X[idx[r]], w[idx[r]]) for r in ("train", "calibrate", "test")}


def load_sim(key):
    z = np.load(SIM / _name(key))
    X = z["X"].astype(np.float64)
    w = np.unpackbits(z["wrong"], count=X.shape[0], bitorder="little").astype(bool)
    return {"train": (X[:N_TRAIN], w[:N_TRAIN]), "calibrate": (X[N_TRAIN:], w[N_TRAIN:])}


def pool(parts):
    return {r: (np.concatenate([p[r][0] for p in parts]), np.concatenate([p[r][1] for p in parts]))
            for r in ("train", "calibrate")}


def certify_source(src, rounds):
    K = rounds.size - 1
    bn = Binner.fit(src["train"][0], [NBINS_PREFIX] * K + [NBINS_FINAL])
    fam = policy_family(bn.transform(src["train"][0]), rounds, bn.sizes)
    Bc = bn.transform(src["calibrate"][0])
    cert = {a: certify_family(fam, Bc, src["calibrate"][1], a, DELTA) for a in ALPHAS}
    return bn, fam, cert, Bc.shape[0]


def run_group(keys):
    """All pairs whose target is in this (distance, rounds) group."""
    exps = [get_experiment(k) for k in keys]
    by_key = {e.key: e for e in exps}
    hw = {e.key: load_hw(e) for e in exps}
    rounds = np.array([*checkpoints_for(exps[0].rounds), exps[0].rounds], dtype=np.float64)
    peers = collections.defaultdict(list)
    for e in exps:
        peers[e.basis].append(e)
    pairs = []
    for e in exps:
        pairs.append(("same", e.key, lambda e=e: hw[e.key], e))
        pairs.append(("sim", f"sim:{e.key}", lambda e=e: load_sim(e.key), e))
        other = e.key.replace(f"/{e.basis}/", "/Z/" if e.basis == "X" else "/X/")
        if other in by_key:
            pairs.append(("basis", other, lambda o=other: hw[o], e))
        ps = [p for p in peers[e.basis] if p.key != e.key]
        for p in ps:
            pairs.append(("patch", p.key, lambda p=p: hw[p.key], e))
        if len(ps) >= 2:
            pairs.append(("pooled", f"pool:d{e.distance}/{e.basis}/r{e.rounds} without {e.patch} ({len(ps)} patches)",
                          lambda ps=ps: pool([hw[p.key] for p in ps]), e))
    sources, rows = {}, []
    for axis, label, get_src, tgt in pairs:
        if label not in sources:
            sources[label] = certify_source(get_src(), rounds)
        bn, fam, cert, n_cal = sources[label]
        X_te, w_te = hw[tgt.key]["test"]
        B_te = bn.transform(X_te)
        base = {"axis": axis, "source": label, "target": tgt.key, "distance": tgt.distance, "basis": tgt.basis,
                "rounds": tgt.rounds, "patch": tgt.patch, "src_cal_n": n_cal}
        for alpha in ALPHAS:
            i = cert[alpha]
            if i is None:
                rows.append({**base, "alpha": alpha, "certified": 0})
                continue
            pol = fam[i][1]
            ev = evaluate(pol, B_te, w_te, alpha)
            rows.append({**base, "alpha": alpha, "certified": 1, "keep_from": pol.keep_from, "test_n": ev["n"],
                         "test_errors": ev["errors"], "test_rate": ev["rate"], "exceed": ev["exceed"],
                         "sig_exceed": ev["sig_exceed"], "keep": ev["keep"], "rounds_mean": ev["rounds_mean"],
                         "aborted": ev["aborted"], "throughput": ev["throughput"]})
    return rows, len(pairs), len(sources)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=9)
    args = ap.parse_args()
    exps = [e for e in list_experiments(rounds=ROUNDS) if (PREFIX / _name(e.key)).exists()]
    t0 = time.time()
    todo = [e.key for e in exps if not (SIM / _name(e.key)).exists()]
    todo.sort(key=lambda k: -get_experiment(k).distance ** 2 * get_experiment(k).rounds)
    print(f"{len(exps)} experiments; simulating {len(todo)}", flush=True)
    with cf.ProcessPoolExecutor(args.workers) as ex:
        for i, _ in enumerate(ex.map(simulate, todo), 1):
            if i % 12 == 0:
                print(f"  simulated {i}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    groups = collections.defaultdict(list)
    for e in exps:
        groups[(e.distance, e.rounds)].append(e.key)
    rows = []
    with cf.ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_group, keys): g for g, keys in groups.items()}
        for fut in cf.as_completed(futs):
            r, npairs, nsrc = fut.result()
            rows += r
            d, rr = futs[fut]
            print(f"d={d} r={rr}: {npairs} pairs, {nsrc} sources ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda x: (x["distance"], x["rounds"], x["target"], x["axis"], x["source"], x["alpha"]))
    with (RES / "m6_drift.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"done in {time.time() - t0:.0f} s: {len(rows)} rows", flush=True)
