"""A7: preregistered confirmatory run on the untouched r = 13 experiments (arXiv v1 plan, task A7).

Everything here is fixed in docs/prereg_r13.md before the r = 13 labels are combined with any score. The run is
done once, after the preregistration commit is tagged v0.2.0 AND pushed to the public remote; it refuses to
overwrite its own outputs.

Data: the 28 experiments at r = 13, MWPM gap under the SI1000 prior (the RL prior was fitted on r = 13).
Split: qeccal.data.splits_extra.confirmatory_indices (seed 20261117; manifest splits/splits_r13_v1.json).
C2: per experiment, thresholds from Train, calibrators on Train, rules ltt, plugin, naive_raw, naive_platt,
    bonferroni, sgr, ltt_hb at M4's alphas, delta = 0.05; Test exceedance and significant exceedance (one-sided
    95% Clopper-Pearson).
C4: Learn-then-Test certificates moved between patches (all ordered pairs, same distance and basis) at M5's
    alphas, transfers threshold (primary) and keep_fraction; the same-experiment rows are the reference. A6
    statistics: pair p-value, BH at 5% within (distance, transfer, axis, alpha), 90% cluster-bootstrap interval
    (2,000 resamples; two-way over source and target patch for the patch axis, one-way for same).

    python scripts/p6_confirmatory_r13.py --run                   # the confirmatory run (once)
    python scripts/p6_confirmatory_r13.py --dry-run-rounds 10     # same code on already-examined r = 10 data,
                                                                  # outputs under results/cache/p6_dryrun/
"""
import argparse
import csv
import pathlib
import subprocess

import numpy as np
import pandas as pd

from qeccal.calibration import CALIBRATORS
from qeccal.data import list_experiments
from qeccal.data.splits_extra import confirmatory_indices
from qeccal.guarantees import (KEEP_GRID, bonferroni, certify, cp_lower, evaluate, learn_then_test, loosest_where,
                               ltt_hb, sgr, threshold_for_keep, thresholds)
from qeccal.shift import bh, cluster_bootstrap_share, pair_p_value
from qeccal.soft.cache import load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
M4_ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
M5_ALPHAS = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
DELTA = 0.05
METHOD, PRIOR = "mwpm_gap", "si1000"
RULES = ("ltt", "plugin", "naive_raw", "naive_platt", "bonferroni", "sgr", "ltt_hb")


def data(e):
    idx = confirmatory_indices(e.key, e.shots)
    o = load(METHOD, PRIOR, e.key)
    s = o["gap"].astype(np.float64)
    w = o["pred"] != e.observable_flips()
    return {r: (s[idx[r]], w[idx[r]]) for r in ("train", "calibrate", "test")}


def c2_rows(e, d):
    (s_tr, w_tr), (s_ca, w_ca), (s_te, w_te) = d["train"], d["calibrate"], d["test"]
    lams = thresholds(s_tr)
    fit = {n: CALIBRATORS[n]().fit(s_tr, w_tr) for n in ("raw", "platt")}
    kept = [s_ca >= lam for lam in lams]
    err = [w_ca[k].mean() if k.any() else np.inf for k in kept]
    mq = {n: [fit[n].predict(s_ca)[k].mean() if k.any() else np.inf for k in kept] for n in fit}
    rows = []
    for a in M4_ALPHAS:
        ch = {"ltt": learn_then_test(s_ca, w_ca, lams, a, DELTA)[0], "plugin": loosest_where(err, lambda v: v <= a),
              "naive_raw": loosest_where(mq["raw"], lambda v: v <= a), "naive_platt": loosest_where(mq["platt"], lambda v: v <= a),
              "bonferroni": bonferroni(s_ca, w_ca, lams, a, DELTA), "sgr": sgr(s_ca, w_ca, lams, a, DELTA),
              "ltt_hb": ltt_hb(s_ca, w_ca, lams, a, DELTA)}
        for r in RULES:
            row = {"key": e.key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "alpha": a, "rule": r}
            i = ch[r]
            if i is None:
                rows.append({**row, "keep": "", "test_n": "", "test_errors": "", "exceed": "", "sig_exceed": ""})
                continue
            ev = evaluate(lams[i], s_te, w_te, a)
            rows.append({**row, "keep": KEEP_GRID[i], "test_n": ev["n"], "test_errors": ev["errors"],
                         "exceed": ev["exceed"], "sig_exceed": ev["sig_exceed"]})
    return rows


def c4_rows(exps, D):
    rows = []
    certs = {(e.key, a): certify(D[e.key]["train"][0], *D[e.key]["calibrate"], a, DELTA) for e in exps for a in M5_ALPHAS}
    for t in exps:
        for s in exps:
            if s.distance != t.distance or s.basis != t.basis or (s.key != t.key and s.patch == t.patch):
                continue
            axis = "same" if s.key == t.key else "patch"
            for a in M5_ALPHAS:
                c = certs[(s.key, a)]
                if c is None:
                    continue
                for tr in ("threshold", "keep_fraction"):
                    lam = c[0] if tr == "threshold" else threshold_for_keep(D[t.key]["train"][0], c[1])
                    ev = evaluate(lam, *D[t.key]["test"], a)
                    rows.append({"axis": axis, "source": s.key, "target": t.key, "distance": t.distance, "basis": t.basis,
                                 "source_patch": s.patch, "target_patch": t.patch, "alpha": a, "transfer": tr,
                                 "keep": c[1], "test_n": ev["n"], "test_errors": ev["errors"],
                                 "p": pair_p_value(ev["n"], ev["errors"], a)})
    df = pd.DataFrame(rows)
    df["fail"] = False
    for _, g in df.groupby(["distance", "transfer", "axis", "alpha"]):
        df.loc[g.index, "fail"] = bh(g.p.values, 0.05)
    return df


def summary(c2, pairs, tag, n_exp):
    sh = []
    for (dist, tr, ax, a), g in pairs.groupby(["distance", "transfer", "axis", "alpha"]):
        rng = np.random.default_rng([20261117, 6, int(dist), ("threshold", "keep_fraction").index(tr), ("same", "patch").index(ax),
                                     M5_ALPHAS.index(a)])
        src = g.source_patch.values if ax == "patch" else g.target_patch.values
        share, lo, hi = cluster_bootstrap_share(g.fail.values, src, g.target_patch.values, 2000, rng)
        sh.append({"distance": dist, "transfer": tr, "axis": ax, "alpha": a, "pairs": len(g), "failing": int(g.fail.sum()),
                   "share": share, "lo": lo, "hi": hi})
    sh = pd.DataFrame(sh)
    L = [f"# A7: confirmatory run on r = 13 (preregistered at tag v0.2.0 = {tag})", "",
         f"{n_exp} experiments, {METHOD}/{PRIOR}, split seed 20261117. Endpoints as stated in docs/prereg_r13.md.", "",
         "## Endpoints 1 and 2 (C2)", "", "| alpha | rule | selected | exceeds | significantly exceeds |", "|---|---|---|---|---|"]
    sel = c2[c2.keep != ""].copy()
    for c in ("exceed", "sig_exceed"):
        sel[c] = sel[c].astype(int)
    e1, e2 = [], []
    for a in M4_ALPHAS:
        for r in RULES:
            v = sel[(sel.alpha == a) & (sel.rule == r)]
            L.append(f"| {a:g} | {r} | {len(v)}/{n_exp} | {int(v.exceed.sum())} | {int(v.sig_exceed.sum())} |")
            if r == "ltt" and len(v):
                e1.append(int(v.sig_exceed.sum()) <= 2)
            if r == "naive_raw" and len(v):
                e2.append(v.sig_exceed.mean() > 0.5)
    L += ["", f"Endpoint 1 (ltt: at most 2 certified experiments significantly exceed alpha, at every alpha with a "
          f"certificate): {'CONFIRMED' if e1 and all(e1) else ('NOT CONFIRMED' if e1 else 'not evaluable (no certificate)')}.",
          f"Endpoint 2 (naive_raw: more than half of the selected experiments significantly exceed alpha, at every alpha "
          f"with a selection): {'CONFIRMED' if e2 and all(e2) else ('NOT CONFIRMED' if e2 else 'not evaluable')}.", "",
          "## Endpoint 3 (C4): patch transfer", "",
          "| distance | transfer | axis | alpha | certified pairs | failing (BH 5%) | share % [90% CI] |", "|---|---|---|---|---|---|---|"]
    for _, r in sh.sort_values(["distance", "transfer", "alpha", "axis"]).iterrows():
        L.append(f"| {r.distance} | {r.transfer} | {r.axis} | {r.alpha:g} | {r.pairs} | {r.failing} | "
                 f"{100 * r.share:.1f} [{100 * r.lo:.1f}, {100 * r.hi:.1f}] |")
    L.append("")
    for dist in (3, 5):
        v = sh[(sh.distance == dist) & (sh.transfer == "threshold") & (sh.alpha == 0.1)].set_index("axis")
        if "patch" in v.index and "same" in v.index:
            ok = v.loc["patch", "lo"] > v.loc["same", "share"]
            L.append(f"Endpoint 3, d = {dist}, threshold transfer, alpha = 0.1 (patch interval's lower end above the "
                     f"same-experiment share): {'CONFIRMED' if ok else 'NOT CONFIRMED'}.")
        else:
            L.append(f"Endpoint 3, d = {dist}: not evaluable (no certified patch or same pairs at alpha = 0.1).")
    return "\n".join(L) + "\n", sh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run-rounds", type=int)
    args = ap.parse_args()
    if args.run == (args.dry_run_rounds is not None):
        raise SystemExit("give exactly one of --run or --dry-run-rounds")
    if args.run:
        tag = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--short", "v0.2.0^{commit}"], cwd=REPO,
                             capture_output=True, text=True).stdout.strip()
        if not tag:
            raise SystemExit("tag v0.2.0 not found: commit and tag the preregistration first")
        out = RES
        if (out / "p6_confirmatory_r13.csv").exists():
            raise SystemExit("results/p6_confirmatory_r13.csv exists: the confirmatory run is done once")
        rounds = 13
    else:
        if args.dry_run_rounds == 13:
            raise SystemExit("the dry run must not touch r = 13")
        tag, rounds = "dry run", args.dry_run_rounds
        out = RES / "cache" / "p6_dryrun"
        out.mkdir(parents=True, exist_ok=True)
    exps = list_experiments(rounds=(rounds,))
    D = {e.key: data(e) for e in exps}
    c2 = pd.DataFrame([r for e in exps for r in c2_rows(e, D[e.key])])
    pairs = c4_rows(exps, D)
    text, sh = summary(c2, pairs, tag, len(exps))
    c2.to_csv(out / "p6_confirmatory_r13.csv", index=False)
    pairs.to_csv(out / "p6_confirmatory_r13_pairs.csv", index=False)
    (out / "p6_confirmatory_r13.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
