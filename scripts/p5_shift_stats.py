"""A6: shift statistics that respect dependence between pairs (arXiv v1 plan, task A6).

Input: results/m5_drift.csv as committed (certified source-target pairs on the axes same, patch, basis, pooled,
sim; transfers threshold and keep_fraction), and the score caches for the exchangeability test.

1. Every certified pair gets pair_p_value(test_n, test_errors, alpha) = P(Bin(n, alpha) >= e), the ratio of the
   target's Test rate to alpha, and the ratio to the source's own Test rate (from its "same" row; NaN for pooled
   and simulated sources).
2. Per (score, prior, transfer, axis, alpha): a pair fails if Benjamini-Hochberg at 5% rejects it within that
   group. Share of failing pairs with a 90% two-way cluster-bootstrap interval (2,000 resamples of source patches
   and target patches; one-way where source and target clusters coincide: same, basis, sim, pooled use the target
   patch). Reference: the same statistic on the "same" axis. Wording: "fails" where the axis interval's lower end
   is above the reference share; otherwise "not distinguishable".
3. Leave-one-source-cluster-out: the share recomputed (BH redone) without each source patch; the range is
   reported.
4. Exchangeability, independent of any threshold: exchangeability_test (1,000 permutations) between Test shots of
   every unordered patch pair (same distance, basis, round count), every basis pair (same patch and round count),
   every hardware-simulation pair (MWPM gap: Test shots against the 15,000 simulated shots not used for Train),
   and, as a reference that is exchangeable by construction, Calibrate against Test of the same experiment.
   Share rejected under BH at 5% per (score, prior, axis), with the cluster-bootstrap interval.
Seeds: [20261117, 5, k] with k from a SHA-256 of the group or pair label.

Writes results/p5_shift_stats.csv, results/p5_exchangeability.csv, results/p5_shift_stats.md, results/p5_shift.png.
"""
import hashlib
import pathlib
import sys

import numpy as np
import pandas as pd

from qeccal.calibration import score_from_q
from qeccal.data import list_experiments, split_indices
from qeccal.guarantees import KEEP_GRID
from qeccal.shift import bh, cluster_bootstrap_share, exchangeability_test, pair_p_value
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
sys.path.insert(0, str(REPO / "scripts"))
from m5_drift import ROUNDS, load_sim  # noqa: E402

COMBOS = (("mwpm_gap", "si1000"), ("mwpm_gap", "rl"), ("bm_gap", "rl"), ("nn", "rl"))
AXES = ("same", "patch", "basis", "pooled", "sim")
N_BOOT, N_PERM, Q = 2000, 1000, 0.05


def rng_for(*parts):
    h = int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")
    return np.random.default_rng([20261117, 5, h])


def patch_of(key):
    return key.split("/")[0].split("_at_")[1]


def pair_table():
    d = pd.read_csv(RES / "m5_drift.csv")
    d = d[d.certified == 1].copy()
    d["p"] = [pair_p_value(int(n), int(e), a) for n, e, a in zip(d.test_n, d.test_errors, d.alpha)]
    d["ratio_alpha"] = d.test_rate / d.alpha
    own = d[(d.axis == "same") & (d.transfer == "threshold")].set_index(["method", "prior", "alpha", "target"]).test_rate
    src_rate = [own.get((m, p, a, s), np.nan) for m, p, a, s in zip(d.method, d.prior, d.alpha, d.source)]
    d["ratio_src"] = d.test_rate / np.array(src_rate, float)
    d["target_patch"] = d.target.map(patch_of)
    d["source_cluster"] = np.where(d.axis == "patch", d.source.map(lambda s: patch_of(s) if "_at_" in s and ":" not in s else ""),
                                   d.target_patch)
    d["fail"] = False
    for _, g in d.groupby(["method", "prior", "transfer", "axis", "alpha"]):
        d.loc[g.index, "fail"] = bh(g.p.values, Q)
    return d


def share_rows(d):
    rows = []
    for (m, p, tr, ax, a), g in d.groupby(["method", "prior", "transfer", "axis", "alpha"]):
        share, lo, hi = cluster_bootstrap_share(g.fail.values, g.source_cluster.values, g.target_patch.values, N_BOOT,
                                                rng_for("share", m, p, tr, ax, a))
        loo = []
        for c in np.unique(g.source_cluster):
            h = g[g.source_cluster != c]
            if len(h):
                loo.append(bh(h.p.values, Q).mean())
        rows.append({"method": m, "prior": p, "transfer": tr, "axis": ax, "alpha": a, "pairs": len(g),
                     "failing": int(g.fail.sum()), "share": share, "lo": lo, "hi": hi,
                     "loo_min": min(loo) if loo else np.nan, "loo_max": max(loo) if loo else np.nan,
                     "median_ratio_alpha": g.ratio_alpha.median(), "median_ratio_src": g.ratio_src.median()})
    s = pd.DataFrame(rows)
    ref = s[s.axis == "same"].set_index(["method", "prior", "transfer", "alpha"])
    s["ref_share"] = [ref.share.get((m, p, t, a), np.nan) for m, p, t, a in zip(s.method, s.prior, s.transfer, s.alpha)]
    s["ref_hi"] = [ref.hi.get((m, p, t, a), np.nan) for m, p, t, a in zip(s.method, s.prior, s.transfer, s.alpha)]
    s["verdict"] = np.where(s.axis == "same", "reference", np.where(s.lo > s.ref_share, "fails", "not distinguishable"))
    return s


def hw_test(e, method, prior):
    idx = split_indices(e)
    o = load(method, prior, e.key)
    s = o["gap"].astype(np.float64) if "gap" in o else score_from_q(o["q"])
    w = o["pred"] != e.observable_flips()
    return {r: (s[idx[r]], w[idx[r]]) for r in ("calibrate", "test")}


def exchangeability_rows():
    rows = []
    exps = list_experiments(rounds=ROUNDS)
    for method, prior in COMBOS:
        cov = [e for e in exps if cache_path(method, prior, e.key).exists()]
        hw = {e.key: hw_test(e, method, prior) for e in cov}
        groups = {}
        for e in cov:
            groups.setdefault((e.distance, e.basis, e.rounds), []).append(e)
        tests = []
        for e in cov:
            tests.append(("same", e.key + " cal", e.key + " test", e.patch, e.patch, hw[e.key]["calibrate"], hw[e.key]["test"]))
            other = e.key.replace(f"/{e.basis}/", "/Z/") if e.basis == "X" else None
            if other and other in hw:
                tests.append(("basis", e.key, other, e.patch, e.patch, hw[e.key]["test"], hw[other]["test"]))
            if method == "mwpm_gap":
                tests.append(("sim", "sim:" + e.key, e.key, e.patch, e.patch, load_sim(prior, e.key)["calibrate"], hw[e.key]["test"]))
            for f in groups[(e.distance, e.basis, e.rounds)]:
                if f.key > e.key:
                    tests.append(("patch", e.key, f.key, e.patch, f.patch, hw[e.key]["test"], hw[f.key]["test"]))
        for ax, a, b, ca, cb, (sa, wa), (sb, wb) in tests:
            p = exchangeability_test(sa, wa, sb, wb, KEEP_GRID, N_PERM, rng_for("exch", method, prior, ax, a, b))
            rows.append({"method": method, "prior": prior, "axis": ax, "a": a, "b": b, "cluster_a": ca, "cluster_b": cb,
                         "rate_a": float(np.mean(wa)), "rate_b": float(np.mean(wb)), "p": p})
        print(f"{method}/{prior}: {len(tests)} exchangeability tests", flush=True)
    x = pd.DataFrame(rows)
    x["reject"] = False
    for _, g in x.groupby(["method", "prior", "axis"]):
        x.loc[g.index, "reject"] = bh(g.p.values, Q)
    return x


def md_table(df, cols, fmts):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f(r) for f in fmts) + " |")
    return out


def main():
    d = pair_table()
    d.drop(columns=["fail"]).assign(bh_fail=d.fail.astype(int)).to_csv(RES / "p5_shift_stats.csv", index=False)
    s = share_rows(d)
    x = exchangeability_rows()
    x.to_csv(RES / "p5_exchangeability.csv", index=False)
    ex = []
    for (m, p, ax), g in x.groupby(["method", "prior", "axis"]):
        share, lo, hi = cluster_bootstrap_share(g.reject.values, g.cluster_a.values, g.cluster_b.values, N_BOOT,
                                                rng_for("exshare", m, p, ax))
        ex.append({"method": m, "prior": p, "axis": ax, "pairs": len(g), "raw_p05": float((g.p < 0.05).mean()),
                   "share": share, "lo": lo, "hi": hi})
    ex = pd.DataFrame(ex)
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    L = ["# A6: shift statistics that respect dependence", "",
         "Generated by `scripts/p5_shift_stats.py` from `results/m5_drift.csv` (method in the script docstring). "
         "A certified pair fails if Benjamini-Hochberg at 5% rejects \"target risk at the transferred threshold <= alpha\" "
         "within its (score, prior, transfer, axis, alpha) group. Share = failing / certified pairs, with a 90% "
         "cluster-bootstrap interval over source and target patches; LOO = range of the share when one source patch is "
         "left out; reference = the same-experiment share. Verdict \"fails\" where the interval's lower end is above "
         "the reference share.", ""]
    for tr in ("threshold", "keep_fraction"):
        for (m, p), g in s[s.transfer == tr].groupby(["method", "prior"], sort=False):
            L += [f"## {m} / {p}, transfer = {tr}", ""]
            g = g.sort_values(["axis", "alpha"], key=lambda c: c.map({a: i for i, a in enumerate(AXES)}) if c.name == "axis" else c)
            L += md_table(g, ["axis", "alpha", "pairs", "failing", "share % [90% CI]", "LOO range %", "ref %", "median rate/alpha", "verdict"],
                          [lambda r: r.axis, lambda r: f"{r.alpha:g}", lambda r: str(r.pairs), lambda r: str(r.failing),
                           lambda r: f"{pct(r.share)} [{pct(r.lo)}, {pct(r.hi)}]",
                           lambda r: f"{pct(r.loo_min)}-{pct(r.loo_max)}" if not np.isnan(r.loo_min) else "-",
                           lambda r: pct(r.ref_share) if not np.isnan(r.ref_share) else "-",
                           lambda r: f"{r.median_ratio_alpha:.2f}", lambda r: r.verdict])
            L.append("")
    L += ["## Exchangeability of the risk-versus-keep curves (no threshold involved)", "",
          "Test shots of the two members of each pair (same: Calibrate vs Test of one experiment, exchangeable by "
          "construction). 1,000 permutations; share rejected under BH at 5% within (score, prior, axis), with the "
          "90% cluster-bootstrap interval; raw = share with p < 0.05.", ""]
    L += md_table(ex.sort_values(["method", "prior", "axis"]), ["score", "axis", "pairs", "raw p<0.05 %", "BH-rejected % [90% CI]"],
                  [lambda r: f"{r.method}/{r.prior}", lambda r: r.axis, lambda r: str(r.pairs), lambda r: pct(r.raw_p05),
                   lambda r: f"{pct(r.share)} [{pct(r.lo)}, {pct(r.hi)}]"])
    L.append("")
    (RES / "p5_shift_stats.md").write_text("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sub = s[(s.method == "mwpm_gap") & (s.prior == "rl") & (s.transfer == "threshold")]
    alphas = sorted(sub.alpha.unique())
    for k, ax_name in enumerate(AXES):
        v = sub[sub.axis == ax_name].set_index("alpha").reindex(alphas)
        xs = np.arange(len(alphas)) + (k - 2) * 0.15
        axes[0].errorbar(xs, 100 * v.share, yerr=[100 * (v.share - v.lo), 100 * (v.hi - v.share)], fmt="o", capsize=2, label=ax_name)
    axes[0].set_xticks(np.arange(len(alphas)), [f"{a:g}" for a in alphas])
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("certified pairs failing (BH 5%), %")
    axes[0].set_title("Transferred thresholds, MWPM gap, RL prior (90% cluster CI)")
    axes[0].legend(fontsize=8)
    order = ["same", "basis", "patch", "sim"]
    for k, (m, p) in enumerate(COMBOS):
        v = ex[(ex.method == m) & (ex.prior == p)].set_index("axis").reindex(order)
        xs = np.arange(len(order)) + (k - 1.5) * 0.18
        axes[1].errorbar(xs, 100 * v.share, yerr=[100 * (v.share - v.lo), 100 * (v.hi - v.share)], fmt="s", capsize=2, label=f"{m}/{p}")
    axes[1].set_xticks(np.arange(len(order)), order)
    axes[1].set_ylabel("pairs rejected (BH 5%), %")
    axes[1].set_title("Exchangeability of risk-vs-keep curves")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "p5_shift.png", dpi=200)
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
