"""A3: blocked split, calibrate on earlier shots and deploy on later ones (arXiv v1 plan, task A3).

Same rules, keep grid, alphas and delta as M4 (scripts/m4_calibrate_certify.py), on the scores without a
learned component: mwpm_gap under both priors (392 experiments, r != 13) and bm_gap/rl (112). The learned
decoder is excluded: it was fine-tuned on randomly chosen shots that include late ones.

  --split random   the frozen split of splits.py; must reproduce the matching rows of results/m4_guarantees.csv
  --split blocked  first 40% of shots in file order = Train, next 30% = Calibrate, last 30% = Test
  --split both     run both and write the summary (default)

Each selected threshold is also evaluated on three consecutive windows of shot indices, 35,000-40,000,
40,000-45,000 and 45,000-50,000 (for the random split: the Test shots that fall in those windows), to see whether
exceedance grows with distance from calibration. A2 found that file order carries drift (case 2), so the blocked
split is read as a temporal test.

Writes results/p2_blocked.csv, results/p2_blocked.md, results/p2_blocked.png.
"""
import argparse
import csv
import pathlib

import numpy as np

from qeccal.calibration import CALIBRATORS
from qeccal.data import list_experiments, split_indices
from qeccal.data.splits_extra import blocked_indices
from qeccal.guarantees import KEEP_GRID, cp_lower, learn_then_test, loosest_where, thresholds
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)          # as M4
DELTA = 0.05
COMBOS = [("mwpm_gap", "si1000"), ("mwpm_gap", "rl"), ("bm_gap", "rl")]
RULES = ("ltt", "plugin", "naive_raw", "naive_platt")
WINDOWS = ((35000, 40000), (40000, 45000), (45000, 50000))
M4_FIELDS = ["key", "distance", "patch", "basis", "rounds", "method", "prior", "alpha", "rule", "keep",
             "cal_n", "cal_errors", "test_n", "test_errors", "test_rate", "exceed", "sig_exceed"]
FIELDS = ["split"] + M4_FIELDS + [f"w{j}_{c}" for j in (1, 2, 3) for c in ("n", "errors")]


def run(e, split):
    idx = split_indices(e) if split == "random" else blocked_indices(e.shots)
    obs = e.observable_flips()
    base = {"split": split, "key": e.key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds}
    rows = []
    for method, prior in COMBOS:
        if not cache_path(method, prior, e.key).exists():
            continue
        o = load(method, prior, e.key)
        s = o["gap"].astype(np.float64)
        wrong = o["pred"] != obs
        tr, ca, te = idx["train"], idx["calibrate"], idx["test"]
        fitted = {name: CALIBRATORS[name]().fit(s[tr], wrong[tr]) for name in ("raw", "platt")}
        lams = thresholds(s[tr])
        s_ca, w_ca, s_te, w_te = s[ca], wrong[ca], s[te], wrong[te]
        kept_ca = [s_ca >= lam for lam in lams]
        cal_err = [w_ca[k].mean() if k.any() else np.inf for k in kept_ca]
        q_raw, q_platt = fitted["raw"].predict(s_ca), fitted["platt"].predict(s_ca)
        mean_raw = [q_raw[k].mean() if k.any() else np.inf for k in kept_ca]
        mean_platt = [q_platt[k].mean() if k.any() else np.inf for k in kept_ca]
        win = [(te >= a) & (te < b) for a, b in WINDOWS]
        for alpha in ALPHAS:
            choice = {
                "ltt": learn_then_test(s_ca, w_ca, lams, alpha, DELTA)[0],
                "plugin": loosest_where(cal_err, lambda v: v <= alpha),
                "naive_raw": loosest_where(mean_raw, lambda v: v <= alpha),
                "naive_platt": loosest_where(mean_platt, lambda v: v <= alpha),
            }
            for rule in RULES:
                i = choice[rule]
                row = {**base, "method": method, "prior": prior, "alpha": alpha, "rule": rule}
                if i is None:
                    rows.append({**row, **{f: "" for f in FIELDS if f not in row}})
                    continue
                kt = s_te >= lams[i]
                n, er = int(kt.sum()), int(w_te[kt].sum())
                row.update({"keep": KEEP_GRID[i], "cal_n": int(kept_ca[i].sum()), "cal_errors": int(w_ca[kept_ca[i]].sum()),
                            "test_n": n, "test_errors": er, "test_rate": er / n if n else "",
                            "exceed": int(n > 0 and er / n > alpha), "sig_exceed": int(cp_lower(er, n) > alpha)})
                for j, w in enumerate(win, 1):
                    row[f"w{j}_n"], row[f"w{j}_errors"] = int((kt & w).sum()), int((kt & w & w_te).sum())
                rows.append(row)
    return rows


def regression(rows):
    """--split random against results/m4_guarantees.csv on the same combos: every field must match as written."""
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=M4_FIELDS, extrasaction="ignore")
    w.writerows(r for r in rows if r["split"] == "random")
    mine = buf.getvalue().splitlines()
    combos = {f"{m},{p}" for m, p in COMBOS}
    with (RES / "m4_guarantees.csv").open() as f:
        next(f)
        ref = [ln.rstrip("\r\n") for ln in f if ",".join(ln.split(",")[5:7]) in combos]
    ok = mine == ref
    print(f"regression vs m4_guarantees.csv: {len(mine)} rows here, {len(ref)} in M4 for these combos -> "
          f"{'IDENTICAL' if ok else 'DIFFERENT'}")
    return ok


def summarize(rows):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import beta

    df = pd.DataFrame(rows)
    sel = df[df["keep"] != ""].copy()
    for c in ["test_n", "test_errors", "exceed", "sig_exceed", "alpha"] + [f"w{j}_{c}" for j in (1, 2, 3) for c in ("n", "errors")]:
        sel[c] = pd.to_numeric(sel[c])
    tot = df.groupby(["method", "prior", "alpha", "rule", "split"]).size()
    g = sel.groupby(["method", "prior", "alpha", "rule", "split"])
    t = pd.DataFrame({"selected": g.size(), "exceeds": g.exceed.mean(), "significant": g.sig_exceed.mean(),
                      "kept": g.test_n.mean() / 15000}).reindex(tot.index)
    t["of"] = tot
    L = ["# A3: blocked split (calibrate on earlier shots, deploy on later ones)", "",
         "Generated by `scripts/p2_blocked_split.py`. Random = the frozen M4 split; blocked = shots 0-19,999 Train, "
         "20,000-34,999 Calibrate, 35,000-49,999 Test, in file order. Same rules, grid, alphas and delta = 0.05 as M4. "
         "Exceeds = Test error among kept shots above alpha; significant = one-sided 95% Clopper-Pearson lower bound "
         "above alpha. Shares are over experiments where the rule selected a threshold.", ""]
    for (m, p) in COMBOS:
        L += [f"## {m} / {p}", "", "| alpha | rule | selected (random / blocked) | exceeds | significant | mean kept |",
              "|---|---|---|---|---|---|"]
        for a in ALPHAS:
            for r in RULES:
                x = {s: t.loc[(m, p, a, r, s)] for s in ("random", "blocked")}
                f = lambda k, pct=True: " / ".join(  # noqa: E731
                    "-" if pd.isna(x[s][k]) else (f"{100 * x[s][k]:.1f}%" if pct else f"{x[s][k]:.2f}") for s in ("random", "blocked"))
                sel_s = " / ".join(f"{0 if pd.isna(x[s]['selected']) else int(x[s]['selected'])}/{int(x[s]['of'])}" for s in ("random", "blocked"))
                L.append(f"| {a:g} | {r} | {sel_s} | {f('exceeds')} | {f('significant')} | {f('kept', False)} |")
        L.append("")
    # windows, ltt only
    L += ["## Learn-then-Test by Test window", "",
          "Certified thresholds only, pooled over experiments: errors / kept shots in each window, the share of "
          "experiments whose window rate exceeds alpha, and the share whose window rate significantly exceeds alpha "
          "(one-sided 95% Clopper-Pearson). Windows of 5,000 shot indices; for the random split only the Test shots "
          "inside each window count (about 1,500).", "",
          "| score | alpha | split | experiments | 35-40k: pooled rate / alpha, exceeds, significant | 40-45k | 45-50k |",
          "|---|---|---|---|---|---|---|"]
    lt = sel[sel.rule == "ltt"]
    cpl = lambda e, n: 0.0 if e == 0 or n == 0 else float(beta.ppf(0.05, e, n - e + 1))  # noqa: E731
    wtab = []
    for (m, p) in COMBOS:
        for a in ALPHAS:
            for s in ("random", "blocked"):
                v = lt[(lt.method == m) & (lt.prior == p) & (lt.alpha == a) & (lt.split == s)]
                if v.empty:
                    continue
                cells = []
                for j in (1, 2, 3):
                    n, e = v[f"w{j}_n"].values, v[f"w{j}_errors"].values
                    ok = n > 0
                    rate = e[ok] / n[ok]
                    sig = [cpl(ei, ni) > a for ei, ni in zip(e[ok], n[ok])]
                    pooled = e.sum() / max(n.sum(), 1) / a
                    cells.append(f"{pooled:.2f}, {100 * np.mean(rate > a):.0f}%, {100 * np.mean(sig):.0f}%")
                    wtab.append({"method": m, "prior": p, "alpha": a, "split": s, "window": j, "pooled_ratio": pooled,
                                 "exceeds": np.mean(rate > a), "significant": np.mean(sig)})
                L.append(f"| {m}/{p} | {a:g} | {s} | {len(v)} | " + " | ".join(cells) + " |")
    L.append("")
    (RES / "p2_blocked.md").write_text("\n".join(L))

    w = pd.DataFrame(wtab)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sub = t.reset_index()
    sub = sub[(sub.method == "mwpm_gap") & (sub.prior == "rl")]
    xs = np.arange(len(ALPHAS))
    for k, r in enumerate(RULES):
        for s, hatch in (("random", ""), ("blocked", "//")):
            y = [sub[(sub.alpha == a) & (sub.rule == r) & (sub.split == s)].significant.values[0] for a in ALPHAS]
            axes[0].bar(xs + (2 * k + (s == "blocked")) * 0.1 - 0.35, np.nan_to_num(np.array(y, float)) * 100, 0.1,
                        color=f"C{k}", hatch=hatch, label=f"{r} ({s})")
    axes[0].set_xticks(xs, [f"{a:g}" for a in ALPHAS])
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("significantly exceeding alpha on Test (%)")
    axes[0].set_title("MWPM gap, RL prior")
    axes[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    for s, ls in (("random", "-"), ("blocked", "--")):
        for k, a in enumerate(ALPHAS):
            v = w[(w.method == "mwpm_gap") & (w.prior == "rl") & (w.alpha == a) & (w.split == s)]
            if not v.empty:
                axes[1].plot(v.window, v.pooled_ratio, ls, marker="o", color=f"C{k}", label=f"{a:g} {s}")
    axes[1].axhline(1, color="k", lw=0.8)
    axes[1].set_xticks([1, 2, 3], ["35-40k", "40-45k", "45-50k"])
    axes[1].set_ylabel("pooled Test error among kept / alpha")
    axes[1].set_title("Learn-then-Test, by window (MWPM gap, RL prior)")
    axes[1].legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(RES / "p2_blocked.png", dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("random", "blocked", "both"), default="both")
    args = ap.parse_args()
    splits = ("random", "blocked") if args.split == "both" else (args.split,)
    exps = [e for e in list_experiments() if e.rounds != 13]
    rows = [r for s in splits for e in exps for r in run(e, s)]
    if "random" in splits and not regression(rows):
        raise SystemExit("random split does not reproduce M4: stop")
    if args.split == "both":
        with (RES / "p2_blocked.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        summarize(rows)
        print((RES / "p2_blocked.md").read_text())


if __name__ == "__main__":
    main()
