"""A4: validity over repeated calibration draws (arXiv v1 plan, task A4).

Every (experiment, score, prior) cell that M4 covers (r != 13), with M4's thresholds (from Train), calibrators
(fitted on Train), alphas and delta. The population of a cell is its 30,000 non-Train shots (Calibrate + Test).

Part 1, bootstrap world. 2,000 calibration sets of 15,000 shots are drawn i.i.d. from the population; each rule
selects a threshold; a violation is a selected threshold whose population risk exceeds alpha. Here the guarantee
holds by theorem (ties, the start rule and the grid included), so a Learn-then-Test violation rate above
0.05 + 3 sqrt(0.05 x 0.95 / 2000) = 0.065 is flagged. A flagged cell is rerun once with a fresh seed and
20,000 draws and counts as a bug if the rerun exceeds 0.05 + 3 sqrt(0.05 x 0.95 / 20000) = 0.0546 (plan v1.1).
Rules: CERT_RULES (Learn-then-Test and, from A5, the other certified rules) and the plug-in rule for contrast.

Part 2, repeated real splits. 200 random re-partitions of the population into 15,000 Calibrate and 15,000 Test
(Train fixed). For each, the M4 headline numbers per (score, prior, alpha, rule): experiments with a selection,
share exceeding alpha on Test, share significantly exceeding (one-sided 95% Clopper-Pearson), mean kept fraction.
The committed split is computed by the same code and checked row by row against results/m4_guarantees.csv.
Acceptance: the committed split's exceed and significant shares lie inside the 5th-95th percentile range of the
resplits in at least 90% of table cells; otherwise the paper reports resplit medians.

Seeds: [20261117, 3, part, experiment index, combo index] (rerun: [20261117, 3, 9, exp, combo, alpha index]);
the A5 baselines in Part 1 draw from their own stream [20261117, 3, 4, exp, combo], so the ltt and plugin
rows are unchanged by adding them.

    python scripts/p3_validity.py [--workers 4]      # resumable: per-experiment results in results/cache/p3/
"""
import argparse
import concurrent.futures as cf
import json
import pathlib

import numpy as np
from scipy.stats import beta

from qeccal.calibration import CALIBRATORS, score_from_q
from qeccal.data import list_experiments, split_indices
from qeccal.guarantees import KEEP_GRID, thresholds
from qeccal.guarantees.validity import RULES, band_counts, population_risk, violation_rate
import qeccal.guarantees.baselines  # noqa: F401  (registers bonferroni, sgr, ltt_hb in RULES)
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
CACHE = RES / "cache" / "p3"
SEED = 20261117
ALPHAS = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)          # as M4
DELTA = 0.05
COMBOS = [("mwpm_gap", "si1000"), ("mwpm_gap", "rl"), ("exact", "si1000"), ("exact", "rl"), ("bm_gap", "rl"), ("nn", "rl")]
CERT_RULES = ("ltt", "bonferroni", "sgr", "ltt_hb")          # A5 added the three baselines
PART1_RULES = CERT_RULES + ("plugin",)
P2_RULES = ("ltt", "plugin", "naive_raw", "naive_platt")
DRAWS, RERUN_DRAWS, RESPLITS = 2000, 20000, 200
BOUND = 0.05 + 3 * np.sqrt(0.05 * 0.95 / DRAWS)
RERUN_BOUND = 0.05 + 3 * np.sqrt(0.05 * 0.95 / RERUN_DRAWS)
M = len(KEEP_GRID)


def cp_lower(e, n):
    e, n = np.asarray(e, float), np.asarray(n, float)
    out = np.zeros(np.broadcast(e, n).shape)
    ok = (e > 0) & (n > 0)
    out[ok] = beta.ppf(0.05, e[ok], n[ok] - e[ok] + 1)
    return out


def loosest(ok):
    last = ok.shape[1] - 1 - ok[:, ::-1].argmax(axis=1)
    return np.where(ok.any(axis=1), last, -1)


def part2(band, wrong, q_raw, q_platt, cal_sets, n_cal):
    """Headline outcomes for each calibration set (rows of cal_sets: positions into the population).
    Returns (n_sets, len(ALPHAS), len(P2_RULES), 4): selected, exceed, sig_exceed, kept fraction of Test."""
    k, B = cal_sets.shape[0], M + 1
    flat = (np.arange(k)[:, None] * B + band[cal_sets]).ravel()
    cnt = lambda w=None: np.bincount(flat, weights=None if w is None else w[cal_sets].ravel(), minlength=k * B).reshape(k, B)  # noqa: E731
    cn, ce, cqr, cqp = cnt(), cnt(wrong.astype(float)), cnt(q_raw), cnt(q_platt)
    tot_n = np.bincount(band, minlength=B)
    tot_e = np.bincount(band, weights=wrong.astype(float), minlength=B)
    tn, te = tot_n[None] - cn, tot_e[None] - ce
    cum = lambda a: np.cumsum(a, axis=1)[:, :M]  # noqa: E731
    cn, ce, cqr, cqp, tn, te = map(cum, (cn, ce, cqr, cqp, tn, te))
    ce, te = ce.round(), te.round()
    n_test = band.size - n_cal
    out = np.zeros((k, len(ALPHAS), len(P2_RULES), 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a_i, a in enumerate(ALPHAS):
            sel = {"ltt": RULES["ltt"](cn, ce, n_cal, a, DELTA),
                   "plugin": RULES["plugin"](cn, ce, n_cal, a, DELTA),
                   "naive_raw": loosest((cn > 0) & (cqr / np.maximum(cn, 1) <= a)),
                   "naive_platt": loosest((cn > 0) & (cqp / np.maximum(cn, 1) <= a))}
            for r_i, r in enumerate(P2_RULES):
                i = sel[r]
                got = i >= 0
                j = np.clip(i, 0, M - 1)
                n_t, e_t = tn[np.arange(k), j], te[np.arange(k), j]
                out[:, a_i, r_i, 0] = got
                out[:, a_i, r_i, 1] = got & (n_t > 0) & (e_t / np.maximum(n_t, 1) > a)
                out[:, a_i, r_i, 2] = got & (cp_lower(e_t, n_t) > a)
                out[:, a_i, r_i, 3] = np.where(got, n_t / n_test, np.nan)
    return out


def run(job):
    xi, e = job
    out = CACHE / (e.key.replace("/", "__") + ".npz")
    if out.exists():
        return
    idx = split_indices(e)
    obs = e.observable_flips()
    p1_rows, arrays = [], {}
    for ci, (method, prior) in enumerate(COMBOS):
        if not cache_path(method, prior, e.key).exists():
            continue
        o = load(method, prior, e.key)
        s = o["gap"].astype(np.float64) if "gap" in o else score_from_q(o["q"])
        wrong = o["pred"] != obs
        tr, ca, te = idx["train"], idx["calibrate"], idx["test"]
        if "fit_mask" in o:
            tr = tr[~o["fit_mask"][tr]]
            assert not o["fit_mask"][ca].any() and not o["fit_mask"][te].any()
        lams = thresholds(s[tr])
        pop = np.concatenate([ca, te])
        s_p, w_p = s[pop], wrong[pop]
        # Part 1
        nb, eb = band_counts(s_p, w_p, lams)
        rng = np.random.default_rng([SEED, 3, 1, xi, ci])            # ltt and plugin (as in the first A4 run)
        rng_b = np.random.default_rng([SEED, 3, 4, xi, ci])          # A5 baselines: own stream
        risk = population_risk(nb, eb)
        for a_i, a in enumerate(ALPHAS):
            for rule in PART1_RULES:
                r = violation_rate(nb, eb, a, DELTA, ca.size, DRAWS, rng if rule in ("ltt", "plugin") else rng_b, rule=rule)
                row = {"key": e.key, "distance": e.distance, "rounds": e.rounds, "method": method, "prior": prior,
                       "alpha": a, "rule": rule, **r, "rate": r["violations"] / r["draws"],
                       "min_risk": float(np.nanmin(risk)), "flagged": 0, "rerun_draws": 0, "rerun_violations": 0}
                if rule in CERT_RULES and row["rate"] > BOUND:
                    rr = violation_rate(nb, eb, a, DELTA, ca.size, RERUN_DRAWS,
                                        np.random.default_rng([SEED, 3, 9, xi, ci, a_i, CERT_RULES.index(rule)]), rule=rule)
                    row.update(flagged=1, rerun_draws=rr["draws"], rerun_violations=rr["violations"])
                p1_rows.append(row)
        # Part 2
        fitted = {name: CALIBRATORS[name]().fit(s[tr], wrong[tr]) for name in ("raw", "platt")}
        band = np.searchsorted(-lams, -s_p, side="left")
        q_raw, q_platt = fitted["raw"].predict(s_p), fitted["platt"].predict(s_p)
        rng2 = np.random.default_rng([SEED, 3, 2, xi, ci])
        perms = rng2.permuted(np.broadcast_to(np.arange(pop.size), (RESPLITS, pop.size)), axis=1)[:, : ca.size]
        committed = np.arange(ca.size)[None]                 # pop = [Calibrate, Test]
        arrays[f"{method}__{prior}"] = part2(band, w_p, q_raw, q_platt, np.concatenate([committed, perms]), ca.size)
    np.savez(out.with_suffix(".tmp.npz"), p1=json.dumps(p1_rows), **arrays)
    out.with_suffix(".tmp.npz").rename(out)


def summarize(keys):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p1, p2 = [], {}
    for k in keys:
        z = np.load(CACHE / (k.replace("/", "__") + ".npz"))
        p1 += json.loads(str(z["p1"]))
        for c in z.files:
            if c != "p1":
                p2[(k, c)] = z[c]
    p1 = pd.DataFrame(p1)
    p1.to_csv(RES / "p3_validity_bootstrap.csv", index=False)

    # committed split vs M4, row by row
    m4 = pd.read_csv(RES / "m4_guarantees.csv")
    m4 = m4[m4.rule.isin(P2_RULES)]
    mism = 0
    for (k, c), arr in p2.items():
        method, prior = c.split("__")
        sub = m4[(m4.key == k) & (m4.method == method) & (m4.prior == prior)]
        for a_i, a in enumerate(ALPHAS):
            for r_i, r in enumerate(P2_RULES):
                row = sub[(sub.alpha == a) & (sub.rule == r)].iloc[0]
                sel = not pd.isna(row["keep"])
                mine = arr[0, a_i, r_i]
                if bool(mine[0]) != sel or (sel and (int(mine[1]) != int(row["exceed"]) or int(mine[2]) != int(row["sig_exceed"]))):
                    mism += 1
    # aggregate per split: rows = (split, method, prior, alpha, rule)
    agg = []
    combos = sorted({c for _, c in p2})
    for c in combos:
        stack = np.stack([v for (k, cc), v in p2.items() if cc == c])           # (exp, sets, alpha, rule, 4)
        sel = stack[..., 0]
        nsel = sel.sum(axis=0)
        with np.errstate(invalid="ignore"):
            ex = np.where(nsel > 0, stack[..., 1].sum(axis=0) / np.maximum(nsel, 1), np.nan)
            sg = np.where(nsel > 0, stack[..., 2].sum(axis=0) / np.maximum(nsel, 1), np.nan)
            kept = np.nanmean(np.where(sel > 0, stack[..., 3], np.nan), axis=0)
        method, prior = c.split("__")
        for s_i in range(stack.shape[1]):
            for a_i, a in enumerate(ALPHAS):
                for r_i, r in enumerate(P2_RULES):
                    agg.append({"split": "committed" if s_i == 0 else s_i, "method": method, "prior": prior, "alpha": a,
                                "rule": r, "experiments": stack.shape[0], "selected": int(nsel[s_i, a_i, r_i]),
                                "exceed_share": ex[s_i, a_i, r_i], "sig_share": sg[s_i, a_i, r_i],
                                "mean_kept": kept[s_i, a_i, r_i]})
    agg = pd.DataFrame(agg)
    agg.to_csv(RES / "p3_validity_resplit.csv", index=False)

    # acceptance, Part 2
    rs = agg[agg.split != "committed"]
    cm = agg[agg.split == "committed"].set_index(["method", "prior", "alpha", "rule"])
    q = rs.groupby(["method", "prior", "alpha", "rule"])[["exceed_share", "sig_share", "mean_kept", "selected"]].quantile([0.05, 0.5, 0.95]).unstack()
    cells, inside, tab = 0, 0, []
    for ix in q.index:
        for met in ("exceed_share", "sig_share"):
            lo, med, hi = q.loc[ix, (met, 0.05)], q.loc[ix, (met, 0.5)], q.loc[ix, (met, 0.95)]
            v = cm.loc[ix, met]
            if pd.isna(v) or pd.isna(lo):
                continue
            cells += 1
            inside += int(lo - 1e-12 <= v <= hi + 1e-12)
        tab.append((ix, cm.loc[ix], q.loc[ix]))
    share_inside = inside / cells

    # acceptance, Part 1
    lt = p1[p1.rule.isin(CERT_RULES)]
    flagged = lt[lt.flagged == 1]
    bugs = flagged[flagged.rerun_violations / flagged.rerun_draws.clip(lower=1) > RERUN_BOUND]
    pl = p1[p1.rule == "plugin"]
    L = ["# A4: validity over repeated calibration draws", "",
         "Generated by `scripts/p3_validity.py` (method, seeds and acceptance rules in its docstring). "
         f"{p1[['key', 'method', 'prior']].drop_duplicates().shape[0]} (experiment, score, prior) cells x "
         f"{len(ALPHAS)} alphas; population = the 30,000 non-Train shots of each cell; delta = {DELTA}.", "",
         "## Part 1: bootstrap world (2,000 calibration draws per cell)", "",
         f"Bound for a certified rule: violation rate <= {BOUND:.4f}; flagged cells rerun with 20,000 draws, bug if "
         f"> {RERUN_BOUND:.4f}.", "",
         "| rule | cells | cells with a certificate in some draw | max violation rate | 99th percentile | cells above 0.05 | flagged (> bound) | bugs after rerun |",
         "|---|---|---|---|---|---|---|---|"]
    for rule in PART1_RULES:
        v = p1[p1.rule == rule]
        fl = v[v.rate > BOUND]
        bg = fl[fl.rerun_violations / fl.rerun_draws.clip(lower=1) > RERUN_BOUND] if rule in CERT_RULES else fl.iloc[:0]
        L.append(f"| {rule} | {len(v)} | {(v.certified > 0).sum()} | {v.rate.max():.4f} | {v.rate.quantile(0.99):.4f} | "
                 f"{(v.rate > 0.05).sum()} | {len(fl)} | {len(bg) if rule in CERT_RULES else '(not certified)'} |")
    L += ["", f"**Part 1 acceptance: {'PASS' if bugs.empty else 'FAIL'}** "
          f"({len(flagged)} certified-rule cells flagged, {len(bugs)} still above {RERUN_BOUND:.4f} after the rerun).", ""]
    if not flagged.empty:
        L += ["Flagged cells:", "", "| key | score | alpha | rule | rate (2,000) | rerun rate (20,000) |", "|---|---|---|---|---|---|"]
        for _, r in flagged.iterrows():
            L.append(f"| {r.key} | {r.method}/{r.prior} | {r.alpha:g} | {r.rule} | {r.rate:.4f} | {r.rerun_violations / r.rerun_draws:.4f} |")
        L.append("")
    L += ["## Part 2: 200 random re-partitions of Calibrate and Test", "",
          f"Committed split recomputed by the same code: {mism} disagreements with `results/m4_guarantees.csv` "
          f"(selection, exceed, significant; {len(p2) * len(ALPHAS) * len(P2_RULES)} rows).", "",
          f"**Part 2 acceptance: {'PASS' if share_inside >= 0.9 else 'FAIL'}**: committed value inside the resplit "
          f"5th-95th percentile range in {inside} of {cells} cells ({100 * share_inside:.1f}%; need 90%).", "",
          "Exceeds and significant: share of experiments with a selection. Columns give committed / resplit median "
          "[5th, 95th percentile].", "",
          "| score | alpha | rule | selected | exceeds | significant | mean kept |", "|---|---|---|---|---|---|---|"]
    for ix, c, qq in tab:
        f = lambda met, pct=True: (f"{100 * c[met]:.1f} / {100 * qq[(met, 0.5)]:.1f} [{100 * qq[(met, 0.05)]:.1f}, {100 * qq[(met, 0.95)]:.1f}]"  # noqa: E731
                                   if pct else f"{c[met]:.2f} / {qq[(met, 0.5)]:.2f}") if not pd.isna(c[met]) else "-"
        L.append(f"| {ix[0]}/{ix[1]} | {ix[2]:g} | {ix[3]} | {int(c['selected'])} / {qq[('selected', 0.5)]:.0f} | "
                 f"{f('exceed_share')} | {f('sig_share')} | {f('mean_kept', False)} |")
    L.append("")
    (RES / "p3_validity.md").write_text("\n".join(L))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(0, max(0.3, p1.rate.max()), 61)
    for rule in ("ltt", "plugin"):
        axes[0].hist(p1[(p1.rule == rule) & (p1.certified > 0)].rate, bins=bins, alpha=0.6, label=rule)
    axes[0].axvline(DELTA, color="k", lw=0.8)
    axes[0].axvline(BOUND, color="k", lw=0.8, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("violation rate over 2,000 calibration draws")
    axes[0].set_ylabel("cells")
    axes[0].set_title("Part 1: bootstrap world (cells with any certificate)")
    axes[0].legend()
    sub = rs[(rs.method == "mwpm_gap") & (rs.prior == "rl")]
    pos = 0
    ticks, labels = [], []
    for a in ALPHAS:
        for r_i, r in enumerate(("ltt", "plugin", "naive_platt")):
            v = sub[(sub.alpha == a) & (sub.rule == r)].sig_share.dropna() * 100
            if not v.empty:
                axes[1].boxplot(v, positions=[pos], widths=0.6, showfliers=False, patch_artist=True,
                                boxprops={"facecolor": f"C{r_i}", "alpha": 0.5})
                axes[1].plot(pos, 100 * cm.loc[("mwpm_gap", "rl", a, r), "sig_share"], "k*")
            ticks.append(pos)
            labels.append(f"{r}\n{a:g}")
            pos += 1
        pos += 0.5
    axes[1].set_xticks(ticks, labels, fontsize=6, rotation=90)
    axes[1].set_ylabel("significantly exceeding alpha (%)")
    axes[1].set_title("Part 2: 200 resplits (box), committed (*); MWPM gap, RL")
    fig.tight_layout()
    fig.savefig(RES / "p3_validity.png", dpi=200)
    print("\n".join(L[:40]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    exps = list(enumerate(list_experiments()))
    exps = [(i, e) for i, e in exps if e.rounds != 13]
    todo = [(i, e) for i, e in exps if not (CACHE / (e.key.replace("/", "__") + ".npz")).exists()]
    print(f"{len(exps) - len(todo)} of {len(exps)} done; running {len(todo)}", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(run, todo), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    summarize([e.key for _, e in exps])


if __name__ == "__main__":
    main()
