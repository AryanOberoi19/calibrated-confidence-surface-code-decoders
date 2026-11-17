"""A2: does file order within an experiment carry structure? (arXiv v1 plan, task A2)

For each of the 420 experiments, in file order:
  x1 = detection events per shot (row sum of the detection events)
  x2 = decoder-error indicator, MWPM prediction (mwpm_gap/rl cache) != observable flip
Tests: block_heterogeneity on x1 and x2 (50 blocks of 1,000 shots, 2,000 permutations); trend on x1 and x2;
autocorrelation to lag 10 on x1 and x2 with a Ljung-Box p-value; runs test on x2. Effect sizes: block variance
ratio (1 = random order), relative change of the mean between the first and last 10,000 shots, Spearman rho.
Seeds: np.random.default_rng([20261117, 1, experiment_index]), index into list_experiments().

Aggregate, per test: experiments with p < 0.05 (of 420), one-sided binomial p against 5%, Benjamini-Hochberg
rejections at 5%. Classification, fixed before the data were read:
  a test "rejects more often than chance" if its binomial p < 0.001; it "rejects widely" if BH rejects in >= 10%
  of experiments (>= 42).
  case 1  no test rejects more often than chance: no detectable order structure.
  case 2  block heterogeneity or trend (x1 or x2) rejects widely: file order carries drift; A3 is a temporal test.
  case 3  otherwise, if any test rejects more often than chance: serial correlation or bursts only; A4 adds a
          block bootstrap.
  The error rate is "reached" if a test on x2 rejects more often than chance; otherwise any structure found is
  reported as structure in the syndrome statistics that does not reach the error rate.

    python scripts/p1_shot_order.py [--workers 4]      # resumable: per-experiment results in results/cache/p1/
Writes results/p1_shot_order.csv, .md, .png once all experiments are done.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib

import numpy as np
from scipy.stats import binom

from qeccal.data import list_experiments
from qeccal.diagnostics import autocorr, block_heterogeneity, block_variance_ratio, ljung_box, runs_test, trend
from qeccal.soft.cache import load

REPO = pathlib.Path(__file__).resolve().parents[1]
RES = REPO / "results"
CACHE = RES / "cache" / "p1"
SEED = 20261117
N_BLOCKS, N_PERM, MAX_LAG, EDGE = 50, 2000, 10, 10000
TESTS = ["bh_x1", "bh_x2", "trend_x1", "trend_x2", "lb_x1", "lb_x2", "runs_x2"]
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def edge_rel(x):
    x = np.asarray(x, np.float64)
    m = x.mean()
    return float((x[-EDGE:].mean() - x[:EDGE].mean()) / m) if m else float("nan")


def run(job):
    i, e = job
    out = CACHE / (e.key.replace("/", "__") + ".json")
    if out.exists():
        return
    rng = np.random.default_rng([SEED, 1, i])
    packed = e.detection_events(packed=True)
    x1 = POP[packed].sum(axis=1, dtype=np.int64)
    del packed
    x2 = load("mwpm_gap", "rl", e.key)["pred"] != e.observable_flips()
    row = {"index": i, "key": e.key, "distance": e.distance, "patch": e.patch, "basis": e.basis, "rounds": e.rounds,
           "shots": int(x1.size), "x1_mean": float(x1.mean()), "x2_rate": float(x2.mean())}
    for name, x in (("x1", x1), ("x2", x2)):
        row[f"bh_{name}"] = block_heterogeneity(x, N_BLOCKS, N_PERM, rng)
        row[f"ratio_{name}"] = block_variance_ratio(x, N_BLOCKS)
        row[f"rho_{name}"], row[f"trend_{name}"] = trend(x)
        row[f"edge_{name}"] = edge_rel(x)
        r = autocorr(x, MAX_LAG)
        row[f"lb_{name}"] = ljung_box(r, x.size)
        row.update({f"ac{k}_{name}": float(v) for k, v in enumerate(r, 1)})
    row["runs_x2"] = runs_test(x2)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.rename(out)


def bh_count(p, q=0.05):
    p = np.sort(np.asarray(p))
    ok = np.flatnonzero(p <= q * np.arange(1, p.size + 1) / p.size)
    return int(ok[-1] + 1) if ok.size else 0


def md_table(df, fmt=".3g"):
    """Markdown table without the optional tabulate dependency."""
    def cell(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return format(float(v), fmt)
        return str(v)
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join([str(i)] + [cell(v) for v in r]) + " |" for i, r in zip(df.index, df.itertuples(index=False))]
    return "\n".join(rows)


def summarize(n_total):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame([json.loads(f.read_text()) for f in CACHE.glob("*.json")]).sort_values("index")
    assert len(df) == n_total, f"{len(df)} of {n_total} experiments done"
    df.to_csv(RES / "p1_shot_order.csv", index=False)
    n = len(df)
    agg = []
    for t in TESTS:
        k = int((df[t] < 0.05).sum())
        pb = float(binom.sf(k - 1, n, 0.05))
        b = bh_count(df[t].values)
        agg.append({"test": t, "p<0.05": k, "binom_p": pb, "bh": b, "more_than_chance": pb < 0.001, "widely": b >= 0.1 * n})
    agg = pd.DataFrame(agg).set_index("test")
    chance = agg["more_than_chance"]
    if not chance.any():
        case = 1
    elif agg.loc[["bh_x1", "bh_x2", "trend_x1", "trend_x2"], "widely"].any():
        case = 2
    else:
        case = 3
    reached = bool(chance[["bh_x2", "trend_x2", "lb_x2", "runs_x2"]].any())
    words = {1: "No test rejects more often than chance: no detectable order structure at 50,000 shots; consistent "
                "with stationarity or with shuffled files.",
             2: "Block heterogeneity or trend rejects widely: file order carries drift. A3 is read as a temporal test.",
             3: "Only serial correlation or bursts reject more often than chance: A4 adds a block-bootstrap "
                "sensitivity analysis."}
    reach = ("A test on the decoder-error indicator (x2) rejects more often than chance, so the structure reaches "
             "the error rate." if reached else
             "No test on the decoder-error indicator (x2) rejects more often than chance: any structure found is in "
             "the syndrome statistics and does not reach the error rate at this sample size.")
    eff = df[["ratio_x1", "ratio_x2", "edge_x1", "edge_x2", "rho_x1", "rho_x2", "ac1_x1", "ac1_x2"]].describe(
        percentiles=[0.05, 0.5, 0.95]).T[["5%", "50%", "95%", "min", "max"]]
    by_d = df.groupby("distance")[TESTS].apply(lambda g: (g < 0.05).sum()).assign(experiments=df.groupby("distance").size())
    lines = ["# A2: shot order and independence", "",
             f"{n} experiments, {int(df.shots.iloc[0]):,} shots each, in file order. Method and classification rules: "
             "`scripts/p1_shot_order.py` (fixed before the data were read). x1 = detection events per shot; "
             "x2 = MWPM (RL prior) decoder-error indicator.", "",
             f"**Case {case}.** {words[case]} {reach}", "",
             "## Rejections per test", "", md_table(agg), "",
             f"Expected under no structure: {0.05 * n:.0f} of {n} below 0.05; BH rejections near 0.", "",
             "## Effect sizes (distribution over experiments)", "",
             "ratio = variance of 50 block means / its expectation under random order (1 = none); edge = relative "
             "change of the mean, last 10,000 vs first 10,000 shots; rho = Spearman correlation with shot index; "
             "ac1 = lag-1 autocorrelation.", "", md_table(eff), "",
             "## Experiments with p < 0.05, by distance", "", md_table(by_d), ""]
    (RES / "p1_shot_order.md").write_text("\n".join(lines))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
    u = (np.arange(1, n + 1) - 0.5) / n
    for ax, t in zip(axes.flat, TESTS):
        ax.loglog(u, np.sort(df[t].values), ".", ms=3)
        ax.loglog([u[0], 1], [u[0], 1], "k-", lw=0.8)
        ax.set_title(t)
        ax.set_xlabel("uniform quantile")
        ax.set_ylabel("p-value")
    axes.flat[-1].hist(df["ratio_x1"], bins=40, alpha=0.6, label="x1")
    axes.flat[-1].hist(df["ratio_x2"], bins=40, alpha=0.6, label="x2")
    axes.flat[-1].axvline(1, color="k", lw=0.8)
    axes.flat[-1].set_title("block variance ratio")
    axes.flat[-1].legend()
    fig.tight_layout()
    fig.savefig(RES / "p1_shot_order.png", dpi=200)
    print((RES / "p1_shot_order.md").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    exps = list_experiments()
    jobs = [(i, e) for i, e in enumerate(exps) if not (CACHE / (e.key.replace("/", "__") + ".json")).exists()]
    print(f"{len(exps) - len(jobs)} of {len(exps)} done; running {len(jobs)}", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, _ in enumerate(pool.map(run, jobs), 1):
            if k % 20 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    summarize(len(exps))


if __name__ == "__main__":
    main()
