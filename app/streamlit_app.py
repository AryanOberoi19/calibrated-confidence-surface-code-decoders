"""Calibrated confidence and certified abort thresholds for surface-code decoders: interactive results.

Run locally:  streamlit run app/streamlit_app.py
Data: app/data/ (built by scripts/build_app_data.py) and results/*.csv (written by the milestone scripts).
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymatching
import stim
import streamlit as st

APP = pathlib.Path(__file__).resolve().parent
REPO = APP.parent
sys.path.insert(0, str(REPO / "src"))
from qeccal.guarantees.ltt import KEEP_GRID, evaluate, learn_then_test, loosest_where  # noqa: E402
from qeccal.soft.gap import LogicalGap  # noqa: E402

DATA, RES = APP / "data", REPO / "results"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical palette (same as the report)
GREY = "#898781"
RULES = {"ltt": ("Learn-then-Test (certified)", C[0]), "plugin": ("Plug-in on Calibrate", C[1]),
         "naive_raw": ("Trust raw q", C[2]), "naive_platt": ("Trust Platt-calibrated q", C[3])}
ALPHA_CHOICES = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2]
PATHWAY_NAME = {"correlated_matching_decoder_with_si1000_prior": "Correlated matching, SI1000 prior",
                "correlated_matching_decoder_with_rl_optimized_prior": "Correlated matching, RL prior",
                "harmony_decoder_with_si1000_prior": "Harmony, SI1000 prior",
                "harmony_decoder_with_rl_optimized_prior": "Harmony, RL prior",
                "libra_decoder_with_rl_optimized_prior": "Libra, RL prior"}

st.set_page_config(page_title="Calibrated confidence for surface-code decoders", layout="wide")


# ---------------------------------------------------------------------------------------------- data
def unpack(bits, n):
    return np.unpackbits(bits, count=n, bitorder="little").astype(bool)


def label(key):
    d, rest = key.split("_at_")
    patch, basis, r = rest.split("/")
    return f"d={d[1:]}, patch {patch}, {basis} basis, {int(r[1:])} rounds"


def safe(key):
    return key.replace("/", "__")


@st.cache_data
def manifest():
    return json.loads((DATA / "manifest.json").read_text())


@st.cache_data
def scores(key):
    z = np.load(DATA / "scores" / f"{safe(key)}.npz")
    return {"lams": z["lams"], "platt": z["platt"],
            "cal_s": z["cal_s"].astype(np.float64), "cal_w": unpack(z["cal_w"], int(z["n_cal"])),
            "test_s": z["test_s"].astype(np.float64), "test_w": unpack(z["test_w"], int(z["n_test"]))}


@st.cache_data
def csv(name):
    return pd.read_csv(RES / name)


@st.cache_resource
def explorer(key):
    z = np.load(DATA / "explorer" / f"{safe(key)}.npz")
    D = int(z["num_detectors"])
    det = np.unpackbits(z["det"], axis=1, bitorder="little")[:, :D].astype(bool)
    dem = stim.DetectorErrorModel(str(z["dem"]))
    pred, gap = LogicalGap(dem).decode(det)
    a, b = z["platt"]
    return {"det": det, "obs": z["obs"].astype(bool), "google": z["google"], "coords": z["coords"],
            "shot_index": z["shot_index"], "pred": pred, "gap": gap,
            "q_raw": 1 / (1 + np.exp(gap)), "q_platt": 1 / (1 + np.exp(a * gap + b)),
            "matching": pymatching.Matching.from_detector_error_model(dem)}


def fig_layout(fig, height=420, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=70),
                      legend=dict(orientation="h", yanchor="top", y=-0.3), **kw)
    fig.update_xaxes(dtick=1, exponentformat="power", selector=dict(type="log"))
    fig.update_yaxes(dtick=1, exponentformat="power", selector=dict(type="log"))
    return fig


# ------------------------------------------------------------------------------------------- views
def overview():
    st.markdown(
        "Quantum error correction decoders can report how confident they are in each correction. This project asks "
        "whether that confidence can be turned into **abort thresholds with a statistical guarantee**: discard the "
        "least trustworthy runs so that the logical error rate among the kept runs is at most α, with probability "
        "at least 1 − δ. It is evaluated on Google's public data from the Willow chip "
        "(420 surface-code memory experiments, 50,000 shots each), with the MWPM logical gap as the confidence score.")
    g = csv("m4_guarantees.csv")
    sel = g[(g.method == "mwpm_gap") & (g.prior == "rl") & (g.alpha == 0.01) & g.keep.notna()]
    d = csv("m5_drift.csv")
    dsel = d[(d.prior == "rl") & (d.alpha == 0.01) & (d.certified == 1)]
    c1, c2, c3 = st.columns(3)
    ltt = sel[sel.rule == "ltt"]
    raw = sel[sel.rule == "naive_raw"]
    patch = dsel[dsel.axis == "patch"]
    c1.metric("Certified, same experiment", f"{100 * ltt.sig_exceed.mean():.0f}%",
              help=f"Learn-then-Test; {len(ltt)} certified experiments")
    c2.metric("Trusting the decoder's confidence", f"{100 * raw.sig_exceed.mean():.0f}%",
              help=f"{len(raw)} experiments with a selected threshold")
    c3.metric("Certified, moved to another patch", f"{100 * patch.sig_exceed.mean():.0f}%",
              help=f"{len(patch)} source-target pairs")
    st.caption("Share of abort thresholds whose Test error among kept shots is significantly above the target "
               "α = 1% (one-sided 95% Clopper-Pearson). MWPM gap, RL-optimized prior.")
    st.markdown(
        "- **RQ1, calibration.** The raw gap is overconfident; temperature, Platt or isotonic recalibration fitted on "
        "held-out shots fixes most of it. At one round, the recalibrated gap matches the exact Bayesian posterior.\n"
        "- **RQ2, guarantees.** Learn-then-Test thresholds keep their promise on unseen shots; trusting the decoder's "
        "own confidence does not.\n"
        "- **RQ3, drift.** The guarantee is local: moved to another patch, basis or noise model, about one "
        "certified threshold in ten fails at α ≥ 1%, and thresholds certified on simulated data fail badly.\n\n"
        "Split discipline: calibrators are fitted on a Train split, thresholds chosen on a Calibrate split, and every "
        "number here is measured on a Test split that neither saw.")
    st.caption("Data: Google Quantum AI, Data for \"Quantum error correction below the surface code threshold\", "
               "Zenodo 10.5281/zenodo.13273331, CC BY 4.0. Code: github.com/AryanOberoi19/"
               "calibrated-confidence-surface-code-decoders.")


def shot_explorer():
    m = manifest()
    key = st.selectbox("Experiment", m["explorer"], format_func=label, index=2)
    ex = explorer(key)
    n = ex["det"].shape[0]
    wrong = ex["pred"] != ex["obs"]
    c1, c2 = st.columns([1, 2])
    with c1:
        which = st.radio("Shots", ["All", "MWPM wrong", "Least confident 10%"], horizontal=True)
        if which == "MWPM wrong":
            pool = np.flatnonzero(wrong)
        elif which == "Least confident 10%":
            pool = np.argsort(ex["gap"])[: max(1, n // 10)]
        else:
            pool = np.arange(n)
        if pool.size == 0:
            st.info("No shots of that kind in this sample.")
            return
        i = pool[st.slider("Shot", 0, pool.size - 1, 0) if pool.size > 1 else 0]
        truth, pred = bool(ex["obs"][i]), bool(ex["pred"][i])
        st.markdown(f"**Test shot #{int(ex['shot_index'][i])}** of this experiment")
        st.markdown(f"Logical observable actually flipped: **{'yes' if truth else 'no'}**")
        st.markdown(f"MWPM predicts: **{'flip' if pred else 'no flip'}** "
                    f"({'correct' if pred == truth else 'wrong'})")
        st.markdown(f"Logical gap: **{ex['gap'][i]:.2f}**; P(wrong): raw **{ex['q_raw'][i]:.2%}**, "
                    f"Platt-calibrated **{ex['q_platt'][i]:.2%}**")
        rows = []
        for p, g in zip(m["pathways"], ex["google"][:, i]):
            if g >= 0:
                rows.append({"Google decoder": PATHWAY_NAME[p], "Prediction": "flip" if g else "no flip",
                             "Correct": "yes" if bool(g) == truth else "no"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"{int(ex['det'][i].sum())} detection events. Sample: the first {n} Test shots; "
                   f"MWPM wrong on {wrong.sum()} of them.")
    with c2:
        xyz = ex["coords"]
        fired = ex["det"][i]
        edges = ex["matching"].decode_to_edges_array(ex["det"][i].astype(np.uint8))
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="detector",
                                   marker=dict(size=2, color=GREY, opacity=0.35), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=xyz[fired, 0], y=xyz[fired, 1], z=xyz[fired, 2], mode="markers",
                                   name="detection event", marker=dict(size=5, color=C[1]),
                                   text=[f"detector {j}" for j in np.flatnonzero(fired)], hoverinfo="text"))
        ex_, ey, ez, bx, by, bz = [], [], [], [], [], []
        for a, b in edges:
            if a >= 0 and b >= 0:
                for arr, v in ((ex_, 0), (ey, 1), (ez, 2)):
                    arr += [xyz[a, v], xyz[b, v], None]
            else:
                j = a if a >= 0 else b
                bx.append(xyz[j, 0])
                by.append(xyz[j, 1])
                bz.append(xyz[j, 2])
        fig.add_trace(go.Scatter3d(x=ex_, y=ey, z=ez, mode="lines", name="MWPM matching",
                                   line=dict(color=C[0], width=5), hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode="markers", name="matched to the boundary",
                                   marker=dict(size=7, color=C[0], symbol="diamond-open"), hoverinfo="skip"))
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="round", aspectmode="manual",
                                     aspectratio=dict(x=1, y=1, z=1.4)))
        st.plotly_chart(fig_layout(fig, 560, title="Detection events in space-time and the MWPM matching"),
                        width="stretch")


def calibration():
    rel = pd.read_csv(DATA / "reliability.csv")
    c1, c2, c3, c4 = st.columns(4)
    d = c1.selectbox("Distance", [3, 5, 7])
    r = c2.selectbox("Rounds", [1, 10, 30, 50], index=1)
    prior = c3.selectbox("Prior DEM", ["rl", "si1000"], format_func={"rl": "RL-optimized (fitted)", "si1000": "SI1000"}.get)
    methods = sorted(rel[(rel.distance == d) & (rel.rounds == r)].method.unique(), key=lambda m: m != "mwpm_gap")
    method = c4.selectbox("Soft output", methods, format_func={"mwpm_gap": "MWPM gap", "exact": "Exact posterior"}.get)
    sel = rel[(rel.distance == d) & (rel.rounds == r) & (rel.prior == prior) & (rel.method == method)]
    fig = go.Figure()
    lo = max(1e-6, sel.mean_q[sel.mean_q > 0].min() / 2)
    fig.add_trace(go.Scatter(x=[lo, 0.5], y=[lo, 0.5], mode="lines", name="perfect calibration",
                             line=dict(color=GREY, dash="dash", width=1)))
    for (name, col) in zip(["raw", "temperature", "platt", "isotonic"], C):
        s = sel[sel.calibrator == name]
        fig.add_trace(go.Scatter(x=s.mean_q, y=s.observed.where(s.observed > 0), mode="lines+markers", name=name,
                                 line=dict(color=col, width=2), marker=dict(size=7),
                                 customdata=s["count"], hovertemplate="mean q %{x:.2e}<br>observed %{y:.2e}"
                                 "<br>%{customdata} shots<extra>" + name + "</extra>"))
    fig.update_xaxes(type="log", title="Predicted P(wrong), bin mean")
    fig.update_yaxes(type="log", title="Observed error rate in bin")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig_layout(fig, 460, title="Reliability on the Test split (10 equal-mass bins)"),
                    width="stretch")
    m4 = csv("m4_calibration.csv")
    t = m4[(m4.distance == d) & (m4.rounds == r) & (m4.prior == prior) & (m4.method == method)]
    table = t.groupby("calibrator")[["ece", "brier", "nll"]].mean().reindex(["raw", "temperature", "platt", "isotonic"])
    c2.markdown("**Mean over patches and bases (Test)**")
    c2.dataframe(table.style.format("{:.5f}"), width="stretch")
    c2.caption("Points on the diagonal are perfectly calibrated; above it, the method is overconfident. Calibrators "
               "are fitted on the Train split of each experiment. Bins with no errors are not drawn (log scale).")


def postselection():
    m = manifest()
    exps = pd.DataFrame(m["experiments"])
    c1, c2, c3, c4 = st.columns(4)
    d = c1.selectbox("Distance", [3, 5, 7], key="ps_d")
    r = c2.selectbox("Rounds", [1, 10, 30, 50], index=1, key="ps_r")
    basis = c3.selectbox("Basis", ["X", "Z"], key="ps_b")
    patch = c4.selectbox("Patch", sorted(exps[(exps.distance == d)].patch.unique()), key="ps_p")
    c1, c2 = st.columns(2)
    alpha = c1.select_slider("Target α (error rate among kept shots)", ALPHA_CHOICES, value=1e-2,
                             format_func=lambda a: f"{a:g}")
    delta = c2.select_slider("δ (allowed probability of missing α)", [0.01, 0.05, 0.1], value=0.05)
    key = f"d{d}_at_{patch}/{basis}/r{r:02d}"
    s = scores(key)
    lams = s["lams"]
    kept_ca = [s["cal_s"] >= lam for lam in lams]
    q_raw = 1 / (1 + np.exp(s["cal_s"]))
    a, b = s["platt"]
    q_pl = 1 / (1 + np.exp(a * s["cal_s"] + b))
    choice = {
        "ltt": learn_then_test(s["cal_s"], s["cal_w"], lams, alpha, delta)[0],
        "plugin": loosest_where([s["cal_w"][k].mean() if k.any() else np.inf for k in kept_ca], lambda v: v <= alpha),
        "naive_raw": loosest_where([q_raw[k].mean() if k.any() else np.inf for k in kept_ca], lambda v: v <= alpha),
        "naive_platt": loosest_where([q_pl[k].mean() if k.any() else np.inf for k in kept_ca], lambda v: v <= alpha),
    }
    curve = [evaluate(lam, s["test_s"], s["test_w"], alpha) for lam in lams]
    kept = np.array([c["n"] for c in curve]) / s["test_s"].size
    rate = np.array([c["rate"] for c in curve])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=kept, y=np.where(rate > 0, rate, np.nan), mode="lines+markers", name="Test error among kept",
                             line=dict(color=GREY, width=2), marker=dict(size=5),
                             hovertemplate="kept %{x:.0%}<br>error %{y:.2e}<extra></extra>"))
    fig.add_hline(y=alpha, line=dict(color=GREY, dash="dash", width=1), annotation_text=f"α = {alpha:g}")
    rows = []
    for size, (rule, i) in zip((24, 19, 14, 9), choice.items()):
        name, col = RULES[rule]
        if i is None:
            rows.append({"Rule": name, "Keep (Train quantile)": "none", "Kept (Test)": "", "Errors / kept": "",
                         "Error rate": "", "Meets α?": ""})
            continue
        ev = curve[i]
        if ev["rate"] > 0:
            fig.add_trace(go.Scatter(x=[kept[i]], y=[ev["rate"]], mode="markers", name=name,
                                     marker=dict(size=size, color=col, symbol="circle-open", line=dict(width=3)),
                                     hovertemplate=f"{name}<br>kept %{{x:.1%}}<br>error %{{y:.2e}}<extra></extra>"))
        rows.append({"Rule": name, "Keep (Train quantile)": f"{KEEP_GRID[i]:.0%}", "Kept (Test)": f"{kept[i]:.1%}",
                     "Errors / kept": f"{ev['errors']} / {ev['n']:,}", "Error rate": f"{ev['rate']:.2e}",
                     "Meets α?": "yes" if ev["rate"] <= alpha else ("no, significantly" if ev["sig_exceed"] else "no")})
    fig.update_xaxes(title="Fraction of Test shots kept", autorange="reversed", tickformat=".0%")
    fig.update_yaxes(type="log", title="Logical error among kept shots")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(fig_layout(fig, 440, title=label(key)), width="stretch")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    ltt = choice["ltt"]
    c2.markdown(
        f"**Learn-then-Test** {'certifies keeping ' + f'{KEEP_GRID[ltt]:.0%}' + ' of shots' if ltt is not None else 'cannot certify α here'}: "
        f"with probability at least {1 - delta:.0%} over the Calibrate draw, the error rate among kept shots is at most "
        f"{alpha:g}. It tests thresholds from strict to loose on the Calibrate split with binomial p-values and stops "
        "at the first that fails. The other rules take no statistical margin; the raw-q rule trusts the decoder's own "
        "probabilities.")
    c2.caption("Thresholds come from the Train split, selection from the Calibrate split, and everything plotted is "
               "on the Test split. MWPM gap, RL prior. Points with zero errors are not drawn (log scale).")


def drift():
    d = csv("m5_drift.csv")
    prior = st.radio("Prior DEM", ["rl", "si1000"], horizontal=True,
                     format_func={"rl": "RL-optimized (fitted)", "si1000": "SI1000"}.get)
    names = {"same": "Same experiment (reference)", "patch": "Other patch", "basis": "Other basis",
             "pooled": "Other patches pooled", "sim": "Simulated from the DEM"}
    cols = {"same": C[0], "patch": C[1], "basis": C[2], "pooled": C[3], "sim": "#52514e"}
    cert = d[(d.prior == prior) & (d.certified == 1)]
    agg = cert.groupby(["axis", "alpha"]).agg(sig=("sig_exceed", "mean"), exc=("exceed", "mean"), n=("sig_exceed", "size")).reset_index()
    fig = go.Figure()
    for axis, name in names.items():
        a = agg[agg.axis == axis]
        fig.add_trace(go.Scatter(x=a.alpha, y=100 * a.sig, mode="lines+markers", name=name,
                                 line=dict(color=cols[axis], width=2, dash="dot" if axis == "sim" else None),
                                 customdata=np.c_[100 * a.exc, a.n],
                                 hovertemplate="α %{x:g}<br>significant %{y:.0f}%<br>any exceedance %{customdata[0]:.0f}%"
                                 "<br>%{customdata[1]} certified pairs<extra>" + name + "</extra>"))
    fig.add_hline(y=5, line=dict(color=GREY, dash="dash", width=1), annotation_text="δ = 5%")
    fig.update_xaxes(type="log", title="Target α")
    fig.update_yaxes(title="Significant exceedance (%)")
    st.plotly_chart(fig_layout(fig, 420, title="Certify on a source, test on a shifted target"), width="stretch")

    st.markdown("**Patch to patch**: each cell certifies on the row patch and measures on the column patch's Test split.")
    c1, c2, c3, c4 = st.columns(4)
    dist = c1.selectbox("Distance", [3, 5], key="dr_d")
    r = c2.selectbox("Rounds", [1, 10, 30, 50], index=1, key="dr_r")
    basis = c3.selectbox("Basis", ["X", "Z"], key="dr_b")
    alpha = c4.selectbox("α", sorted(d.alpha.unique()), index=3, format_func=lambda a: f"{a:g}", key="dr_a")
    sub = d[(d.prior == prior) & (d.distance == dist) & (d.rounds == r) & (d.basis == basis) & (d.alpha == alpha)
            & d.axis.isin(["patch", "same"])].copy()
    sub["src"] = np.where(sub.axis == "same", sub.patch, sub.source.str.split("_at_").str[1].str.split("/").str[0])
    order = sorted(sub.patch.unique(), key=lambda p: tuple(int(x) for x in p[1:].split("_")))
    ratio = sub.pivot_table(index="src", columns="patch", values="test_rate", aggfunc="first").reindex(index=order, columns=order) / alpha
    if ratio.notna().sum().sum() == 0:
        st.info("Nothing is certifiable at this α for these experiments.")
        return
    z = np.log10(ratio.clip(lower=0.1, upper=10))
    text = np.where(ratio.notna().values, np.vectorize(lambda v: f"{v:.2f}")(ratio.fillna(0).values), "")
    fig = go.Figure(go.Heatmap(z=z.values, x=order, y=order, zmin=-1, zmax=1,
                               colorscale=[[0, C[0]], [0.5, "#f2f1ec"], [1, C[1]]],
                               text=text, texttemplate="%{text}",
                               hovertemplate="source %{y}<br>target %{x}<br>Test error / α = %{text}<extra></extra>",
                               colorbar=dict(title="error / α", tickvals=[-1, 0, 1], ticktext=["0.1", "1", "10"])))
    fig.update_xaxes(title="Target patch (Test)")
    fig.update_yaxes(title="Source patch (certified on)", autorange="reversed")
    st.plotly_chart(fig_layout(fig, 480), width="stretch")
    st.caption("Blue: the target's Test error is below α; orange: above (the promise failed on that target). "
               "Empty cells: not certifiable on the source. MWPM gap; δ = 0.05.")


st.title("Calibrated confidence for surface-code decoders")
tabs = st.tabs(["Overview", "Shot explorer", "Calibration", "Postselection", "Drift"])
for tab, view in zip(tabs, (overview, shot_explorer, calibration, postselection, drift)):
    with tab:
        view()
