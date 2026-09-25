"""Build the small, committed data files the web app reads (app/data/). Run on a machine with the extracted
archive and the M2 soft-output cache; the app itself never needs the raw data.

  scores/<key>.npz    for every experiment at r in 1, 10, 30, 50 (RL prior, MWPM gap): the 20 Train-defined
                      thresholds, Platt parameters fitted on Train, and the Calibrate and Test scores and labels
  explorer/<key>.npz  300 Test shots of a few experiments: detection events, truth, Google's predictions,
                      detector coordinates and the RL-prior DEM, for live decoding in the shot explorer
  reliability.csv     Test reliability bins (10 equal-mass) pooled over patches and bases, per calibrator
  manifest.json       what is where

    python scripts/build_app_data.py
"""
import csv
import json
import pathlib

import numpy as np

from qeccal.calibration import CALIBRATORS, reliability, score_from_q
from qeccal.data import PATHWAYS, get_experiment, list_experiments, split_indices
from qeccal.guarantees import KEEP_GRID, thresholds
from qeccal.soft.cache import cache_path, load

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "app" / "data"
ROUNDS = (1, 10, 30, 50)
EXPLORER = ("d3_at_q6_7/Z/r01", "d3_at_q6_7/X/r10", "d3_at_q6_7/Z/r10", "d5_at_q6_5/X/r10", "d5_at_q6_5/Z/r10",
            "d7_at_q6_7/X/r10", "d7_at_q6_7/Z/r10", "d7_at_q6_7/Z/r30")
N_EXPLORER = 300


def safe(key):
    return key.replace("/", "__")


def pack(w):
    return np.packbits(np.asarray(w, bool), bitorder="little")


def main():
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    (OUT / "explorer").mkdir(parents=True, exist_ok=True)
    exps = list_experiments(rounds=ROUNDS)
    manifest = {"prior": "rl", "method": "mwpm_gap", "keep_grid": [float(k) for k in KEEP_GRID], "experiments": [],
                "explorer": list(EXPLORER), "pathways": list(PATHWAYS)}
    pooled = {}
    for e in exps:
        idx = split_indices(e)
        obs = e.observable_flips()
        for prior in ("rl", "si1000"):
            for method in ("mwpm_gap", "exact", "bm_gap", "nn"):
                if not cache_path(method, prior, e.key).exists():
                    continue
                o = load(method, prior, e.key)
                s = o["gap"].astype(np.float64) if "gap" in o else score_from_q(o["q"])
                w = o["pred"] != obs
                tr, ca, te = idx["train"], idx["calibrate"], idx["test"]
                if "fit_mask" in o:
                    tr = tr[~o["fit_mask"][tr]]
                for name, C in CALIBRATORS.items():
                    q = C().fit(s[tr], w[tr]).predict(s[te])
                    k = (e.distance, e.rounds, prior, method, name)
                    pooled.setdefault(k, ([], []))
                    pooled[k][0].append(q)
                    pooled[k][1].append(w[te])
                if prior == "rl" and method == "mwpm_gap":
                    platt = CALIBRATORS["platt"]().fit(s[tr], w[tr])
                    np.savez_compressed(OUT / "scores" / f"{safe(e.key)}.npz", lams=thresholds(s[tr]),
                                        platt=np.array([platt.a, platt.b]), cal_s=s[ca].astype(np.float32),
                                        cal_w=pack(w[ca]), n_cal=ca.size, test_s=s[te].astype(np.float32),
                                        test_w=pack(w[te]), n_test=te.size)
                    manifest["experiments"].append({"key": e.key, "distance": e.distance, "patch": e.patch,
                                                    "basis": e.basis, "rounds": e.rounds, "n_cal": int(ca.size),
                                                    "n_test": int(te.size)})
                    if e.key in EXPLORER:
                        sel = te[:N_EXPLORER]
                        det = e.detection_events(packed=True)[sel]
                        coords = e.circuit().get_detector_coordinates()
                        xyz = np.array([coords[i][:3] for i in range(e.num_detectors)], dtype=np.float32)
                        google = np.array([g[sel] if (g := e.google_predictions(p)) is not None
                                           else np.full(sel.size, -1) for p in PATHWAYS], dtype=np.int8)
                        np.savez_compressed(OUT / "explorer" / f"{safe(e.key)}.npz", det=det, obs=obs[sel],
                                            google=google, shot_index=sel, coords=xyz,
                                            num_detectors=e.num_detectors, dem=np.array(str(e.dem("rl"))),
                                            platt=np.array([platt.a, platt.b]))
        print(e.key, flush=True)

    with (OUT / "reliability.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["distance", "rounds", "prior", "method", "calibrator", "bin", "mean_q", "observed", "count"])
        for (d, r, prior, method, name), (qs, ws) in sorted(pooled.items()):
            mq, mw, cnt = reliability(np.concatenate(qs), np.concatenate(ws), n_bins=10)
            for b in range(len(mq)):
                wr.writerow([d, r, prior, method, name, b, f"{mq[b]:.6g}", f"{mw[b]:.6g}", int(cnt[b])])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"wrote {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
