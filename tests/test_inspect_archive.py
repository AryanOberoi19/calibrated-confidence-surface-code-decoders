"""Run scripts/inspect_archive.py on a small synthetic archive laid out like the real Zenodo 13273331 zip."""
import json
import pathlib
import subprocess
import sys
import zipfile

import numpy as np
import stim

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = "google_105Q_surface_code_d3_d5_d7"
SHOTS = 1000
CONFIGS = [(3, "q4_9", "Z", 1), (3, "q4_9", "Z", 10), (3, "q2_7", "X", 10), (5, "q6_7", "Z", 10)]


def b8(tmp, data, **kw):
    stim.write_shot_data_file(data=data, path=str(tmp), format="b8", **kw)
    return tmp.read_bytes()


def make_archive(path):
    tmp = path.parent / "tmp.b8"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{ROOT}/README/README.md", "# synthetic test archive\n")
        zf.writestr(f"{ROOT}/README/patches.png", b"\x89PNG\r\n\x1a\n" + bytes(64))
        for d, patch, basis, r in CONFIGS:
            kind = "surface_code:rotated_memory_" + basis.lower()
            ideal = stim.Circuit.generated(kind, distance=d, rounds=r)
            noisy = stim.Circuit.generated(kind, distance=d, rounds=r, before_measure_flip_probability=0.01,
                                           after_clifford_depolarization=0.005)
            det, obs = noisy.compile_detector_sampler(seed=1).sample(SHOTS, separate_observables=True)
            base = f"{ROOT}/d{d}_at_{patch}/{basis}/r{r:02d}"
            zf.writestr(f"{base}/circuit_ideal.stim", str(ideal))
            zf.writestr(f"{base}/circuit_noisy_si1000.stim", str(noisy))
            zf.writestr(f"{base}/detection_events.b8", b8(tmp, det, num_detectors=ideal.num_detectors))
            zf.writestr(f"{base}/obs_flips_actual.b8", b8(tmp, obs, num_observables=1))
            zf.writestr(f"{base}/metadata.json", json.dumps({"distance": d, "rounds": r, "shots": SHOTS}))
            dem = str(noisy.detector_error_model(decompose_errors=True))
            pred = np.zeros((SHOTS, 1), dtype=bool)
            pathways = ["matching_si1000"] + ([] if r == 1 else ["ensemble_rl"])   # second pathway missing at r=1
            for p in pathways:
                zf.writestr(f"{base}/decoding_results/{p}/error_model.dem", dem)
                zf.writestr(f"{base}/decoding_results/{p}/obs_flips_predicted.b8", b8(tmp, pred, num_observables=1))


def test_inspect_archive(tmp_path):
    archive = tmp_path / "fake.zip"
    make_archive(archive)
    out = tmp_path / "docs"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "inspect_archive.py"), str(archive), "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    strays = [ln for ln in res.stdout.splitlines() if ln.startswith("other directories")]
    assert strays == [f"other directories (1): {ROOT}/README"]   # pathway folders are attached, not listed as strays

    inv = json.loads((out / "data_inventory.json").read_text())
    assert len(inv) == 4
    assert all(c["shots"] == SHOTS for c in inv)          # includes the obs_flips_predicted files
    assert {c["num_detectors"] for c in inv if c["distance"] == 3 and c["basis"] == "Z"} == {8, 80}
    assert sorted(len(c["pathways"]) for c in inv) == [1, 2, 2, 2]

    md = (out / "data_inventory.md").read_text()
    assert "`obs_flips_actual.b8`" in md
    assert "| `ensemble_rl` | 3 of 4 | d3 r1: 1 |" in md
    assert "Decoder predictions present: yes, 2 pathways" in md
    assert "`circuit_noisy_si1000.stim`" in md
    assert "metadata.json `shots` differs from the file-derived count: 0" in md
    assert "Time-related keys: none found" in md

    readme = (out / "archive_README.txt").read_text()
    assert "synthetic test archive" in readme and "PNG" not in readme
