"""Run scripts/inspect_archive.py on a small synthetic archive with the expected layout."""
import json
import pathlib
import subprocess
import sys
import zipfile

import stim

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = "google_105Q_surface_code_d3_d5_d7"
SHOTS = 1000


def make_archive(path):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{ROOT}/README.md", "synthetic test archive\n")
        for d, patch, basis, r in [(3, "q4_9", "Z", 1), (3, "q4_9", "Z", 10), (3, "q2_7", "X", 10), (5, "q6_7", "Z", 10)]:
            kind = "surface_code:rotated_memory_" + basis.lower()
            ideal = stim.Circuit.generated(kind, distance=d, rounds=r)
            noisy = stim.Circuit.generated(kind, distance=d, rounds=r, before_measure_flip_probability=0.01,
                                           after_clifford_depolarization=0.005)
            det, obs = noisy.compile_detector_sampler(seed=1).sample(SHOTS, separate_observables=True)
            base = f"{ROOT}/d{d}_at_{patch}/{basis}/r{r:02d}"
            zf.writestr(f"{base}/circuit_ideal.stim", str(ideal))
            tmp = path.parent / "tmp.b8"
            stim.write_shot_data_file(data=det, path=str(tmp), format="b8", num_detectors=ideal.num_detectors)
            zf.writestr(f"{base}/detection_events.b8", tmp.read_bytes())
            stim.write_shot_data_file(data=obs, path=str(tmp), format="01", num_observables=1)
            zf.writestr(f"{base}/obs_flips_actual.01", tmp.read_bytes())
            zf.writestr(f"{base}/metadata.json", json.dumps({"distance": d, "rounds": r, "start_time": "2024-01-01"}))


def test_inspect_archive(tmp_path):
    archive = tmp_path / "fake.zip"
    make_archive(archive)
    out = tmp_path / "docs"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "inspect_archive.py"), str(archive), "--out", str(out)],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr + res.stdout
    inv = json.loads((out / "data_inventory.json").read_text())
    assert len(inv) == 4
    assert all(c["shots"] == SHOTS for c in inv)
    assert {c["num_detectors"] for c in inv if c["distance"] == 3 and c["basis"] == "Z"} == {8, 80}
    md = (out / "data_inventory.md").read_text()
    assert "`obs_flips_actual.01`" in md and "Time-related keys: `start_time`" in md
    assert (out / "archive_README.txt").exists()
