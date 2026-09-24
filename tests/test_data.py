"""Loader and split tests on a synthetic tree, plus a check of the committed split manifest."""
import json

import numpy as np
import pymatching
import pytest
import stim

from qeccal.data import ROLES, get_experiment, list_experiments, split_indices, verify_manifest
from qeccal.data import splits as splits_mod
from synthetic import SHOTS, make_tree


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return make_tree(tmp_path_factory.mktemp("willow"))


def test_list_and_filter(root):
    exps = list_experiments(root)
    assert [e.key for e in exps] == ["d3_at_q2_7/X/r10", "d3_at_q4_9/Z/r01", "d3_at_q4_9/Z/r10", "d3_at_q4_9/Z/r13",
                                     "d5_at_q6_7/Z/r10"]
    assert len(list_experiments(root, distance=3)) == 4
    assert len(list_experiments(root, rounds=10)) == 3
    assert [e.patch for e in list_experiments(root, basis="X")] == ["q2_7"]
    assert len(list_experiments(root, distance=[3, 5], rounds=[1, 13])) == 2
    assert get_experiment("d3_at_q4_9/Z/r1", root).key == "d3_at_q4_9/Z/r01"
    with pytest.raises(KeyError):
        get_experiment("d7_at_q6_7/Z/r10", root)
    with pytest.raises(FileNotFoundError):
        list_experiments(root / "missing")


def test_shot_data_matches_stim(root):
    e = get_experiment("d3_at_q4_9/Z/r10", root)
    assert e.shots == SHOTS and e.num_detectors == 80
    det = e.detection_events()
    ref = stim.read_shot_data_file(path=str(e.path / "detection_events.b8"), format="b8", num_detectors=80)
    assert det.dtype == bool and det.shape == (SHOTS, 80) and np.array_equal(det, ref)
    assert e.detection_events(packed=True).shape == (SHOTS, 10)
    obs = e.observable_flips()
    ref_obs = stim.read_shot_data_file(path=str(e.path / "obs_flips_actual.b8"), format="b8", num_observables=1)[:, 0]
    assert obs.shape == (SHOTS,) and np.array_equal(obs, ref_obs)
    assert np.array_equal(e.google_predictions("correlated_matching_decoder_with_si1000_prior"), obs)
    assert e.google_predictions("libra_decoder_with_rl_optimized_prior") is None


def test_dems_and_packed_decoding(root):
    e = get_experiment("d5_at_q6_7/Z/r10", root)
    for prior in ("si1000", "rl", "si1000_circuit"):
        assert e.dem(prior).num_detectors == e.num_detectors
    with pytest.raises(ValueError):
        e.dem("fitted")
    m = pymatching.Matching.from_detector_error_model(e.dem("si1000"))
    a = m.decode_batch(e.detection_events())
    b = m.decode_batch(e.detection_events(packed=True), bit_packed_shots=True)
    assert np.array_equal(a, b)                       # packed layout is what PyMatching expects
    assert np.mean(a[:, 0] != e.observable_flips()) < 0.2


def test_splits(root):
    e = get_experiment("d3_at_q4_9/Z/r10", root)
    s = split_indices(e)
    assert {k: len(v) for k, v in s.items()} == {"train": 400, "calibrate": 300, "test": 300}
    allidx = np.concatenate(list(s.values()))
    assert np.array_equal(np.sort(allidx), np.arange(SHOTS))          # disjoint and complete
    assert all(np.array_equal(split_indices(e)[k], v) for k, v in s.items())   # deterministic
    other = split_indices(get_experiment("d3_at_q2_7/X/r10", root))
    assert not np.array_equal(other["test"], s["test"])               # seeded per experiment
    held = split_indices(get_experiment("d3_at_q4_9/Z/r13", root))
    assert list(held) == ["sanity"] and len(held["sanity"]) == SHOTS
    assert ROLES == ("train", "calibrate", "test", "sanity")


def test_manifest_roundtrip(root, tmp_path):
    path = tmp_path / "splits.json"
    splits_mod.write_manifest(list_experiments(root), path)
    assert verify_manifest(path) == []
    m = json.loads(path.read_text())
    m["experiments"]["d3_at_q4_9/Z/r10"]["sha256"] = "0" * 64
    path.write_text(json.dumps(m))
    assert verify_manifest(path) == ["d3_at_q4_9/Z/r10"]


def test_committed_manifest_still_reproduces():
    if not splits_mod.MANIFEST.exists():
        pytest.skip("splits manifest not generated yet")
    assert verify_manifest() == []
