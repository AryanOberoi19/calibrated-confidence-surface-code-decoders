"""Logical gap, exact posterior and selective metrics on small Stim circuits with known answers."""
import itertools

import numpy as np
import pymatching
import pytest
import stim

from qeccal.calibration import brier, ece, error_vs_discard, nll, reliability
from qeccal.exact import ExactPosterior, mechanisms
from qeccal.soft import LogicalGap, PreconditionError, gap_dem, q_from_gap


def circuit(d, r, p=0.004, basis="z"):
    return stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=r,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


@pytest.fixture(scope="module")
def sim():
    c = circuit(3, 3)
    det, obs = c.compile_detector_sampler(seed=7).sample(20_000, separate_observables=True)
    return c, c.detector_error_model(decompose_errors=True), det, obs[:, 0]


def test_gap_matches_plain_matching(sim):
    c, dem, det, obs = sim
    g = gap_dem(dem)
    assert g.num_detectors == dem.num_detectors + 1
    pred, gap = LogicalGap(dem).decode(det)
    plain = pymatching.Matching.from_detector_error_model(dem).decode_batch(det)[:, 0].astype(bool)
    decided = gap > 1e-9
    assert np.array_equal(pred[decided], plain[decided])      # ties aside, same predictions as plain MWPM
    assert (gap >= 0).all()
    packed = np.packbits(det, axis=1, bitorder="little")
    p2, g2 = LogicalGap(dem).decode(packed, packed=True, chunk=777)   # packed input and chunking agree
    assert np.array_equal(p2, pred) and np.allclose(g2, gap)


def test_gap_is_informative(sim):
    _, dem, det, obs = sim
    pred, gap = LogicalGap(dem).decode(det)
    wrong = pred != obs
    lo = gap < np.quantile(gap, 0.2)
    assert wrong[lo].mean() > 3 * wrong[~lo].mean()             # small gaps carry most errors
    q = q_from_gap(gap)
    assert q.min() > 0 and q.max() <= 0.5


def test_precondition_violation_raises():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1")
    with pytest.raises(PreconditionError):
        gap_dem(dem)


def test_exact_posterior_matches_brute_force():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.2) D0 D1
        error(0.05) D1 ^ D2 L0
        error(0.3) D2
        error(0.1) D0 L0
    """)
    ex = ExactPosterior(dem)
    mech = [(0.1, (1, 0, 0), 1), (0.2, (1, 1, 0), 0), (0.05, (0, 1, 1), 1), (0.3, (0, 0, 1), 0), (0.1, (1, 0, 0), 1)]
    table = {}
    for fires in itertools.product([0, 1], repeat=len(mech)):
        pr, s, o = 1.0, np.zeros(3, int), 0
        for f, (p, pat, ob) in zip(fires, mech):
            pr *= p if f else 1 - p
            if f:
                s ^= np.array(pat)
                o ^= ob
        k = tuple(s)
        table.setdefault(k, [0.0, 0.0])[o] += pr
    for k, (p0, p1) in table.items():
        assert ex.p_flip(np.array([k], bool))[0] == pytest.approx(p1 / (p0 + p1), rel=1e-12)
    assert len(mechanisms(dem)) == 4                               # the two identical mechanisms merge


def test_exact_posterior_beats_matching_and_is_calibrated():
    c = circuit(3, 1, p=0.01)
    ex = ExactPosterior(c.detector_error_model(decompose_errors=False))
    det, obs = c.compile_detector_sampler(seed=3).sample(100_000, separate_observables=True)
    pred, q = ex.decode(det)
    wrong = pred != obs[:, 0]
    plain = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
    mw_wrong = plain.decode_batch(det)[:, 0] != obs[:, 0]
    assert wrong.mean() <= mw_wrong.mean() + 1e-3
    assert abs(q.mean() - wrong.mean()) < 3 * np.sqrt(wrong.mean() / wrong.size) + 1e-4   # calibrated in-model
    with pytest.raises(ValueError):
        ExactPosterior(circuit(5, 3).detector_error_model())      # too many bits


def test_selective_metrics():
    rng = np.random.default_rng(0)
    q = np.array([0.4, 0.3, 0.3, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    wrong = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], bool)
    kept, err = error_vs_discard(q, wrong, [0.0, 0.1, 0.3, 0.5], rng)
    assert kept.tolist() == [10, 9, 7, 5] and err.tolist()[0] == 2 and err.tolist()[2:] == [0, 0]
    assert err[1] == 1
    mq, mw, cnt = reliability(q, wrong, n_bins=5)
    assert cnt.sum() == 10 and np.all(np.diff(mq) >= 0)
    assert 0 <= ece(q, wrong, 5) <= 1 and brier(q, wrong) < 0.2 and nll(q, wrong) > 0
