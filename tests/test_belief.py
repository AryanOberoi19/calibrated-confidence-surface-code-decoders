"""Belief-matching gap: exact on a tree, close to the beliefmatching package, informative, input handling."""
import itertools

import numpy as np
import pymatching
import pytest
import stim

from qeccal.soft import LogicalGap, PreconditionError
from qeccal.soft.belief import BeliefMatchingGap


def circuit(d, r, p=0.005):
    return stim.Circuit.generated("surface_code:rotated_memory_x", distance=d, rounds=r,
                                  after_clifford_depolarization=p, before_round_data_depolarization=p,
                                  before_measure_flip_probability=p, after_reset_flip_probability=p)


@pytest.fixture(scope="module")
def sim():
    c = circuit(3, 3)
    det, obs = c.compile_detector_sampler(seed=11).sample(5_000, separate_observables=True)
    dem = c.detector_error_model(decompose_errors=True)
    pred, gap = BeliefMatchingGap(dem).decode(det)
    return dem, det, obs[:, 0], pred, gap


def test_edge_posteriors_exact_on_a_tree():
    # Tanner graph e0 - D0 - e1 - D1 - e2 is a tree, so BP marginals are exact; one component per mechanism.
    mech = [(0.1, (1, 0)), (0.2, (1, 1)), (0.3, (0, 1))]
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.3) D1")
    bm = BeliefMatchingGap(dem)
    for syn in itertools.product([0, 1], repeat=2):
        joint = np.zeros(3)
        z = 0.0
        for fires in itertools.product([0, 1], repeat=3):
            s = np.zeros(2, int)
            pr = 1.0
            for f, (p, pat) in zip(fires, mech):
                pr *= p if f else 1 - p
                if f:
                    s ^= np.array(pat)
            if tuple(s) == syn:
                z += pr
                joint += pr * np.array(fires)
        assert bm.edge_probabilities(np.array(syn, np.uint8)) == pytest.approx(joint / z, abs=1e-6)


def test_shared_component_combines_by_xor():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 ^ D2\nerror(0.2) D0 D1\nerror(0.05) D2 L0\nerror(0.1) D0")
    bm = BeliefMatchingGap(dem)
    syn = np.array([1, 1, 0], np.uint8)
    p_e = bm.edge_probabilities(syn)
    p_h = 1 / (1 + np.exp(bm.bp.llr(syn[None])[:, 0]))
    assert bm.edges.shape == (4, 4)                               # components: D0D1, D2, D2+L0, D0
    assert p_e[0] == pytest.approx(p_h[0] * (1 - p_h[1]) + p_h[1] * (1 - p_h[0]), rel=1e-9)


def test_bp_matches_ldpc_when_ldpc_runs_all_iterations(sim):
    ldpc = pytest.importorskip("ldpc")
    dem, det, *_ = sim
    bm = BeliefMatchingGap(dem)
    ref = ldpc.BpDecoder(pcm=bm.check, error_channel=list(bm.priors), max_iter=20, bp_method="product_sum",
                         input_vector_type="syndrome")
    llr = bm.bp.llr(det[:300])
    checked = 0
    for i in range(300):
        ref.decode(det[i].astype(np.uint8))
        if not ref.converge:                                       # ldpc stops early once it converges
            assert np.allclose(np.asarray(ref.log_prob_ratios), llr[:, i], atol=1e-8)
            checked += 1
    assert checked > 10


def test_close_to_beliefmatching_and_better_than_mwpm(sim):
    bmpkg = pytest.importorskip("beliefmatching")
    dem, det, obs, pred, gap = sim
    ref = bmpkg.BeliefMatching(dem, max_bp_iters=20).decode_batch(det)[:, 0].astype(bool)
    mwpm = pymatching.Matching.from_detector_error_model(dem).decode_batch(det)[:, 0].astype(bool)
    assert np.mean(pred == ref) > 0.98
    assert np.mean(pred != obs) <= np.mean(mwpm != obs) + 0.003
    assert (gap >= 0).all() and np.isfinite(gap).all()


def test_gap_is_informative(sim):
    _, _, obs, pred, gap = sim
    wrong = pred != obs
    lo = gap < np.quantile(gap, 0.2)
    assert wrong[lo].mean() > 3 * wrong[~lo].mean()


def test_packed_input_and_mwpm_gap_agree_in_scale(sim):
    dem, det, _, pred, gap = sim
    bm = BeliefMatchingGap(dem)
    p2, g2 = bm.decode(np.packbits(det[:200], axis=1, bitorder="little"), packed=True)
    assert np.array_equal(p2, pred[:200]) and np.allclose(g2, gap[:200])
    _, g_mwpm = LogicalGap(dem).decode(det)
    assert 0.2 < np.median(gap) / np.median(g_mwpm) < 5            # both are log-likelihood-ratio scale


def test_precondition_violation_raises():
    with pytest.raises(PreconditionError):
        BeliefMatchingGap(stim.DetectorErrorModel("error(0.1) D0 D1 L0\nerror(0.1) D0\nerror(0.1) D1"))
