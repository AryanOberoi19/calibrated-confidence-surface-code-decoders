"""Transfer helpers: a certified threshold holds in-distribution and breaks under a known shift; DEM sampling
matches circuit sampling."""
import numpy as np
import stim

from qeccal.guarantees import certify, evaluate
from qeccal.soft import LogicalGap, sample_gap
from test_m4 import synthetic


def test_certified_threshold_transfers_only_without_shift():
    alpha, delta = 0.02, 0.05
    s_tr, w_tr = synthetic(20_000, T=1.0, b=1.5, seed=1)
    s_ca, w_ca = synthetic(15_000, T=1.0, b=1.5, seed=2)
    lam, keep = certify(s_tr, s_ca, w_ca, alpha, delta)
    assert 0 < keep < 1
    same = evaluate(lam, *synthetic(200_000, T=1.0, b=1.5, seed=3), alpha)
    assert same["rate"] <= alpha and same["exceed"] == 0
    shifted = evaluate(lam, *synthetic(200_000, T=1.0, b=2.5, seed=4), alpha)   # noisier target
    assert shifted["rate"] > alpha and shifted["sig_exceed"] == 1


def test_dem_sampling_matches_circuit_sampling():
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                               after_clifford_depolarization=0.004, before_measure_flip_probability=0.004)
    dem = c.detector_error_model(decompose_errors=True)
    gap, wrong = sample_gap(dem, 100_000, seed=5)
    det, obs = c.compile_detector_sampler(seed=6).sample(100_000, separate_observables=True)
    pred, _ = LogicalGap(dem).decode(det)
    circ = np.mean(pred != obs[:, 0])
    assert gap.shape == wrong.shape == (100_000,)
    assert abs(wrong.mean() - circ) < 4 * np.sqrt(circ / 100_000) + 5e-4
