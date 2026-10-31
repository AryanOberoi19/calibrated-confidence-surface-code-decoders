"""Early abort: the prefix DEM, Q-learning against backward induction, Dinkelbach, Learn-then-Test over policies."""
import numpy as np
import pytest
import stim

from qeccal.sequential import (Binner, certify_family, detector_steps, fitted_q, policy_family, prefix_dem,
                               prefix_gaps, q_learning_online, reward_rate, run, solve_rate)
from qeccal.soft.gap import LogicalGap


def _circuit(r=6, p=0.004):
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=r,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p)


def test_full_prefix_is_the_logical_gap():
    c = _circuit()
    dem = c.detector_error_model(decompose_errors=True)
    det, _ = c.compile_detector_sampler(seed=1).sample(3000, separate_observables=True)
    steps = detector_steps(c)
    assert np.all(np.diff(steps) >= 0)
    _, full = LogicalGap(dem).decode(det)
    g, cnt = prefix_gaps(dem, steps, det, [int(steps.max()) + 1])
    assert np.allclose(g[:, 0], full, atol=1e-4)
    assert np.array_equal(cnt[:, 0], det.sum(axis=1))


def test_prefix_dem_keeps_only_measured_detectors():
    c = _circuit()
    dem = c.detector_error_model(decompose_errors=True)
    n = int(np.searchsorted(detector_steps(c), 3))
    p = prefix_dem(dem, n)
    assert p.num_detectors == n and p.num_observables == 1
    for inst in p.flattened():
        if inst.type == "error":
            assert all(t.val < n for t in inst.targets_copy() if t.is_relative_detector_id())


def test_prefix_gap_predicts_final_failure():
    c = _circuit(r=10, p=0.008)
    dem = c.detector_error_model(decompose_errors=True)
    det, obs = c.compile_detector_sampler(seed=2).sample(20000, separate_observables=True)
    pred, _ = LogicalGap(dem).decode(det)
    wrong = pred != obs[:, 0]
    g, _ = prefix_gaps(dem, detector_steps(c), det, [5])
    lo, hi = g[:, 0] <= np.quantile(g[:, 0], 0.25), g[:, 0] >= np.quantile(g[:, 0], 0.75)
    assert wrong[lo].mean() > 1.2 * wrong[hi].mean()


def _synthetic(n=4000, K=3, seed=0):
    """Shot quality z; noisy views of it at K checkpoints and at the end (later views are sharper)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    X = np.stack([z + rng.normal(scale=1.0 - 0.2 * k, size=n) for k in range(K + 1)], axis=1)
    wrong = rng.random(n) < 1 / (1 + np.exp(3 * z + 1.5))
    return X, wrong, np.array([5.0, 10.0, 15.0, 20.0])[: K + 1]


def test_online_q_learning_matches_backward_induction():
    X, wrong, rounds = _synthetic()
    bn = Binner.fit(X, 8)
    B = bn.transform(X)
    a = fitted_q(B, rounds, keep_from=5, rho=0.01, sizes=bn.sizes)
    b = q_learning_online(B, rounds, keep_from=5, rho=0.01, sizes=bn.sizes)
    for x, y in zip(a.q_cont, b.q_cont):
        assert np.allclose(x, y)


def test_dinkelbach_finds_the_best_rate():
    X, wrong, rounds = _synthetic()
    bn = Binner.fit(X, 8)
    B = bn.transform(X)
    best = solve_rate(B, rounds, 5, bn.sizes)
    never = fitted_q(B, rounds, 5, 0.0, bn.sizes)                # same keep rule, never aborts
    assert reward_rate(best, B) >= reward_rate(never, B) - 1e-12
    assert best.rho == pytest.approx(reward_rate(best, B), rel=1e-9)
    _, stop = run(best, B)
    assert (stop < rounds[-1]).any()


def test_certified_policies_keep_the_promise():
    X, wrong, rounds = _synthetic(n=20000, seed=1)
    bn = Binner.fit(X, 10)
    fam = policy_family(bn.transform(X), rounds, bn.sizes)
    keeps = [k for k, _ in fam]
    assert keeps == sorted(keeps)
    Xb, wb, _ = _synthetic(n=200000, seed=2)
    Bb = bn.transform(Xb)
    alpha, delta = 0.05, 0.1
    fails = certs = 0
    for s in range(100):
        Xc, wc, _ = _synthetic(n=3000, seed=100 + s)
        i = certify_family(fam, bn.transform(Xc), wc, alpha, delta)
        if i is None:
            continue
        certs += 1
        keep, _ = run(fam[i][1], Bb)
        fails += int(wb[keep].mean() > alpha)
    assert certs > 50 and fails / certs <= delta + 0.05


def test_isotonic():
    from qeccal.sequential.mdp import isotonic
    y = np.array([0.0, 2.0, 1.0, 3.0, 9.0, 4.0])
    w = np.array([1.0, 1.0, 1.0, 0.0, 1.0, 3.0])
    out = isotonic(y, w)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.allclose(out[[0, 1, 2, 4, 5]], [0.0, 1.5, 1.5, 5.25, 5.25])
    assert out[3] == pytest.approx((1.5 + 5.25) / 2)
