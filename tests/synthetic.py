"""A small extracted archive with the same layout as the real one, for loader tests."""
import json
import pathlib

import numpy as np
import stim

SHOTS = 1000
CONFIGS = [(3, "q4_9", "Z", 1), (3, "q4_9", "Z", 10), (3, "q4_9", "Z", 13), (3, "q2_7", "X", 10), (5, "q6_7", "Z", 10)]
PRIOR_DIRS = ("correlated_matching_decoder_with_si1000_prior", "correlated_matching_decoder_with_rl_optimized_prior")


def write_b8(path, data):
    data = np.asarray(data, dtype=bool)
    np.packbits(data, axis=1, bitorder="little").tofile(path)


def make_tree(root: pathlib.Path, shots: int = SHOTS) -> pathlib.Path:
    """Writes <root>/google_105Q_surface_code_d3_d5_d7/... and returns that directory."""
    top = root / "google_105Q_surface_code_d3_d5_d7"
    (top / "README").mkdir(parents=True)
    (top / "README" / "README.md").write_text("# synthetic\n")
    for d, patch, basis, r in CONFIGS:
        kind = "surface_code:rotated_memory_" + basis.lower()
        ideal = stim.Circuit.generated(kind, distance=d, rounds=r)
        noisy = stim.Circuit.generated(kind, distance=d, rounds=r, before_measure_flip_probability=0.01,
                                       after_clifford_depolarization=0.005)
        det, obs = noisy.compile_detector_sampler(seed=d * 100 + r).sample(shots, separate_observables=True)
        e = top / f"d{d}_at_{patch}" / basis / f"r{r:02d}"
        e.mkdir(parents=True)
        (e / "circuit_ideal.stim").write_text(str(ideal))
        (e / "circuit_noisy_si1000.stim").write_text(str(noisy))
        write_b8(e / "detection_events.b8", det)
        write_b8(e / "obs_flips_actual.b8", obs)
        (e / "metadata.json").write_text(json.dumps({"basis": basis, "rounds": r, "shots": shots, "distance": d}))
        dem = noisy.detector_error_model(decompose_errors=True)
        for name in PRIOR_DIRS:
            p = e / "decoding_results" / name
            p.mkdir(parents=True)
            (p / "error_model.dem").write_text(str(dem))
            write_b8(p / "obs_flips_predicted.b8", obs)   # a perfect "decoder", enough to test reading
    return top
