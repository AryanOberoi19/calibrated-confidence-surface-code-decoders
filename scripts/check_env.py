"""Check that the project environment is complete and working.

Usage:  python scripts/check_env.py

Prints the version of every required package, whether PyTorch can use the
Apple GPU (MPS), and runs a tiny Stim + PyMatching decode as a smoke test.
Exit code is non-zero if anything required is missing or broken.
"""
import importlib
import platform
import sys

REQUIRED = ["stim", "pymatching", "sinter", "beliefmatching", "ldpc",
            "numpy", "scipy", "sklearn", "matplotlib", "torch"]
OPTIONAL = ["tesseract_decoder"]


def version_of(name):
    mod = importlib.import_module(name)
    return getattr(mod, "__version__", "installed")


def main():
    ok = True
    print(f"python   {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
    if sys.version_info < (3, 11):
        print("  !! Python 3.11+ required (3.12 recommended)")
        ok = False

    for name in REQUIRED + OPTIONAL:
        try:
            print(f"{name:<18} {version_of(name)}")
        except Exception as exc:  # noqa: BLE001 - report any import failure
            tag = "optional, skipped" if name in OPTIONAL else "!! MISSING"
            print(f"{name:<18} {tag}: {exc.__class__.__name__}: {exc}")
            ok = ok and name in OPTIONAL

    try:
        import torch
        mps = torch.backends.mps.is_available()
        print(f"torch MPS available: {mps}")
        if not mps and platform.system() == "Darwin":
            print("  !! MPS not available: the learned decoder will fall back to CPU")
    except Exception:  # noqa: BLE001
        pass

    try:
        import numpy as np
        import pymatching
        import stim
        c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3,
                                   after_clifford_depolarization=0.004,
                                   before_round_data_depolarization=0.004,
                                   before_measure_flip_probability=0.004,
                                   after_reset_flip_probability=0.004)
        det, obs = c.compile_detector_sampler(seed=0).sample(50_000, separate_observables=True)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        ler = np.mean(m.decode_batch(det)[:, 0] != obs[:, 0])
        print(f"smoke test: d=3 r=3 p=0.4% MWPM logical error = {ler:.4f} (expect about 0.011)")
        if not 0.005 < ler < 0.02:
            print("  !! smoke-test logical error outside the expected range")
            ok = False
    except Exception as exc:  # noqa: BLE001
        print(f"smoke test FAILED: {exc!r}")
        ok = False

    print("\nALL GOOD" if ok else "\nPROBLEMS FOUND (see '!!' lines above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
