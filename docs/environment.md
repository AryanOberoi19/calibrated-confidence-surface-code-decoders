# Environment

Target machine: MacBook Pro, Apple M4 Pro (arm64), macOS. Everything is free and runs locally.

## One-time setup

1. Install Python 3.12 (python.org installer, or `brew install python@3.12`).
2. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                                  # makes `import qeccal` work
pip install --pre -r requirements-optional.txt    # optional: Tesseract decoder
python scripts/check_env.py                       # should end with "ALL GOOD"
python -m pytest -q                               # should pass
```

`check_env.py` prints every package version, whether PyTorch sees the Apple GPU (`torch MPS available: True` is expected on the M4 Pro), and runs a small decode whose logical error should be about 0.011.

## Known issues

| Issue | Workaround |
|---|---|
| `beliefmatching` 0.2.0 requires `numpy<=2.2.6` | numpy is pinned to 2.2.6 in `requirements.txt` |
| `tesseract-decoder` publishes only pre-release builds | install with `--pre`, then pin the exact build you got in `requirements-optional.txt` |
| `sinter` scripts crash on macOS without a main guard | every script that calls `sinter.collect` keeps its work under `if __name__ == "__main__":` |

Record any further install problems here rather than switching machines.

## Version control

```bash
git init
git add .
git commit -m "Initial skeleton"
```

`data/raw/` is git-ignored: the 5.7 GB archive never goes into git.

## Second environment: Linux VM (17 November 2026)

Used for the arXiv v1 work: a Linux aarch64 VM on this Mac (4 cores, 3 GB RAM) with CPython 3.12.15 installed by `uv`, and the packages of `requirements.txt` at their pinned versions. `pymatching` 2.4.0 has no Linux aarch64 wheel and builds from source.

`torch` is not installed there: download.pytorch.org is not reachable from the VM, and the PyPI aarch64 wheel pulls about twenty CUDA packages that do not fit on the VM's disk. So `tests/test_learned.py` is skipped (35 passed, 1 skipped, instead of 37 passed), and `scripts/m3_learned.py` cannot run there. The arXiv v1 plan does not retrain the learned decoder; its cached scores are used as they are. Run the full suite on macOS with the `.venv` above to get all 37.

Regression against the CSVs committed from macOS (tag `v0.1.0`), rerun on 17 November 2026:

| File | Result |
|---|---|
| `results/m4_guarantees.csv` | byte-identical |
| `results/m5_drift.csv` | byte-identical |
| `results/m4_calibration.csv` | keys and counts identical; 1,157 of 4,080 rows differ in a float column (859 Platt, 253 temperature, 30 raw, 15 isotonic). Largest relative differences: mean_q 3.2e-7, ece 4.3e-6, brier 6.8e-8, nll 7.5e-8. The 4-digit `params` string differs in 11 rows. |
| `results/m5_calibration.csv` | keys identical; 3,052 of 8,032 rows differ (2,987 Platt, 65 isotonic). Largest absolute differences: ece 6.7e-7, nll 4.5e-7. |

Cause: floating-point differences between the macOS arm64 and Linux aarch64 builds, mostly in the iterative Platt and temperature fits and in summation order. No certificate, keep fraction, count or exceedance flag changes. The committed macOS files are kept.
