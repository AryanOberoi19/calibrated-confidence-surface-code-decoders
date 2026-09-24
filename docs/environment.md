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
