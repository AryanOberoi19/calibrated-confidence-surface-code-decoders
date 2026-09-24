# Web app

`app/streamlit_app.py` has five views:

- **Overview**: headline numbers, computed live from `results/`.
- **Shot explorer**: 300 Test shots from each of 8 experiments. Each shot is decoded live: the logical gap, raw and Platt-calibrated P(wrong), Google's five predictions, and a 3D space-time plot of the detection events and the MWPM matching.
- **Calibration**: Test reliability diagrams and ECE, Brier and NLL per calibrator, by distance, round count and prior. At r=1 it also covers the exact posterior.
- **Postselection**: pick an experiment, α and δ. The view runs Learn-then-Test live on the Calibrate split and compares it on the Test split with the plug-in, raw-q and Platt-q rules.
- **Drift**: how often certified thresholds miss α after a shift, and a patch-to-patch matrix.

The app never needs the raw archive. It reads `app/data/` (6.7 MB, committed) and `results/*.csv`.

## Run locally

```bash
pip install -r app/requirements.txt      # or use the project .venv, which already has them
streamlit run app/streamlit_app.py
```

## Rebuild the app data

This needs the extracted archive and the M2 cache (`results/cache/soft/`):

```bash
python scripts/build_app_data.py
```

## Deploy on Streamlit Community Cloud (free)

1. Commit and push `app/` (including `app/data/`) and `results/`.
2. Go to https://share.streamlit.io and sign in with the GitHub account that owns the repository.
3. Click **Create app**, then deploy from GitHub:
   - Repository: `AryanOberoi19/calibrated-confidence-surface-code-decoders`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - Optional: a custom subdomain for the URL
4. Under **Advanced settings**, choose Python 3.12, the version the project was developed on.
5. Deploy. Community Cloud looks for a requirements file next to the entrypoint first, so it installs `app/requirements.txt` (Streamlit, Plotly, pandas, NumPy, SciPy, Stim, PyMatching). The root `requirements.txt`, which includes torch, is not used.

A push to `main` redeploys the app automatically. If the repository is private, Community Cloud asks for permission to read private repositories.
