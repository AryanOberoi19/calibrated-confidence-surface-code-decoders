# Data

## Source

Google Quantum AI, *Data for "Quantum error correction below the surface code threshold"*, Zenodo record [10.5281/zenodo.13273331](https://doi.org/10.5281/zenodo.13273331), v1.0.0, 26 Aug 2024, licence CC BY 4.0. Cite the dataset and the paper (arXiv:2408.13687; Nature 638, 920–926, 2025) in the report.

## Download

Record page: https://zenodo.org/records/13273331

- File: `google_105Q_surface_code_d3_d5_d7.zip`, 5,716,907,033 bytes. The other three archives are not needed.
- MD5 `21fa6ad35b395d838ebcdbc92e364a12` (matches the record page; verified on the full download, 24 Sep 2026).
- Put the zip in `data/raw/` (git-ignored). Do not unzip it; the scripts read it directly.
- Zenodo accepts HTTP range requests, and a single connection is slow (about 0.25-0.4 MB/s). If the browser crawls, download the file in parallel byte ranges with `curl -r` and join the pieces. Then check the MD5.

## Verify and inventory

```bash
python scripts/inspect_archive.py data/raw/google_105Q_surface_code_d3_d5_d7.zip --md5
python scripts/data_checks.py     data/raw/google_105Q_surface_code_d3_d5_d7.zip
```

Both scripts read the zip. After them, extract the zip into `data/raw/`; everything else reads the extracted folder, and the zip can be deleted. The first script saves the archive's own README to `docs/archive_README.txt` and writes `docs/data_inventory.md` and `docs/data_inventory.json`. The second checks the logical-gap precondition and tabulates Google's own decoder results, writing `docs/data_checks.md`.

## Confirmed facts (from the inventory, the archive README and data_checks)

| Question (spec section 14) | Answer |
|---|---|
| Directory layout | `google_105Q_surface_code_d3_d5_d7/d<d>_at_q<r>_<c>/<X or Z>/r<NN>/`, where the round count is zero-padded (`r01`). Each directory has a `decoding_results/<pathway>/` subfolder. There are 420 experiment directories, plus a `README/` folder holding README.md and two PNGs. |
| File holding the ground-truth logical flips | `obs_flips_actual.b8`: 1 observable, 1 byte per shot, bit 0. |
| Noisy circuit or DEM included? | Yes. Every directory has `circuit_noisy_si1000.stim`. Every pathway folder has an `error_model.dem` built with either the SI1000 prior or the RL-optimized prior (Sivak et al., arXiv:2406.02700). The RL prior was "optimized jointly for all distance-3 and distance-5 patches using the 13-cycle calibration data" (archive README). |
| Decoder predictions included? | Yes, hard predictions only (`obs_flips_predicted.b8`, no soft outputs), for 5 pathways: correlated matching and Harmony, each with the SI1000 and RL priors (all 420 directories), and Libra with the RL prior (364 directories; missing at r=1 and r=13). |
| Shots per directory | 50,000 everywhere. File sizes and `metadata.json` agree in all 420 directories. |
| Exact list of round counts | 1, 10, 13, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250 (15 values). |
| Distances, patches and bases present | d=3: 9 patches (q10_7, q2_7, q4_5, q4_9, q6_11, q6_3, q6_7, q8_5, q8_9). d=5: 4 patches (q4_7, q6_5, q6_9, q8_7). d=7: 1 patch (q6_7). All have bases X and Z. |
| Code variant | XZZX surface code. The README says the "X" and "Z" basis labels are "an arbitrary designation" for this code. Also present: `measurements.b8` and `sweep_bits.b8` (raw data; `stim m2d` rebuilds the detection events from them). |
| Is shot order within a file acquisition order? (metadata / README) | Not established. `metadata.json` has only basis, rounds, shots, distance and qubit coordinates, with no timestamps, and the README says nothing about shot order. |
| Does the logical-gap precondition hold on these circuits? | Yes. There are 0 violations across the 420 SI1000 DEMs built from the noisy circuits and the 840 distinct shipped DEMs (see `docs/data_checks.md`). |

## Decisions (24 September 2026)

| Decision | Value |
|---|---|
| r=13 | Held out. Every r=13 shot has the role `sanity`: decoder sanity checks only, never Calibrate or Test, because the RL-optimized prior was fitted on the 13-cycle data. Libra is also absent at r=1 and r=13. |
| Fitted DEM | Google's shipped RL-optimized prior (`Experiment.dem("rl")`). We do not fit our own; the p_ij estimator is an optional stretch goal. Prior DEM: the shipped SI1000 DEM (`Experiment.dem("si1000")`). |
| Split shares (Train / Calibrate / Test) | 40 / 30 / 30 per experiment (20,000 / 15,000 / 15,000 shots) |
| Split method and seed | Seeded permutation per experiment, seed 20260924 combined with a hash of the experiment key (`qeccal.data.splits`). Manifest with per-experiment checksums: `splits/splits_v1.json` (`python scripts/make_splits.py`). |
| Round counts | All 15 are loaded. r=13 is held out as above, and r=1 is excluded from per-cycle fits. Which round counts each analysis uses is stated in that analysis's script. |
| Shift axes | Patch, basis and simulation → hardware. Acquisition order is dropped: no timestamps, and the README does not say that file order is time order. |
| Wording | The code is the XZZX variant of the rotated surface code. |

## Observations from M1 (`results/m1_baselines.csv`)

- Logical error per shot does not always rise with round count by more than shot noise allows. For example, plain MWPM with the RL prior on `d7_at_q6_7/Z` gives 29.9% at r=90 and 29.5% at r=110 (standard error about 0.2%), and the same flat or falling step from r=90 to r=110 appears in several d=5 Z-basis patches. Treat each round count as a separately acquired experiment, and read per-cycle fits as approximate.
- Our correlated PyMatching, run with Google's shipped DEM, lands 2-7% above Google's correlated matching in per-cycle error, and agrees with it on 97.6-99.3% of shots at r=10 (`results/m1_summary.md`).

## Loading

```python
from qeccal.data import list_experiments, get_experiment, split_indices

e = get_experiment("d5_at_q6_5/X/r50")
det = e.detection_events()            # (50000, 1200) bool; packed=True for PyMatching's bit_packed_shots
obs = e.observable_flips()            # (50000,) bool, the ground truth
dem = e.dem("rl")                     # or "si1000", or "si1000_circuit"
g = e.google_predictions("libra_decoder_with_rl_optimized_prior")   # None where absent
idx = split_indices(e)                # {"train": ..., "calibrate": ..., "test": ...}
```

Set `QECCAL_DATA_ROOT` if the extracted archive is not at `data/raw/google_105Q_surface_code_d3_d5_d7`.
