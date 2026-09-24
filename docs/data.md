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

The first script saves the archive's own README to `docs/archive_README.txt` and writes `docs/data_inventory.md` and `docs/data_inventory.json`. The second checks the logical-gap precondition and tabulates Google's own decoder results, writing `docs/data_checks.md`.

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

## Implications for the plan (to discuss; not yet decided)

- **r=13 is in-sample for the RL prior.** Any result that uses the RL-optimized DEM should leave r=13 out of test and calibration. Libra is also absent at r=1 and r=13.
- **The "fitted DEM" in spec 8.2 already exists.** The shipped RL-optimized DEM is a hardware-fitted prior, so fitting our own becomes optional.
- **Acquisition-order drift (spec 8.7) can't be supported from the data as released,** because nothing ties shot order to time. The drift axes that remain are patch, basis, round count and simulation → hardware.
- **Wording:** the spec and report say "rotated surface code". This is the XZZX variant of the rotated layout, so the text should say so.
- **M1 targets:** `docs/data_checks.md` tabulates Google's error fractions per pathway. Our loader plus PyMatching with the SI1000 DEM should land close to `correlated_matching_decoder_with_si1000_prior`. With correlations off, expect it to be slightly worse. With `enable_correlations=True`, it should be closer still.

## Decisions (record before any analysis)

| Decision | Value | Date |
|---|---|---|
| Round counts used, per distance | | |
| Split shares (Train / Calibrate / Test) | 40 / 30 / 30 (provisional) | |
| Split seed | | |
