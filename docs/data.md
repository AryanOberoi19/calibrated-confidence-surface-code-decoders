# Data

## Source

Google Quantum AI, *Data for "Quantum error correction below the surface code threshold"*, Zenodo record [10.5281/zenodo.13273331](https://doi.org/10.5281/zenodo.13273331), v1.0.0, 26 Aug 2024, licence CC BY 4.0. Cite the dataset and the paper (arXiv:2408.13687; Nature 638, 920–926, 2025) in the report.

## Download (manual)

The download is done by hand in a browser: https://zenodo.org/records/13273331

- File: `google_105Q_surface_code_d3_d5_d7.zip` (5.7 GB). The other three archives are not needed.
- MD5 listed on the record page: `21fa6ad35b395d838ebcdbc92e364a12` (check it against the page itself).
- Put the zip in `data/raw/` (git-ignored). Do not unzip it; the scripts read it directly.

## Verify and inventory

```bash
python scripts/inspect_archive.py data/raw/google_105Q_surface_code_d3_d5_d7.zip --md5
```

This checks the checksum, saves the archive's own README to `docs/archive_README.txt`, and writes `docs/data_inventory.md` and `docs/data_inventory.json`.

## Confirmed facts (fill in from the inventory and the archive README)

| Question (spec section 14) | Answer |
|---|---|
| Directory layout | |
| File holding the ground-truth logical flips | |
| Noisy circuit or DEM included? | |
| Decoder predictions included? | |
| Shots per directory | |
| Exact list of round counts | |
| Distances, patches and bases present | |
| Is shot order within a file acquisition order? (metadata / README) | |
| Does the logical-gap precondition hold on these circuits? | |

## Decisions (record before any analysis)

| Decision | Value | Date |
|---|---|---|
| Round counts used, per distance | | |
| Split shares (Train / Calibrate / Test) | 40 / 30 / 30 (provisional) | |
| Split seed | | |
