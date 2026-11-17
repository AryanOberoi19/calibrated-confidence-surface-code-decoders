# Preregistration: confirmatory run on the held-out r = 13 experiments

arXiv v1 plan, task A7. Written 17 November 2026, before any r = 13 label has been combined with an SI1000 score,
a calibrator, a threshold or a certificate. The commit that adds this file is tagged `v0.2.0`; the tag, once
pushed to the public remote, is the timestamp. `scripts/p6_confirmatory_r13.py --run` writes the tag's commit hash
into its results and refuses to run without the tag or to run twice.

## Why

Every Test split of the 392 experiments at r != 13 was examined many times while the pipeline was designed. The
28 experiments at r = 13 were held out from M2 onwards (`splits.py` gives them the role `sanity`). Under the SI1000
prior they are clean for this purpose: the RL prior was fitted on r = 13 data, so it is excluded.

## Prior uses of the r = 13 data (complete as far as the repository shows)

- M1 (`scripts/m1_baselines.py`): logical error rate of each decoder and prior over all 50,000 shots (decoder
  sanity check). This includes the error rate of the MWPM decoder whose gap is the score below.
- M2 (`scripts/m2_soft_outputs.py`): MWPM-gap scores computed and cached under both priors; no labels combined.
- A2 (`scripts/p1_shot_order.py`, 17 November 2026): order statistics of the detection events and of the MWPM
  error indicator under the RL prior (block heterogeneity, trend, autocorrelation, runs). No score, calibrator,
  threshold or certificate.
- Dry run (17 November 2026, while writing this file): the confirmatory script was run unchanged on the r = 10
  experiments (`--dry-run-rounds 10`, already examined in M4 and M5) to check that it works. On r = 10 the
  analogues of endpoints 1 and 3 came out confirmed and not confirmed respectively, and endpoint 2 not confirmed
  (at alpha = 0.1 only 3 of 28 naive_raw selections significantly exceed alpha). The endpoints and predictions
  below are those of the plan (v1.0 and v1.1, written before the dry run) and were not changed after it.

## Data, split and pipeline (frozen)

- Experiments: the 28 at r = 13 (d = 3: 9 patches, d = 5: 4, d = 7: 1; bases X and Z).
- Score: `mwpm_gap` under the SI1000 prior, from the existing cache.
- Split: `qeccal.data.splits_extra.confirmatory_indices`, seed 20261117, 40 / 30 / 30 by a seeded permutation of
  shot indices, built like `splits.role_codes`. Manifest: `splits/splits_r13_v1.json` (SHA-256 of the role codes per
  experiment).
- Thresholds: `KEEP_GRID` (5% steps) from Train; calibrators (raw, Platt) fitted on Train; delta = 0.05.
- Rules (C2): `ltt`, `plugin`, `naive_raw`, `naive_platt` (as M4) and `bonferroni`, `sgr`, `ltt_hb` (as A5), at M4's
  alphas {1e-3, 3e-3, 1e-2, 3e-2, 1e-1}.
- Shift (C4): Learn-then-Test certificates from each experiment's Train and Calibrate, moved to every other patch
  of the same distance and basis (ordered pairs: 144 at d = 3, 24 at d = 5), transfers `threshold` (primary) and
  `keep_fraction`, at M5's alphas {3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}; the same-experiment rows are the
  reference. A6 statistics: `pair_p_value`, BH at 5% within (distance, transfer, axis, alpha), 90% cluster-bootstrap
  interval with 2,000 resamples (two-way over source and target patch for the patch axis, one-way for same), seeds
  [20261117, 6, distance, transfer, axis, alpha index].
- No alpha subset is chosen; every cell is reported.

## Endpoints and predictions

1. C2, Learn-then-Test. At every alpha at which at least one experiment is certified, at most 2 certified
   experiments have a Test error among kept shots whose one-sided 95% Clopper-Pearson lower bound is above alpha.
   Prediction: confirmed.
2. C2, trusting the raw model. At every alpha at which naive_raw selects a threshold in at least one experiment,
   more than half of those experiments significantly exceed alpha (same criterion). Prediction: confirmed.
3. C4, patch transfer. With threshold transfer at alpha = 0.1, the lower end of the 90% cluster-bootstrap interval
   of the share of certified patch pairs failing (BH 5%) is above the same-experiment share; separately for d = 3
   and d = 5. Prediction: confirmed at d = 3; d = 5 (24 pairs from 4 patches) is likely underpowered and a "not
   confirmed" there is expected to be uninformative.

Everything else in the output (other rules, other alphas, keep-fraction transfer, d = 7 same rows) is descriptive.

## Procedure

```bash
git push -u origin preprint && git push origin v0.2.0      # makes the timestamp public
python scripts/p6_confirmatory_r13.py --run                  # once
```

Outputs: `results/p6_confirmatory_r13.csv`, `results/p6_confirmatory_r13_pairs.csv`, `results/p6_confirmatory_r13.md`.
The results are committed as they come out and not rerun with changed settings. A prediction that fails is reported
in the paper's confirmatory subsection, and the corresponding claim is weakened there.
