# A7: confirmatory run on r = 13 (preregistered at tag v0.2.0 = 2afaa7a)

28 experiments, mwpm_gap/si1000, split seed 20261117. Endpoints as stated in docs/prereg_r13.md.

## Endpoints 1 and 2 (C2)

| alpha | rule | selected | exceeds | significantly exceeds |
|---|---|---|---|---|
| 0.001 | ltt | 1/28 | 0 | 0 |
| 0.001 | plugin | 23/28 | 8 | 1 |
| 0.001 | naive_raw | 28/28 | 28 | 26 |
| 0.001 | naive_platt | 27/28 | 16 | 4 |
| 0.001 | bonferroni | 0/28 | 0 | 0 |
| 0.001 | sgr | 0/28 | 0 | 0 |
| 0.001 | ltt_hb | 1/28 | 0 | 0 |
| 0.003 | ltt | 18/28 | 0 | 0 |
| 0.003 | plugin | 28/28 | 5 | 0 |
| 0.003 | naive_raw | 28/28 | 28 | 28 |
| 0.003 | naive_platt | 28/28 | 5 | 1 |
| 0.003 | bonferroni | 11/28 | 0 | 0 |
| 0.003 | sgr | 16/28 | 0 | 0 |
| 0.003 | ltt_hb | 14/28 | 0 | 0 |
| 0.01 | ltt | 28/28 | 0 | 0 |
| 0.01 | plugin | 28/28 | 2 | 1 |
| 0.01 | naive_raw | 28/28 | 28 | 28 |
| 0.01 | naive_platt | 28/28 | 2 | 0 |
| 0.01 | bonferroni | 28/28 | 0 | 0 |
| 0.01 | sgr | 28/28 | 0 | 0 |
| 0.01 | ltt_hb | 28/28 | 0 | 0 |
| 0.03 | ltt | 28/28 | 0 | 0 |
| 0.03 | plugin | 28/28 | 2 | 0 |
| 0.03 | naive_raw | 28/28 | 28 | 28 |
| 0.03 | naive_platt | 28/28 | 4 | 2 |
| 0.03 | bonferroni | 28/28 | 0 | 0 |
| 0.03 | sgr | 28/28 | 0 | 0 |
| 0.03 | ltt_hb | 28/28 | 0 | 0 |
| 0.1 | ltt | 28/28 | 0 | 0 |
| 0.1 | plugin | 28/28 | 0 | 0 |
| 0.1 | naive_raw | 28/28 | 10 | 9 |
| 0.1 | naive_platt | 28/28 | 2 | 0 |
| 0.1 | bonferroni | 28/28 | 0 | 0 |
| 0.1 | sgr | 28/28 | 0 | 0 |
| 0.1 | ltt_hb | 28/28 | 0 | 0 |

Endpoint 1 (ltt: at most 2 certified experiments significantly exceed alpha, at every alpha with a certificate): CONFIRMED.
Endpoint 2 (naive_raw: more than half of the selected experiments significantly exceed alpha, at every alpha with a selection): NOT CONFIRMED.

## Endpoint 3 (C4): patch transfer

| distance | transfer | axis | alpha | certified pairs | failing (BH 5%) | share % [90% CI] |
|---|---|---|---|---|---|---|
| 3 | keep_fraction | patch | 0.003 | 72 | 11 | 15.3 [0.0, 34.5] |
| 3 | keep_fraction | same | 0.003 | 9 | 0 | 0.0 [0.0, 0.0] |
| 3 | keep_fraction | patch | 0.01 | 144 | 40 | 27.8 [8.6, 49.3] |
| 3 | keep_fraction | same | 0.01 | 18 | 0 | 0.0 [0.0, 0.0] |
| 3 | keep_fraction | patch | 0.03 | 144 | 50 | 34.7 [15.0, 56.4] |
| 3 | keep_fraction | same | 0.03 | 18 | 0 | 0.0 [0.0, 0.0] |
| 3 | keep_fraction | patch | 0.1 | 144 | 38 | 26.4 [8.1, 47.1] |
| 3 | keep_fraction | same | 0.1 | 18 | 0 | 0.0 [0.0, 0.0] |
| 3 | threshold | patch | 0.003 | 72 | 0 | 0.0 [0.0, 0.0] |
| 3 | threshold | same | 0.003 | 9 | 0 | 0.0 [0.0, 0.0] |
| 3 | threshold | patch | 0.01 | 144 | 2 | 1.4 [0.0, 5.2] |
| 3 | threshold | same | 0.01 | 18 | 0 | 0.0 [0.0, 0.0] |
| 3 | threshold | patch | 0.03 | 144 | 27 | 18.8 [6.3, 35.1] |
| 3 | threshold | same | 0.03 | 18 | 0 | 0.0 [0.0, 0.0] |
| 3 | threshold | patch | 0.1 | 144 | 37 | 25.7 [7.3, 46.6] |
| 3 | threshold | same | 0.1 | 18 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | patch | 0.003 | 21 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | same | 0.003 | 7 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | patch | 0.01 | 24 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | same | 0.01 | 8 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | patch | 0.03 | 24 | 6 | 25.0 [0.0, 60.0] |
| 5 | keep_fraction | same | 0.03 | 8 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | patch | 0.1 | 24 | 0 | 0.0 [0.0, 0.0] |
| 5 | keep_fraction | same | 0.1 | 8 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | patch | 0.003 | 21 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | same | 0.003 | 7 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | patch | 0.01 | 24 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | same | 0.01 | 8 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | patch | 0.03 | 24 | 1 | 4.2 [0.0, 16.7] |
| 5 | threshold | same | 0.03 | 8 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | patch | 0.1 | 24 | 0 | 0.0 [0.0, 0.0] |
| 5 | threshold | same | 0.1 | 8 | 0 | 0.0 [0.0, 0.0] |
| 7 | keep_fraction | same | 0.001 | 1 | 0 | 0.0 [0.0, 0.0] |
| 7 | keep_fraction | same | 0.003 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | keep_fraction | same | 0.01 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | keep_fraction | same | 0.03 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | keep_fraction | same | 0.1 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | threshold | same | 0.001 | 1 | 0 | 0.0 [0.0, 0.0] |
| 7 | threshold | same | 0.003 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | threshold | same | 0.01 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | threshold | same | 0.03 | 2 | 0 | 0.0 [0.0, 0.0] |
| 7 | threshold | same | 0.1 | 2 | 0 | 0.0 [0.0, 0.0] |

Endpoint 3, d = 3, threshold transfer, alpha = 0.1 (patch interval's lower end above the same-experiment share): CONFIRMED.
Endpoint 3, d = 5, threshold transfer, alpha = 0.1 (patch interval's lower end above the same-experiment share): NOT CONFIRMED.
