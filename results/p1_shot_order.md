# A2: shot order and independence

420 experiments, 50,000 shots each, in file order. Method and classification rules: `scripts/p1_shot_order.py` (fixed before the data were read). x1 = detection events per shot; x2 = MWPM (RL prior) decoder-error indicator.

**Case 2.** Block heterogeneity or trend rejects widely: file order carries drift. A3 is read as a temporal test. A test on the decoder-error indicator (x2) rejects more often than chance, so the structure reaches the error rate.

## Rejections per test

| test | p<0.05 | binom_p | bh | more_than_chance | widely |
|---|---|---|---|---|---|
| bh_x1 | 401 | 0 | 400 | yes | yes |
| bh_x2 | 53 | 9.39e-10 | 19 | yes | no |
| trend_x1 | 315 | 1.17e-311 | 309 | yes | yes |
| trend_x2 | 49 | 4.85e-08 | 13 | yes | no |
| lb_x1 | 306 | 4.38e-296 | 301 | yes | yes |
| lb_x2 | 22 | 0.442 | 4 | no | no |
| runs_x2 | 32 | 0.013 | 4 | no | no |

Expected under no structure: 21 of 420 below 0.05; BH rejections near 0.

## Effect sizes (distribution over experiments)

ratio = variance of 50 block means / its expectation under random order (1 = none); edge = relative change of the mean, last 10,000 vs first 10,000 shots; rho = Spearman correlation with shot index; ac1 = lag-1 autocorrelation.

|  | 5% | 50% | 95% | min | max |
|---|---|---|---|---|---|
| ratio_x1 | 1.4 | 6.52 | 67.1 | 0.789 | 213 |
| ratio_x2 | 0.734 | 1.03 | 1.61 | 0.575 | 11.5 |
| edge_x1 | -0.0702 | -0.000267 | 0.0591 | -0.314 | 0.403 |
| edge_x2 | -0.0933 | 0 | 0.053 | -0.75 | 0.449 |
| rho_x1 | -0.096 | -0.00286 | 0.0668 | -0.296 | 0.291 |
| rho_x2 | -0.0102 | 0.000252 | 0.00814 | -0.0566 | 0.0554 |
| ac1_x1 | -0.00182 | 0.0166 | 0.192 | -0.00705 | 0.495 |
| ac1_x2 | -0.00642 | 0.000328 | 0.0091 | -0.0125 | 0.028 |

## Experiments with p < 0.05, by distance

| distance | bh_x1 | bh_x2 | trend_x1 | trend_x2 | lb_x1 | lb_x2 | runs_x2 | experiments |
|---|---|---|---|---|---|---|---|---|
| 3 | 253 | 33 | 196 | 31 | 176 | 14 | 22 | 270 |
| 5 | 118 | 15 | 95 | 13 | 104 | 8 | 7 | 120 |
| 7 | 30 | 5 | 24 | 5 | 26 | 0 | 3 | 30 |

## By round count

Descriptive (added after the first run; no test or rule changed): experiments with p < 0.05 and median effect sizes per round count. x2_rate is the MWPM error rate; near 0.5 a drift in the syndrome cannot move it.

| rounds | experiments | bh_x1 p<0.05 | ratio_x1 median | ac1_x1 median | ac1_x1 max | bh_x2 p<0.05 | trend_x2 p<0.05 | ratio_x2 median | x2_rate median |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 28 | 24 | 2.81 | 0.00118 | 0.0189 | 1 | 4 | 1.02 | 0.00397 |
| 10 | 28 | 27 | 3.03 | 0.00429 | 0.027 | 3 | 6 | 1.12 | 0.067 |
| 13 | 28 | 25 | 2.29 | 0.00354 | 0.0261 | 0 | 2 | 1.02 | 0.0878 |
| 30 | 28 | 25 | 9.49 | 0.0176 | 0.495 | 11 | 7 | 1.25 | 0.198 |
| 50 | 28 | 28 | 5.3 | 0.00855 | 0.0728 | 6 | 5 | 1.1 | 0.273 |
| 70 | 28 | 27 | 7.97 | 0.0245 | 0.244 | 5 | 5 | 1.05 | 0.333 |
| 90 | 28 | 28 | 12.3 | 0.0361 | 0.324 | 7 | 5 | 1.15 | 0.397 |
| 110 | 28 | 28 | 4.9 | 0.0178 | 0.217 | 4 | 2 | 0.975 | 0.412 |
| 130 | 28 | 27 | 6.65 | 0.0219 | 0.123 | 4 | 2 | 1.1 | 0.44 |
| 150 | 28 | 27 | 9.65 | 0.0241 | 0.249 | 4 | 1 | 1.06 | 0.457 |
| 170 | 28 | 28 | 9.58 | 0.0358 | 0.362 | 3 | 3 | 0.972 | 0.466 |
| 190 | 28 | 27 | 10.1 | 0.0383 | 0.256 | 2 | 1 | 0.941 | 0.477 |
| 210 | 28 | 27 | 14 | 0.0459 | 0.162 | 2 | 2 | 1 | 0.48 |
| 230 | 28 | 27 | 21.2 | 0.0835 | 0.385 | 0 | 3 | 0.982 | 0.486 |
| 250 | 28 | 26 | 16.2 | 0.0244 | 0.237 | 1 | 1 | 1.03 | 0.489 |
