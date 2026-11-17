# r2: per-clip results, concentration, clip bootstrap

## Per clip (held-out): right / wrong / lost

| clip | scored | strict | balanced | recall | margin_0.50 | plugin | split | mutual | cuts_only |
|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 16 | 4 / 0 / 12 | 10 / 0 / 6 | 15 / 0 / 1 | 10 / 0 / 6 | 1 / 2 / 13 | 0 / 1 / 15 | 3 / 1 / 12 | 0 / 0 / 16 |
| SNGS-163 | 11 | 1 / 0 / 10 | 5 / 0 / 6 | 8 / 0 / 3 | 5 / 0 / 6 | 1 / 0 / 10 | 0 / 0 / 11 | 1 / 0 / 10 | 0 / 0 / 11 |
| SNGS-106 | 7 | 3 / 0 / 4 | 6 / 0 / 1 | 6 / 0 / 1 | 5 / 0 / 2 | 4 / 2 / 1 | 0 / 0 / 7 | 2 / 0 / 5 | 0 / 0 / 7 |
| SNGS-097 | 6 | 3 / 0 / 3 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 | 3 / 0 / 3 | 0 / 0 / 6 | 1 / 0 / 5 | 0 / 0 / 6 |
| SNGS-067 | 10 | 2 / 0 / 8 | 8 / 0 / 2 | 9 / 0 / 1 | 8 / 0 / 2 | 1 / 1 / 8 | 0 / 0 / 10 | 0 / 0 / 10 | 0 / 0 / 10 |
| SNGS-100 | 4 | 2 / 0 / 2 | 3 / 0 / 1 | 3 / 0 / 1 | 3 / 0 / 1 | 1 / 0 / 3 | 0 / 0 / 4 | 3 / 0 / 1 | 0 / 0 / 4 |
| SNGS-066 | 11 | 2 / 0 / 9 | 6 / 1 / 4 | 6 / 1 / 4 | 5 / 0 / 6 | 1 / 3 / 7 | 0 / 1 / 10 | 0 / 1 / 10 | 0 / 0 / 11 |
| SNGS-161 | 14 | 4 / 0 / 10 | 13 / 0 / 1 | 14 / 0 / 0 | 13 / 0 / 1 | 6 / 0 / 8 | 0 / 0 / 14 | 4 / 0 / 10 | 0 / 0 / 14 |
| SNGS-169 | 15 | 2 / 0 / 13 | 12 / 0 / 3 | 15 / 0 / 0 | 12 / 0 / 3 | 4 / 1 / 10 | 0 / 0 / 15 | 4 / 0 / 11 | 0 / 0 / 15 |
| SNGS-162 | 18 | 6 / 4 / 8 | 7 / 6 / 5 | 7 / 6 / 5 | 7 / 4 / 7 | 2 / 5 / 11 | 0 / 2 / 16 | 1 / 4 / 13 | 0 / 0 / 18 |
| SNGS-151 | 9 | 0 / 0 / 9 | 4 / 0 / 5 | 7 / 0 / 2 | 4 / 0 / 5 | 2 / 0 / 7 | 0 / 0 / 9 | 1 / 0 / 8 | 0 / 0 / 9 |
| SNGS-102 | 12 | 2 / 0 / 10 | 10 / 0 / 2 | 10 / 0 / 2 | 10 / 0 / 2 | 2 / 0 / 10 | 0 / 0 / 12 | 4 / 0 / 8 | 0 / 0 / 12 |
| SNGS-112 | 4 | 2 / 0 / 2 | 3 / 0 / 1 | 4 / 0 / 0 | 3 / 0 / 1 | 2 / 0 / 2 | 0 / 0 / 4 | 2 / 0 / 2 | 0 / 0 / 4 |

## Is the gain carried by one or two clips?

**strict** (33 right, 4 wrong of 137):
- the clip with the most right returns (SNGS-162) holds 18% of them, the top two 30% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.0, median 0.25, max 0.5; 3 of 13 clips at or above one half;
- against plugin: 5 clips better, 4 worse, 4 equal in right returns; wrong higher in 0 clips, lower in 6; gain 3 right returns, 233% of it in ['SNGS-162', 'SNGS-107'];
- pooled right share with any one clip removed: [0.227, 0.258]; any two removed: [0.212, 0.274];
- pooled wrong share with any one clip removed: [0.0, 0.0331]; any two removed: [0.0, 0.0377];
- wrong returns: 4, of which 100% in SNGS-162.

**balanced** (93 right, 7 wrong of 137):
- the clip with the most right returns (SNGS-161) holds 14% of them, the top two 27% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.389, median 0.75, max 1.0; 10 of 13 clips at or above one half;
- against plugin: 13 clips better, 0 worse, 0 equal in right returns; wrong higher in 1 clips, lower in 5; gain 63 right returns, 27% of it in ['SNGS-107', 'SNGS-169'];
- pooled right share with any one clip removed: [0.65, 0.723]; any two removed: [0.63, 0.75];
- pooled wrong share with any one clip removed: [0.0084, 0.0579]; any two removed: [0.0, 0.066];
- wrong returns: 7, of which 86% in SNGS-162.

**recall** (110 right, 7 wrong of 137):
- the clip with the most right returns (SNGS-107) holds 14% of them, the top two 27% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.389, median 0.857, max 1.0; 12 of 13 clips at or above one half;
- against plugin: 13 clips better, 0 worse, 0 equal in right returns; wrong higher in 1 clips, lower in 5; gain 80 right returns, 31% of it in ['SNGS-107', 'SNGS-169'];
- pooled right share with any one clip removed: [0.779, 0.866]; any two removed: [0.75, 0.898];
- pooled wrong share with any one clip removed: [0.0084, 0.0579]; any two removed: [0.0, 0.066];
- wrong returns: 7, of which 86% in SNGS-162.

**margin_0.50** (91 right, 4 wrong of 137):
- the clip with the most right returns (SNGS-161) holds 14% of them, the top two 28% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.389, median 0.75, max 1.0; 9 of 13 clips at or above one half;
- against plugin: 13 clips better, 0 worse, 0 equal in right returns; wrong higher in 0 clips, lower in 6; gain 61 right returns, 28% of it in ['SNGS-107', 'SNGS-169'];
- pooled right share with any one clip removed: [0.634, 0.706]; any two removed: [0.611, 0.731];
- pooled wrong share with any one clip removed: [0.0, 0.0331]; any two removed: [0.0, 0.0377];
- wrong returns: 4, of which 100% in SNGS-162.

## Clip bootstrap (2000 resamples of the 13 clips, seed 0, paired)

Shares are of the scored returns in the resample; quantiles 2.5% / 5% / 50% / 95% / 97.5%. x137 = the share times 137 (count-equivalent).

### strict

- right share [0.1769, 0.1871, 0.24, 0.2975, 0.3125] (x137: [24.2, 25.6, 32.9, 40.8, 42.8])
- wrong share [0.0, 0.0, 0.028, 0.0769, 0.0839] (x137: [0.0, 0.0, 3.8, 10.5, 11.5]); P(wrong share <= 4/137, 7/137, 14/137) = [0.575, 0.789, 0.994]
- link precision (judged links) [0.9453, 0.9497, 0.9771, 1.0, 1.0]; impure row share [0.0062, 0.0069, 0.0124, 0.0187, 0.0198]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | -0.067 / +0.019 / +0.104 | 0.6385 | -0.126 / -0.071 / -0.030 | 1.0 | 1.0 | -0.0252 / -0.0181 / -0.0114 | 1.0 |
| split | +0.177 / +0.240 / +0.312 | 1.0 | -0.032 / +0.000 / +0.035 | 0.4625 | 0.627 | -0.0198 / -0.0133 / -0.0074 | 1.0 |
| mutual | -0.040 / +0.048 / +0.144 | 0.832 | -0.034 / -0.014 / +0.000 | 0.896 | 1.0 | -0.0241 / -0.0172 / -0.0108 | 1.0 |
| cuts_only | +0.177 / +0.240 / +0.312 | 1.0 | +0.000 / +0.028 / +0.084 | 0.0 | 0.3585 | +0.0000 / +0.0028 / +0.0083 | 0.0 |

### balanced

- right share [0.5678, 0.5818, 0.6829, 0.7763, 0.7937] (x137: [77.8, 79.7, 93.6, 106.4, 108.7])
- wrong share [0.0, 0.0, 0.047, 0.1208, 0.1325] (x137: [0.0, 0.0, 6.4, 16.5, 18.2]); P(wrong share <= 4/137, 7/137, 14/137) = [0.353, 0.581, 0.908]
- link precision (judged links) [0.9381, 0.9415, 0.9691, 0.9895, 0.9928]; impure row share [0.0084, 0.0097, 0.0182, 0.0283, 0.03]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.375 / +0.461 / +0.542 | 1.0 | -0.106 / -0.052 / -0.006 | 0.977 | 0.9935 | -0.0213 / -0.0124 / -0.0025 | 0.993 |
| split | +0.568 / +0.683 / +0.794 | 1.0 | -0.020 / +0.020 / +0.081 | 0.25 | 0.3665 | -0.0156 / -0.0075 / +0.0014 | 0.9525 |
| mutual | +0.400 / +0.489 / +0.577 | 1.0 | -0.020 / +0.007 / +0.040 | 0.2805 | 0.4725 | -0.0180 / -0.0114 / -0.0056 | 1.0 |
| cuts_only | +0.568 / +0.683 / +0.794 | 1.0 | +0.000 / +0.047 / +0.133 | 0.0 | 0.1205 | +0.0016 / +0.0082 / +0.0191 | 0.0 |

### recall

- right share [0.6667, 0.6861, 0.8125, 0.9083, 0.9188] (x137: [91.3, 94.0, 111.3, 124.4, 125.9])
- wrong share [0.0, 0.0, 0.047, 0.1208, 0.1325] (x137: [0.0, 0.0, 6.4, 16.5, 18.2]); P(wrong share <= 4/137, 7/137, 14/137) = [0.353, 0.581, 0.908]
- link precision (judged links) [0.9408, 0.9447, 0.9712, 0.9905, 0.9935]; impure row share [0.0086, 0.01, 0.0184, 0.0283, 0.03]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.461 / +0.588 / +0.699 | 1.0 | -0.106 / -0.052 / -0.006 | 0.977 | 0.9935 | -0.0212 / -0.0122 / -0.0022 | 0.9915 |
| split | +0.667 / +0.812 / +0.919 | 1.0 | -0.020 / +0.020 / +0.081 | 0.25 | 0.3665 | -0.0154 / -0.0073 / +0.0016 | 0.9445 |
| mutual | +0.503 / +0.618 / +0.711 | 1.0 | -0.020 / +0.007 / +0.040 | 0.2805 | 0.4725 | -0.0179 / -0.0111 / -0.0053 | 1.0 |
| cuts_only | +0.667 / +0.812 / +0.919 | 1.0 | +0.000 / +0.047 / +0.133 | 0.0 | 0.1205 | +0.0018 / +0.0084 / +0.0192 | 0.0 |

### margin_0.50

- right share [0.5522, 0.5676, 0.6688, 0.7609, 0.7778] (x137: [75.7, 77.8, 91.6, 104.2, 106.6])
- wrong share [0.0, 0.0, 0.028, 0.0769, 0.0839] (x137: [0.0, 0.0, 3.8, 10.5, 11.5]); P(wrong share <= 4/137, 7/137, 14/137) = [0.575, 0.789, 0.994]
- link precision (judged links) [0.9472, 0.952, 0.9767, 0.9963, 1.0]; impure row share [0.0075, 0.0086, 0.0146, 0.0218, 0.0231]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.357 / +0.447 / +0.532 | 1.0 | -0.126 / -0.071 / -0.030 | 1.0 | 1.0 | -0.0237 / -0.0159 / -0.0080 | 1.0 |
| split | +0.552 / +0.669 / +0.778 | 1.0 | -0.032 / +0.000 / +0.035 | 0.4625 | 0.627 | -0.0182 / -0.0110 / -0.0043 | 1.0 |
| mutual | +0.388 / +0.474 / +0.559 | 1.0 | -0.034 / -0.014 / +0.000 | 0.896 | 1.0 | -0.0214 / -0.0150 / -0.0089 | 1.0 |
| cuts_only | +0.552 / +0.669 / +0.778 | 1.0 | +0.000 / +0.028 / +0.084 | 0.0 | 0.3585 | +0.0001 / +0.0046 / +0.0124 | 0.0 |

### Baselines (same resamples)

- plugin: right share [0.1471, 0.1562, 0.2177, 0.2991, 0.3111], wrong share [0.0394, 0.0476, 0.0992, 0.1598, 0.1711], impure [0.0207, 0.0223, 0.0307, 0.0398, 0.0415]
- split: right share [0.0, 0.0, 0.0, 0.0, 0.0], wrong share [0.0, 0.0075, 0.0278, 0.0513, 0.0563], impure [0.0159, 0.0171, 0.0257, 0.0352, 0.0369]
- mutual: right share [0.1187, 0.1296, 0.1911, 0.2575, 0.2712], wrong share [0.0, 0.0075, 0.0403, 0.0872, 0.0949], impure [0.0176, 0.0194, 0.0297, 0.0412, 0.0431]
- cuts_only: right share [0.0, 0.0, 0.0, 0.0, 0.0], wrong share [0.0, 0.0, 0.0, 0.0, 0.0], impure [0.0049, 0.0055, 0.0097, 0.0148, 0.0156]

Resamples without SNGS-162: 717 of 2000 (35.9%). Wrong share in those / in the others:

- strict: without SNGS-162 median 0.0000 (max 0.0000); with it median 0.0315 (max 0.1260)
- balanced: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0576 (max 0.1940)
- recall: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0576 (max 0.1940)
- margin_0.50: without SNGS-162 median 0.0000 (max 0.0000); with it median 0.0315 (max 0.1260)
