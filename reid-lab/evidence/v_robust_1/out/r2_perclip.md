# r2: per-clip results, concentration, clip bootstrap

## Per clip (held-out): right / wrong / lost

| clip | scored | strict | balanced | recall | fixed_strict | fixed_balanced | fixed_recall | plugin | split | mutual | cuts_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 16 | 8 / 0 / 8 | 9 / 0 / 7 | 12 / 0 / 4 | 8 / 0 / 8 | 9 / 0 / 7 | 16 / 0 / 0 | 1 / 2 / 13 | 0 / 1 / 15 | 3 / 1 / 12 | 0 / 0 / 16 |
| SNGS-163 | 11 | 2 / 0 / 9 | 4 / 0 / 7 | 8 / 0 / 3 | 3 / 0 / 8 | 5 / 0 / 6 | 8 / 0 / 3 | 1 / 0 / 10 | 0 / 0 / 11 | 1 / 0 / 10 | 0 / 0 / 11 |
| SNGS-106 | 7 | 3 / 0 / 4 | 4 / 0 / 3 | 4 / 0 / 3 | 3 / 0 / 4 | 4 / 0 / 3 | 6 / 0 / 1 | 4 / 2 / 1 | 0 / 0 / 7 | 2 / 0 / 5 | 0 / 0 / 7 |
| SNGS-097 | 6 | 4 / 0 / 2 | 5 / 0 / 1 | 6 / 0 / 0 | 5 / 0 / 1 | 5 / 0 / 1 | 6 / 0 / 0 | 3 / 0 / 3 | 0 / 0 / 6 | 1 / 0 / 5 | 0 / 0 / 6 |
| SNGS-067 | 10 | 5 / 0 / 5 | 8 / 0 / 2 | 9 / 0 / 1 | 5 / 0 / 5 | 8 / 0 / 2 | 9 / 0 / 1 | 1 / 1 / 8 | 0 / 0 / 10 | 0 / 0 / 10 | 0 / 0 / 10 |
| SNGS-100 | 4 | 2 / 0 / 2 | 3 / 0 / 1 | 3 / 0 / 1 | 2 / 0 / 2 | 3 / 0 / 1 | 3 / 0 / 1 | 1 / 0 / 3 | 0 / 0 / 4 | 3 / 0 / 1 | 0 / 0 / 4 |
| SNGS-066 | 11 | 5 / 1 / 5 | 5 / 1 / 5 | 6 / 1 / 4 | 4 / 0 / 7 | 5 / 1 / 5 | 6 / 1 / 4 | 1 / 3 / 7 | 0 / 1 / 10 | 0 / 1 / 10 | 0 / 0 / 11 |
| SNGS-161 | 14 | 8 / 0 / 6 | 13 / 0 / 1 | 14 / 0 / 0 | 10 / 0 / 4 | 12 / 0 / 2 | 14 / 0 / 0 | 6 / 0 / 8 | 0 / 0 / 14 | 4 / 0 / 10 | 0 / 0 / 14 |
| SNGS-169 | 15 | 11 / 0 / 4 | 11 / 0 / 4 | 15 / 0 / 0 | 11 / 0 / 4 | 11 / 0 / 4 | 15 / 0 / 0 | 4 / 1 / 10 | 0 / 0 / 15 | 4 / 0 / 11 | 0 / 0 / 15 |
| SNGS-162 | 18 | 3 / 2 / 13 | 4 / 4 / 10 | 8 / 6 / 4 | 3 / 2 / 13 | 4 / 3 / 11 | 8 / 6 / 4 | 2 / 5 / 11 | 0 / 2 / 16 | 1 / 4 / 13 | 0 / 0 / 18 |
| SNGS-151 | 9 | 3 / 0 / 6 | 5 / 0 / 4 | 7 / 0 / 2 | 3 / 0 / 6 | 5 / 0 / 4 | 7 / 0 / 2 | 2 / 0 / 7 | 0 / 0 / 9 | 1 / 0 / 8 | 0 / 0 / 9 |
| SNGS-102 | 12 | 10 / 0 / 2 | 10 / 0 / 2 | 10 / 0 / 2 | 10 / 0 / 2 | 10 / 0 / 2 | 10 / 0 / 2 | 2 / 0 / 10 | 0 / 0 / 12 | 4 / 0 / 8 | 0 / 0 / 12 |
| SNGS-112 | 4 | 3 / 0 / 1 | 4 / 0 / 0 | 4 / 0 / 0 | 3 / 0 / 1 | 4 / 0 / 0 | 4 / 0 / 0 | 2 / 0 / 2 | 0 / 0 / 4 | 2 / 0 / 2 | 0 / 0 / 4 |

## Is the gain carried by one or two clips?

**strict** (67 right, 3 wrong of 137):
- the clip with the most right returns (SNGS-169) holds 16% of them, the top two 31% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.167, median 0.5, max 0.833; 8 of 13 clips at or above one half;
- against plugin: 12 clips better, 1 worse, 0 equal in right returns; wrong higher in 0 clips, lower in 6; gain 37 right returns, 40% of it in ['SNGS-102', 'SNGS-107'];
- pooled right share with any one clip removed: [0.456, 0.538]; any two removed: [0.418, 0.574];
- pooled wrong share with any one clip removed: [0.0084, 0.0248]; any two removed: [0.0, 0.0283];
- wrong returns: 3, of which 67% in SNGS-162.

**balanced** (85 right, 5 wrong of 137):
- the clip with the most right returns (SNGS-161) holds 15% of them, the top two 28% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.222, median 0.733, max 1.0; 10 of 13 clips at or above one half;
- against plugin: 12 clips better, 0 worse, 1 equal in right returns; wrong higher in 0 clips, lower in 6; gain 55 right returns, 29% of it in ['SNGS-107', 'SNGS-102'];
- pooled right share with any one clip removed: [0.585, 0.681]; any two removed: [0.559, 0.713];
- pooled wrong share with any one clip removed: [0.0084, 0.0413]; any two removed: [0.0, 0.0472];
- wrong returns: 5, of which 80% in SNGS-162.

**recall** (106 right, 7 wrong of 137):
- the clip with the most right returns (SNGS-169) holds 14% of them, the top two 27% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.444, median 0.778, max 1.0; 12 of 13 clips at or above one half;
- against plugin: 12 clips better, 0 worse, 1 equal in right returns; wrong higher in 1 clips, lower in 5; gain 76 right returns, 29% of it in ['SNGS-107', 'SNGS-169'];
- pooled right share with any one clip removed: [0.746, 0.824]; any two removed: [0.713, 0.852];
- pooled wrong share with any one clip removed: [0.0084, 0.0579]; any two removed: [0.0, 0.066];
- wrong returns: 7, of which 86% in SNGS-162.

**fixed_strict** (70 right, 2 wrong of 137):
- the clip with the most right returns (SNGS-169) holds 16% of them, the top two 30% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.167, median 0.5, max 0.833; 8 of 13 clips at or above one half;
- against plugin: 12 clips better, 1 worse, 0 equal in right returns; wrong higher in 0 clips, lower in 6; gain 40 right returns, 38% of it in ['SNGS-102', 'SNGS-107'];
- pooled right share with any one clip removed: [0.48, 0.563]; any two removed: [0.445, 0.593];
- pooled wrong share with any one clip removed: [0.0, 0.0165]; any two removed: [0.0, 0.0189];
- wrong returns: 2, of which 100% in SNGS-162.

**fixed_balanced** (85 right, 4 wrong of 137):
- the clip with the most right returns (SNGS-161) holds 14% of them, the top two 27% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.222, median 0.733, max 1.0; 10 of 13 clips at or above one half;
- against plugin: 12 clips better, 0 worse, 1 equal in right returns; wrong higher in 0 clips, lower in 6; gain 55 right returns, 29% of it in ['SNGS-107', 'SNGS-102'];
- pooled right share with any one clip removed: [0.593, 0.681]; any two removed: [0.568, 0.704];
- pooled wrong share with any one clip removed: [0.0084, 0.0331]; any two removed: [0.0, 0.0377];
- wrong returns: 4, of which 75% in SNGS-162.

**fixed_recall** (112 right, 7 wrong of 137):
- the clip with the most right returns (SNGS-107) holds 14% of them, the top two 28% (the largest clip has 13% of the scored returns);
- right share per clip: min 0.444, median 0.857, max 1.0; 12 of 13 clips at or above one half;
- against plugin: 13 clips better, 0 worse, 0 equal in right returns; wrong higher in 1 clips, lower in 5; gain 82 right returns, 32% of it in ['SNGS-107', 'SNGS-169'];
- pooled right share with any one clip removed: [0.793, 0.874]; any two removed: [0.764, 0.907];
- pooled wrong share with any one clip removed: [0.0084, 0.0579]; any two removed: [0.0, 0.066];
- wrong returns: 7, of which 86% in SNGS-162.

## Clip bootstrap (2000 resamples of the 13 clips, seed 0, paired)

Shares are of the scored returns in the resample; quantiles 2.5% / 5% / 50% / 95% / 97.5%. x137 = the share times 137 (count-equivalent).

### strict

- right share [0.3562, 0.3806, 0.4926, 0.5946, 0.6143] (x137: [48.8, 52.1, 67.5, 81.5, 84.2])
- wrong share [0.0, 0.0, 0.0199, 0.0458, 0.051] (x137: [0.0, 0.0, 2.7, 6.3, 7.0]); P(wrong share <= 4/137, 7/137, 14/137) = [0.746, 0.977, 1.0]
- link precision (judged links) [0.9565, 0.9596, 0.9797, 0.9941, 0.9966]; impure row share [0.0074, 0.0086, 0.0154, 0.024, 0.0254]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.145 / +0.270 / +0.392 | 1.0 | -0.127 / -0.079 / -0.036 | 1.0 | 1.0 | -0.0224 / -0.0148 / -0.0082 | 1.0 |
| split | +0.356 / +0.493 / +0.614 | 1.0 | -0.023 / -0.007 / +0.000 | 0.662 | 1.0 | -0.0168 / -0.0099 / -0.0043 | 1.0 |
| mutual | +0.202 / +0.299 / +0.391 | 1.0 | -0.048 / -0.020 / +0.000 | 0.8905 | 1.0 | -0.0210 / -0.0140 / -0.0083 | 1.0 |
| cuts_only | +0.356 / +0.493 / +0.614 | 1.0 | +0.000 / +0.020 / +0.051 | 0.0 | 0.1205 | +0.0009 / +0.0057 / +0.0127 | 0.0 |

### balanced

- right share [0.4841, 0.5042, 0.6279, 0.7368, 0.752] (x137: [66.3, 69.1, 86.0, 100.9, 103.0])
- wrong share [0.0, 0.0, 0.0328, 0.0828, 0.0903] (x137: [0.0, 0.0, 4.5, 11.3, 12.4]); P(wrong share <= 4/137, 7/137, 14/137) = [0.43, 0.753, 0.992]
- link precision (judged links) [0.9169, 0.9244, 0.9626, 0.9907, 0.9931]; impure row share [0.0084, 0.01, 0.0202, 0.033, 0.0355]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.288 / +0.406 / +0.512 | 1.0 | -0.110 / -0.064 / -0.030 | 1.0 | 1.0 | -0.0205 / -0.0104 / +0.0024 | 0.947 |
| split | +0.484 / +0.628 / +0.752 | 1.0 | -0.020 / +0.007 / +0.040 | 0.2805 | 0.4725 | -0.0148 / -0.0055 / +0.0062 | 0.8305 |
| mutual | +0.325 / +0.431 / +0.533 | 1.0 | -0.023 / -0.007 / +0.000 | 0.662 | 1.0 | -0.0164 / -0.0091 / -0.0030 | 0.9995 |
| cuts_only | +0.484 / +0.628 / +0.752 | 1.0 | +0.000 / +0.033 / +0.090 | 0.0 | 0.1205 | +0.0020 / +0.0103 / +0.0245 | 0.0 |

### recall

- right share [0.6538, 0.6739, 0.7787, 0.8722, 0.8862] (x137: [89.6, 92.3, 106.7, 119.5, 121.4])
- wrong share [0.0, 0.0, 0.047, 0.1208, 0.1325] (x137: [0.0, 0.0, 6.4, 16.5, 18.2]); P(wrong share <= 4/137, 7/137, 14/137) = [0.353, 0.581, 0.908]
- link precision (judged links) [0.9195, 0.9261, 0.964, 0.9902, 0.9926]; impure row share [0.0089, 0.0104, 0.0207, 0.0338, 0.0363]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.442 / +0.559 / +0.650 | 1.0 | -0.106 / -0.052 / -0.006 | 0.977 | 0.9935 | -0.0202 / -0.0098 / +0.0032 | 0.936 |
| split | +0.654 / +0.779 / +0.886 | 1.0 | -0.020 / +0.020 / +0.081 | 0.25 | 0.3665 | -0.0145 / -0.0050 / +0.0070 | 0.805 |
| mutual | +0.487 / +0.585 / +0.680 | 1.0 | -0.020 / +0.007 / +0.040 | 0.2805 | 0.4725 | -0.0161 / -0.0086 / -0.0023 | 0.9965 |
| cuts_only | +0.654 / +0.779 / +0.886 | 1.0 | +0.000 / +0.047 / +0.133 | 0.0 | 0.1205 | +0.0023 / +0.0108 / +0.0253 | 0.0 |

### fixed_strict

- right share [0.375, 0.3984, 0.5154, 0.6205, 0.6397] (x137: [51.4, 54.6, 70.6, 85.0, 87.6])
- wrong share [0.0, 0.0, 0.014, 0.0385, 0.042] (x137: [0.0, 0.0, 1.9, 5.3, 5.8]); P(wrong share <= 4/137, 7/137, 14/137) = [0.883, 0.994, 1.0]
- link precision (judged links) [0.9597, 0.9626, 0.9832, 0.9969, 1.0]; impure row share [0.0073, 0.0083, 0.014, 0.0206, 0.0219]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.170 / +0.293 / +0.402 | 1.0 | -0.142 / -0.086 / -0.038 | 1.0 | 1.0 | -0.0240 / -0.0166 / -0.0093 | 1.0 |
| split | +0.375 / +0.515 / +0.640 | 1.0 | -0.034 / -0.014 / +0.000 | 0.896 | 1.0 | -0.0185 / -0.0116 / -0.0053 | 1.0 |
| mutual | +0.221 / +0.323 / +0.410 | 1.0 | -0.056 / -0.028 / +0.000 | 0.97 | 1.0 | -0.0224 / -0.0156 / -0.0094 | 1.0 |
| cuts_only | +0.375 / +0.515 / +0.640 | 1.0 | +0.000 / +0.014 / +0.042 | 0.0 | 0.3585 | +0.0000 / +0.0040 / +0.0110 | 0.0 |

### fixed_balanced

- right share [0.4921, 0.5123, 0.6285, 0.7273, 0.7414] (x137: [67.4, 70.2, 86.1, 99.6, 101.6])
- wrong share [0.0, 0.0, 0.0258, 0.0638, 0.0702] (x137: [0.0, 0.0, 3.5, 8.7, 9.6]); P(wrong share <= 4/137, 7/137, 14/137) = [0.582, 0.867, 1.0]
- link precision (judged links) [0.9302, 0.9357, 0.9701, 0.9943, 0.997]; impure row share [0.0077, 0.0089, 0.0174, 0.0279, 0.0299]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.289 / +0.404 / +0.509 | 1.0 | -0.116 / -0.073 / -0.033 | 1.0 | 1.0 | -0.0218 / -0.0130 / -0.0036 | 0.9965 |
| split | +0.492 / +0.628 / +0.741 | 1.0 | -0.021 / +0.000 / +0.020 | 0.3705 | 0.669 | -0.0159 / -0.0082 / +0.0003 | 0.9705 |
| mutual | +0.333 / +0.432 / +0.528 | 1.0 | -0.030 / -0.014 / +0.000 | 0.8905 | 1.0 | -0.0185 / -0.0120 / -0.0070 | 1.0 |
| cuts_only | +0.492 / +0.628 / +0.741 | 1.0 | +0.000 / +0.026 / +0.070 | 0.0 | 0.1205 | +0.0009 / +0.0075 / +0.0183 | 0.0 |

### fixed_recall

- right share [0.6865, 0.7062, 0.8261, 0.9193, 0.9329] (x137: [94.1, 96.7, 113.2, 125.9, 127.8])
- wrong share [0.0, 0.0, 0.047, 0.1208, 0.1325] (x137: [0.0, 0.0, 6.4, 16.5, 18.2]); P(wrong share <= 4/137, 7/137, 14/137) = [0.353, 0.581, 0.908]
- link precision (judged links) [0.9201, 0.9266, 0.964, 0.9902, 0.9926]; impure row share [0.0089, 0.0104, 0.0208, 0.0338, 0.0363]

| against | right share diff (2.5 / 50 / 97.5%) | P(right higher) | wrong share diff (2.5 / 50 / 97.5%) | P(wrong lower) | P(wrong not higher) | impure share diff (2.5 / 50 / 97.5%) | P(impure lower) |
|---|---|---|---|---|---|---|---|
| plugin | +0.478 / +0.601 / +0.717 | 1.0 | -0.106 / -0.052 / -0.006 | 0.977 | 0.9935 | -0.0202 / -0.0098 / +0.0032 | 0.936 |
| split | +0.686 / +0.826 / +0.933 | 1.0 | -0.020 / +0.020 / +0.081 | 0.25 | 0.3665 | -0.0145 / -0.0050 / +0.0071 | 0.8025 |
| mutual | +0.520 / +0.632 / +0.723 | 1.0 | -0.020 / +0.007 / +0.040 | 0.2805 | 0.4725 | -0.0161 / -0.0086 / -0.0023 | 0.9965 |
| cuts_only | +0.686 / +0.826 / +0.933 | 1.0 | +0.000 / +0.047 / +0.133 | 0.0 | 0.1205 | +0.0023 / +0.0108 / +0.0254 | 0.0 |

### Baselines (same resamples)

- plugin: right share [0.1471, 0.1562, 0.2177, 0.2991, 0.3111], wrong share [0.0394, 0.0476, 0.0992, 0.1598, 0.1711], impure [0.0207, 0.0223, 0.0307, 0.0398, 0.0415]
- split: right share [0.0, 0.0, 0.0, 0.0, 0.0], wrong share [0.0, 0.0075, 0.0278, 0.0513, 0.0563], impure [0.0159, 0.0171, 0.0257, 0.0352, 0.0369]
- mutual: right share [0.1187, 0.1296, 0.1911, 0.2575, 0.2712], wrong share [0.0, 0.0075, 0.0403, 0.0872, 0.0949], impure [0.0176, 0.0194, 0.0297, 0.0412, 0.0431]
- cuts_only: right share [0.0, 0.0, 0.0, 0.0, 0.0], wrong share [0.0, 0.0, 0.0, 0.0, 0.0], impure [0.0049, 0.0055, 0.0097, 0.0148, 0.0156]

Resamples without SNGS-162: 717 of 2000 (35.9%). Wrong share in those / in the others:

- strict: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0260 (max 0.0746)
- balanced: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0420 (max 0.1343)
- recall: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0576 (max 0.1940)
- fixed_strict: without SNGS-162 median 0.0000 (max 0.0000); with it median 0.0157 (max 0.0630)
- fixed_balanced: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0345 (max 0.1045)
- fixed_recall: without SNGS-162 median 0.0076 (max 0.0388); with it median 0.0576 (max 0.1940)
