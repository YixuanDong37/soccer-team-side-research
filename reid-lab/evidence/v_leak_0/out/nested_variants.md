# Nested design choices (LOCO held-out, pooled; right / wrong / lost | links right / wrong / unknown | impure rows)

## 1. Variant chosen inside each fold (on the 12 fit clips, same J as the margin)

| candidates | point | HELD-OUT | variant chosen per fold | margins |
|---|---|---|---|---|
| fuse | strict | 33 / 4 / 100 | 152 / 4 / 158 | 1.28% | {'fuse': 13} | {0.6: 1, 0.9: 10, 1.0: 2} |
| fuse | balanced | 93 / 7 / 37 | 290 / 10 / 376 | 1.89% | {'fuse': 13} | {0.25: 6, 0.5: 7} |
| fuse | recall | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | {'fuse': 13} | {0.2: 10, 0.25: 3} |
| fuse / fuse_pos | strict | 32 / 4 / 101 | 155 / 4 / 175 | 1.28% | {'fuse': 9, 'fuse_pos': 4} | {0.6: 1, 0.9: 12} |
| fuse / fuse_pos | balanced | 92 / 7 / 38 | 290 / 10 / 373 | 1.89% | {'fuse_pos': 7, 'fuse': 6} | {0.25: 6, 0.5: 7} |
| fuse / fuse_pos | recall | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | {'fuse': 13} | {0.2: 10, 0.25: 3} |
| fuse / fuse_pos / app_cut | strict | 32 / 4 / 101 | 155 / 4 / 175 | 1.28% | {'fuse': 9, 'fuse_pos': 4} | {0.6: 1, 0.9: 12} |
| fuse / fuse_pos / app_cut | balanced | 91 / 8 / 38 | 294 / 10 / 376 | 1.89% | {'fuse_pos': 6, 'fuse': 6, 'app_cut': 1} | {0.25: 6, 0.3: 1, 0.5: 6} |
| fuse / fuse_pos / app_cut | recall | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | {'fuse': 13} | {0.2: 10, 0.25: 3} |
| fuse / fuse_pos / app_cut / fuse_nocut / app_nocut | strict | 36 / 6 / 95 | 158 / 4 / 181 | 1.36% | {'app_nocut': 1, 'fuse': 8, 'fuse_pos': 4} | {0.3: 1, 0.6: 1, 0.9: 11} |
| fuse / fuse_pos / app_cut / fuse_nocut / app_nocut | balanced | 91 / 8 / 38 | 294 / 10 / 376 | 1.89% | {'fuse_pos': 6, 'fuse': 6, 'app_cut': 1} | {0.25: 6, 0.3: 1, 0.5: 6} |
| fuse / fuse_pos / app_cut / fuse_nocut / app_nocut | recall | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | {'fuse': 13} | {0.2: 10, 0.25: 3} |

## 2. Fit-clip rate against held-out rate at the margin chosen in the fold (variant fuse)

| point | fold | margin | fit clips: right share, wrong share | held-out clip: right / wrong / lost |
|---|---|---|---|---|
| strict | SNGS-107 | 0.9 | 0.256, 0.000 | 4 / 0 / 12 |
| strict | SNGS-163 | 0.9 | 0.333, 0.000 | 1 / 0 / 10 |
| strict | SNGS-106 | 0.9 | 0.285, 0.000 | 3 / 0 / 4 |
| strict | SNGS-097 | 0.9 | 0.267, 0.000 | 3 / 0 / 3 |
| strict | SNGS-067 | 0.9 | 0.299, 0.000 | 2 / 0 / 8 |
| strict | SNGS-100 | 0.9 | 0.316, 0.000 | 2 / 0 / 2 |
| strict | SNGS-066 | 0.9 | 0.214, 0.000 | 2 / 0 / 9 |
| strict | SNGS-161 | 0.9 | 0.260, 0.000 | 4 / 0 / 10 |
| strict | SNGS-169 | 1.0 | 0.123, 0.000 | 2 / 0 / 13 |
| strict | SNGS-162 | 0.6 | 0.555, 0.000 | 6 / 4 / 8 |
| strict | SNGS-151 | 1.0 | 0.148, 0.000 | 0 / 0 / 9 |
| strict | SNGS-102 | 0.9 | 0.288, 0.000 | 2 / 0 / 10 |
| strict | SNGS-112 | 0.9 | 0.256, 0.000 | 2 / 0 / 2 |
| strict | **all folds** | | **0.276, 0.000** (mean over folds of the fit-clip shares) | **0.241, 0.029** (33 / 4 of 137) |
| balanced | SNGS-107 | 0.5 | 0.669, 0.033 | 10 / 0 / 6 |
| balanced | SNGS-163 | 0.5 | 0.690, 0.032 | 5 / 0 / 6 |
| balanced | SNGS-106 | 0.25 | 0.808, 0.046 | 6 / 0 / 1 |
| balanced | SNGS-097 | 0.25 | 0.786, 0.046 | 6 / 0 / 0 |
| balanced | SNGS-067 | 0.5 | 0.661, 0.031 | 8 / 0 / 2 |
| balanced | SNGS-100 | 0.25 | 0.805, 0.053 | 3 / 0 / 1 |
| balanced | SNGS-066 | 0.25 | 0.810, 0.040 | 6 / 1 / 4 |
| balanced | SNGS-161 | 0.5 | 0.634, 0.033 | 13 / 0 / 1 |
| balanced | SNGS-169 | 0.5 | 0.631, 0.033 | 12 / 0 / 3 |
| balanced | SNGS-162 | 0.25 | 0.849, 0.008 | 7 / 6 / 5 |
| balanced | SNGS-151 | 0.5 | 0.680, 0.031 | 4 / 0 / 5 |
| balanced | SNGS-102 | 0.25 | 0.792, 0.048 | 10 / 0 / 2 |
| balanced | SNGS-112 | 0.5 | 0.654, 0.030 | 3 / 0 / 1 |
| balanced | **all folds** | | **0.729, 0.036** (mean over folds of the fit-clip shares) | **0.679, 0.051** (93 / 7 of 137) |
| recall | SNGS-107 | 0.25 | 0.777, 0.050 | 15 / 0 / 1 |
| recall | SNGS-163 | 0.25 | 0.810, 0.048 | 8 / 0 / 3 |
| recall | SNGS-106 | 0.2 | 0.831, 0.054 | 6 / 0 / 1 |
| recall | SNGS-097 | 0.2 | 0.809, 0.053 | 6 / 0 / 0 |
| recall | SNGS-067 | 0.2 | 0.819, 0.063 | 9 / 0 / 1 |
| recall | SNGS-100 | 0.2 | 0.827, 0.060 | 3 / 0 / 1 |
| recall | SNGS-066 | 0.2 | 0.841, 0.048 | 6 / 1 / 4 |
| recall | SNGS-161 | 0.2 | 0.797, 0.057 | 14 / 0 / 0 |
| recall | SNGS-169 | 0.2 | 0.803, 0.057 | 15 / 0 / 0 |
| recall | SNGS-162 | 0.2 | 0.874, 0.008 | 7 / 6 / 5 |
| recall | SNGS-151 | 0.25 | 0.805, 0.047 | 7 / 0 / 2 |
| recall | SNGS-102 | 0.2 | 0.816, 0.056 | 10 / 0 / 2 |
| recall | SNGS-112 | 0.2 | 0.812, 0.053 | 4 / 0 / 0 |
| recall | **all folds** | | **0.817, 0.050** (mean over folds of the fit-clip shares) | **0.803, 0.051** (110 / 7 of 137) |

## 3. Picking a fixed-margin row afterwards (variant fuse)

Oracle pick = the fixed margin with the most right returns among those with wrong <= cap on the POOLED HELD-OUT table;
nested pick = the same criterion applied inside each fold to its 12 fit clips (cap scaled by the fit clips' scored returns), then applied to the held-out clip.

| cap on wrong | oracle pick on the held-out table | nested pick (per fold) | margins chosen per fold | pre-fixed J rule (fuse's headline) |
|---|---|---|---|---|
| 4 | margin 0.5: 91 / 4 / 42 | 272 / 7 / 345 | 1.51% | 53 / 6 / 78 | 209 / 6 / 253 | 1.59% | {0.1: 1, 0.8: 12} | strict: 33 / 4 / 100 | 152 / 4 / 158 | 1.28% |
| 7 | margin 0.15: 114 / 7 / 16 | 326 / 12 / 459 | 2.19% | 111 / 7 / 19 | 316 / 10 / 427 | 1.98% | {0.1: 1, 0.15: 1, 0.25: 8, 0.3: 1, 0.4: 2} | balanced: 93 / 7 / 37 | 290 / 10 / 376 | 1.89% |
| 14 | margin 0.1: 116 / 9 / 12 | 330 / 15 / 482 | 2.31% | 116 / 9 / 12 | 330 / 15 / 482 | 2.31% | {0.1: 13} | recall: 110 / 7 / 20 | 314 / 10 / 437 | 1.91% |
