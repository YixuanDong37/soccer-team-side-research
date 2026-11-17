# r10: the choice of the operating points (rule B) and how much it depends on the rule's own numbers

Works on fuse_r1's scored LOCO results (cache/loco_raw.pkl). Nothing is re-linked.

## 1. Integrity

- strict: pooled held-out cells of loco_raw 67 / 3 / 67 | 312 / 7 / 25 = score of METHOD_strict.json: True
- balanced: pooled held-out cells of loco_raw 85 / 5 / 47 | 378 / 16 / 30 = score of METHOD_balanced.json: True
- recall: pooled held-out cells of loco_raw 106 / 7 / 24 | 414 / 17 / 260 = score of METHOD_recall.json: True
- settings re-derived with fuse_r1's rules.choose and with my own re-implementation for 3 points x 14 folds: 42 of 42 equal params_loco.json
- the wrong-return budget of constraint (i) on the 12 fit clips, floor(K x scored_fit / 137), over the 13 folds: {'strict': [3], 'balanced': [6], 'recall': [12, 13]} (K = 4 / 7 / 14)

## 2. Rule B with its own numbers moved (held-out, pooled; nothing re-linked)

| variant | right / wrong / lost | links r / w / u | impure | SNGS-162 | settings chosen by the folds | all 13 | folds = all 13 | limit |
|---|---|---|---|---|---|---|---|---|
| **strict** (K 4, lam 10) | | | | | | | | |
| rule B as fixed | 67 / 3 / 67 | 312 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 9x 0.60 core, 3x 0.70 core | 0.60 core | 9 / 13 | ok |
| lam x 0.9 = 9 | 68 / 3 / 66 | 313 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 10x 0.60 core, 2x 0.70 core | 0.60 core | 10 / 13 | ok |
| lam x 1.1 = 11 | 65 / 3 / 69 | 294 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 7x 0.60 core, 5x 0.70 core | 0.60 core | 7 / 13 | ok |
| lam x 0.5 = 5 | 72 / 4 / 61 | 323 / 11 / 27 | 1.80% | 4 / 3 / 11 | 1x 0.40 core, 1x 0.50 core, 11x 0.60 core | 0.50 core | 1 / 13 | ok |
| lam x 0.7 = 7 | 71 / 3 / 63 | 320 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 12x 0.60 core | 0.60 core | 12 / 13 | ok |
| lam x 1.5 = 15 | 58 / 2 / 77 | 278 / 6 / 19 | 1.44% | 3 / 2 / 13 | 7x 0.60 core, 3x 0.70 core, 3x 0.80 core | 0.70 core | 3 / 13 | ok |
| lam x 2 = 20 | 58 / 2 / 77 | 278 / 6 / 19 | 1.44% | 3 / 2 / 13 | 7x 0.60 core, 3x 0.70 core, 3x 0.80 core | 0.70 core | 3 / 13 | ok |
| cap K - 1 = 3 | 66 / 2 / 69 | 311 / 6 / 22 | 1.44% | 3 / 2 / 13 | 10x 0.60 core, 3x 0.70 core | 0.60 core | 10 / 13 | ok |
| cap K + 1 = 5 | 67 / 3 / 67 | 312 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 9x 0.60 core, 3x 0.70 core | 0.60 core | 9 / 13 | ok |
| budget not scaled to the fit clips (K itself) | 67 / 3 / 67 | 312 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 9x 0.60 core, 3x 0.70 core | 0.60 core | 9 / 13 | ok |
| no fit clip left out of J | 26 / 2 / 109 | 191 / 5 / 65 | 1.33% | 3 / 2 / 13 | 1x 0.60 core, 12x 0.90 +leaves | 0.90 +leaves | 12 / 13 | ok |
| two fit clips left out of J | 69 / 4 / 64 | 317 / 11 / 27 | 1.80% | 4 / 3 / 11 | 2x 0.40 core, 9x 0.60 core, 2x 0.70 core | 0.60 core | 9 / 13 | ok |
| w = wrong returns only (no link terms) | 76 / 7 / 54 | 328 / 13 / 66 | 1.97% | 8 / 6 / 4 | 1x 0.25 +leaves, 1x 0.35 core, 11x 0.60 core | 0.40 core | 0 / 13 | OVER |
| w = wrong returns + links judged wrong (no weak-label leaf term) | 70 / 5 / 62 | 312 / 7 / 42 | 1.61% | 6 / 4 / 8 | 1x 0.50 core, 1x 0.60 +leaves, 8x 0.60 core, 3x 0.70 core | 0.60 +leaves | 1 / 13 | OVER |
| ties towards the smaller margin | 68 / 3 / 66 | 313 / 7 / 25 | 1.61% | 3 / 2 / 13 | 1x 0.50 core, 10x 0.60 core, 2x 0.70 core | 0.60 core | 10 / 13 | ok |
| SNGS-162 may not be the clip left out | 26 / 2 / 109 | 191 / 5 / 65 | 1.33% | 3 / 2 / 13 | 1x 0.60 core, 12x 0.90 +leaves | 0.90 +leaves | 12 / 13 | ok |
| **balanced** (K 7, lam 3) | | | | | | | | |
| rule B as fixed | 85 / 5 / 47 | 378 / 16 / 30 | 2.12% | 4 / 4 / 10 | 1x 0.15 core, 4x 0.20 core, 1x 0.25 core, 4x 0.40 core, 3x 0.50 core | 0.40 core | 4 / 13 | ok |
| lam x 0.9 = 2.7 | 86 / 5 / 46 | 387 / 16 / 34 | 2.14% | 4 / 4 / 10 | 1x 0.15 core, 4x 0.20 core, 4x 0.25 core, 2x 0.40 core, 1x 0.45 core, 1x 0.50 core | 0.25 core | 4 / 13 | ok |
| lam x 1.1 = 3.3 | 84 / 5 / 48 | 364 / 16 / 30 | 2.12% | 4 / 4 / 10 | 1x 0.15 core, 1x 0.20 core, 1x 0.25 core, 4x 0.40 core, 6x 0.50 core | 0.40 core | 4 / 13 | ok |
| lam x 0.5 = 1.5 | 95 / 6 / 36 | 416 / 18 / 36 | 2.15% | 4 / 4 / 10 | 2x 0.15 core, 4x 0.20 core, 7x 0.25 core | 0.25 core | 7 / 13 | ok |
| lam x 0.7 = 2.1 | 93 / 6 / 38 | 409 / 18 / 36 | 2.15% | 4 / 4 / 10 | 2x 0.15 core, 4x 0.20 core, 6x 0.25 core, 1x 0.40 core | 0.25 core | 6 / 13 | ok |
| lam x 1.5 = 4.5 | 77 / 4 / 56 | 338 / 14 / 28 | 1.97% | 4 / 3 / 11 | 1x 0.25 core, 1x 0.40 core, 1x 0.45 core, 5x 0.50 core, 5x 0.60 core | 0.50 core | 5 / 13 | ok |
| lam x 2 = 6 | 71 / 3 / 63 | 322 / 9 / 25 | 1.77% | 3 / 2 / 13 | 1x 0.30 core, 1x 0.50 core, 11x 0.60 core | 0.60 core | 11 / 13 | ok |
| cap K - 1 = 6 | 85 / 5 / 47 | 378 / 16 / 30 | 2.12% | 4 / 4 / 10 | 1x 0.15 core, 4x 0.20 core, 1x 0.25 core, 4x 0.40 core, 3x 0.50 core | 0.40 core | 4 / 13 | ok |
| cap K + 1 = 8 | 85 / 5 / 47 | 378 / 16 / 30 | 2.12% | 4 / 4 / 10 | 1x 0.15 core, 4x 0.20 core, 1x 0.25 core, 4x 0.40 core, 3x 0.50 core | 0.40 core | 4 / 13 | ok |
| budget not scaled to the fit clips (K itself) | 85 / 5 / 47 | 378 / 16 / 30 | 2.12% | 4 / 4 / 10 | 1x 0.15 core, 4x 0.20 core, 1x 0.25 core, 4x 0.40 core, 3x 0.50 core | 0.40 core | 4 / 13 | ok |
| no fit clip left out of J | 72 / 4 / 61 | 327 / 12 / 25 | 1.79% | 4 / 4 / 10 | 1x 0.30 core, 1x 0.55 core, 11x 0.60 core | 0.60 core | 11 / 13 | ok |
| two fit clips left out of J | 87 / 5 / 45 | 386 / 16 / 55 | 2.14% | 4 / 4 / 10 | 4x 0.15 core, 4x 0.20 core, 1x 0.40 core, 1x 0.45 +leaves, 3x 0.45 core | 0.15 core | 4 / 13 | ok |
| w = wrong returns only (no link terms) | 96 / 9 / 32 | 392 / 19 / 186 | 2.21% | 8 / 7 / 3 | 1x 0.10 +leaves, 1x 0.15 +leaves, 3x 0.20 core, 3x 0.25 core, 1x 0.35 +leaves, 4x 0.40 +leaves | 0.20 +leaves | 0 / 13 | OVER |
| w = wrong returns + links judged wrong (no weak-label leaf term) | 96 / 8 / 33 | 370 / 16 / 253 | 2.01% | 8 / 6 / 4 | 2x 0.15 +leaves, 1x 0.35 +leaves, 4x 0.40 +leaves, 1x 0.45 +leaves, 5x 0.50 +leaves | 0.20 +leaves | 0 / 13 | OVER |
| ties towards the smaller margin | 86 / 5 / 46 | 389 / 16 / 34 | 2.14% | 4 / 4 / 10 | 2x 0.15 core, 7x 0.20 core, 2x 0.40 core, 1x 0.45 core, 1x 0.50 core | 0.20 core | 7 / 13 | ok |
| SNGS-162 may not be the clip left out | 73 / 5 / 59 | 335 / 13 / 25 | 1.95% | 4 / 4 / 10 | 1x 0.15 core, 2x 0.45 core, 3x 0.55 core, 7x 0.60 core | 0.60 core | 7 / 13 | ok |
| **recall** (K 14, lam 1) | | | | | | | | |
| rule B as fixed | 106 / 7 / 24 | 414 / 17 / 260 | 2.17% | 8 / 6 / 4 | 10x 0.20 +leaves, 1x 0.20 core, 1x 0.25 +leaves, 1x 0.25 core | 0.20 +leaves | 10 / 13 | ok |
| lam x 0.9 = 0.9 | 108 / 8 / 21 | 416 / 18 / 284 | 2.19% | 8 / 6 / 4 | 1x 0.15 +leaves, 11x 0.20 +leaves, 1x 0.25 core | 0.20 +leaves | 11 / 13 | ok |
| lam x 1.1 = 1.1 | 103 / 7 / 27 | 414 / 17 / 210 | 2.17% | 8 / 6 / 4 | 7x 0.20 +leaves, 2x 0.20 core, 1x 0.25 +leaves, 3x 0.25 core | 0.20 +leaves | 7 / 13 | ok |
| lam x 0.5 = 0.5 | 112 / 8 / 17 | 416 / 18 / 311 | 2.19% | 8 / 6 / 4 | 1x 0.15 +leaves, 12x 0.20 +leaves | 0.20 +leaves | 12 / 13 | ok |
| lam x 0.7 = 0.7 | 112 / 8 / 17 | 416 / 18 / 311 | 2.19% | 8 / 6 / 4 | 1x 0.15 +leaves, 12x 0.20 +leaves | 0.20 +leaves | 12 / 13 | ok |
| lam x 1.5 = 1.5 | 97 / 6 / 34 | 416 / 18 / 73 | 2.16% | 4 / 4 / 10 | 2x 0.15 core, 1x 0.20 +leaves, 4x 0.20 core, 6x 0.25 core | 0.25 core | 6 / 13 | ok |
| lam x 2 = 2 | 93 / 6 / 38 | 409 / 18 / 36 | 2.15% | 4 / 4 / 10 | 2x 0.15 core, 4x 0.20 core, 6x 0.25 core, 1x 0.40 core | 0.25 core | 6 / 13 | ok |
| cap K - 1 = 13 | 106 / 7 / 24 | 414 / 17 / 260 | 2.17% | 8 / 6 / 4 | 10x 0.20 +leaves, 1x 0.20 core, 1x 0.25 +leaves, 1x 0.25 core | 0.20 +leaves | 10 / 13 | ok |
| cap K + 1 = 15 | 106 / 7 / 24 | 414 / 17 / 260 | 2.17% | 8 / 6 / 4 | 10x 0.20 +leaves, 1x 0.20 core, 1x 0.25 +leaves, 1x 0.25 core | 0.20 +leaves | 10 / 13 | ok |
| budget not scaled to the fit clips (K itself) | 106 / 7 / 24 | 414 / 17 / 260 | 2.17% | 8 / 6 / 4 | 10x 0.20 +leaves, 1x 0.20 core, 1x 0.25 +leaves, 1x 0.25 core | 0.20 +leaves | 10 / 13 | ok |
| no fit clip left out of J | 97 / 8 / 32 | 406 / 18 / 90 | 2.17% | 8 / 6 / 4 | 1x 0.15 core, 1x 0.20 +leaves, 3x 0.20 core, 1x 0.25 +leaves, 6x 0.25 core, 1x 0.40 core | 0.25 core | 6 / 13 | ok |
| two fit clips left out of J | 112 / 7 / 18 | 414 / 17 / 310 | 2.17% | 8 / 6 / 4 | 12x 0.20 +leaves, 1x 0.25 +leaves | 0.20 +leaves | 12 / 13 | ok |
| w = wrong returns only (no link terms) | 112 / 9 / 16 | 419 / 19 / 318 | 2.22% | 8 / 7 / 3 | 2x 0.10 +leaves, 11x 0.20 +leaves | 0.20 +leaves | 11 / 13 | ok |
| w = wrong returns + links judged wrong (no weak-label leaf term) | 112 / 8 / 17 | 418 / 18 / 312 | 2.19% | 8 / 6 / 4 | 2x 0.15 +leaves, 11x 0.20 +leaves | 0.20 +leaves | 11 / 13 | ok |
| ties towards the smaller margin | 106 / 8 / 23 | 416 / 18 / 261 | 2.19% | 8 / 6 / 4 | 1x 0.15 +leaves, 10x 0.20 +leaves, 2x 0.20 core | 0.20 +leaves | 10 / 13 | ok |
| SNGS-162 may not be the clip left out | 93 / 8 / 36 | 385 / 18 / 193 | 2.18% | 8 / 6 / 4 | 2x 0.15 core, 1x 0.20 +leaves, 2x 0.20 core, 4x 0.25 +leaves, 2x 0.45 +leaves, 1x 0.45 core, 1x 0.55 +leaves | 0.15 core | 2 / 13 | ok |

## 3. Which fit clip rule B leaves out of J at the chosen setting

- strict: SNGS-162 in 12 folds, SNGS-112 in 1 folds
- balanced: SNGS-162 in 12 folds, SNGS-066 in 1 folds
- recall: SNGS-162 in 12 folds, SNGS-100 in 1 folds

## 4. Parameter transfer at the frozen settings: a clip under the numbers of its own fold (held-out) against the 12 folds that had it in their fit

| setting | held-out pooled right / wrong | mean over the 12 in-fit folds, summed over clips: right / wrong | held-out minus in-fit: right / wrong | all-13 numbers (in-sample) right / wrong | pooled over all 13 clips under each single fold's numbers: right min-max, wrong min-max |
|---|---|---|---|---|---|
| strict: 0.60 core | 70 / 2 | 69.3 / 2.0 | +0.7 / +0.0 | 69 / 2 | 66-72, 2-2 |
| balanced: 0.40 core | 85 / 4 | 86.2 / 4.0 | -1.2 / +0.0 | 86 / 4 | 85-87, 4-4 |
| recall: 0.20 +leaves | 112 / 7 | 112.3 / 7.1 | -0.3 / -0.1 | 112 / 7 | 112-113, 7-8 |

Pair-model weights over the 13 folds (cos_w, sink_lo, plug, bias) and the bonus per unit of sink_lo / plug in cosine units:

- cos_w 6.85 to 8.43; sink_lo 0.620 to 0.663; plug -0.37 to 0.33; bias -7.62 to -7.05; all13: [7.176, 0.647, -0.109, -7.183]
- sink_lo / cos_w: 0.0787 to 0.0942 (largest possible position bonus = ratio x 10.80: 0.85 to 1.02); plug / cos_w: -0.053 to 0.046
