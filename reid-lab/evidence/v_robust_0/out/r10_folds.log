# r10: integrity and stability of fuse's leave-one-clip-out run

## 1. Margins re-derived from the raw results with fuse's rule

Mismatches with params_loco.json: none (39 of 39 equal).

Held-out pooled result re-derived from the raw results: strict 33 / 4 / 100 | 152 / 4 / 158 | 1.28%; balanced 93 / 7 / 37 | 290 / 10 / 376 | 1.89%; recall 110 / 7 / 20 | 314 / 10 / 437 | 1.91%

## 2. How flat is the objective, and what if lam had been a little different?

J of the chosen margin minus J of the best other margin on the 12 fit clips, per fold (small = the choice can flip):

| fold (held-out clip) | strict: chosen, J, runner-up (J) | balanced | recall |
|---|---|---|---|
| SNGS-107 | 0.9 (11); next 0.8 (4) | 0.5 (51); next 0.25 (46) | 0.25 (78); next 0.2 (78) |
| SNGS-163 | 0.9 (22); next 1.2 (3) | 0.5 (54); next 0.25 (54) | 0.25 (86); next 0.2 (86) |
| SNGS-106 | 0.9 (17); next 1.2 (2) | 0.25 (57); next 0.5 (56) | 0.2 (90); next 0.25 (89) |
| SNGS-097 | 0.9 (15); next 1.2 (2) | 0.25 (58); next 0.2 (55) | 0.2 (89); next 0.25 (88) |
| SNGS-067 | 0.9 (18); next 1.2 (4) | 0.5 (51); next 0.55 (50) | 0.2 (86); next 0.25 (84) |
| SNGS-100 | 0.9 (22); next 1.2 (5) | 0.25 (62); next 0.5 (60) | 0.2 (93); next 0.25 (92) |
| SNGS-066 | 0.9 (7); next 0.8 (6) | 0.25 (60); next 0.2 (58) | 0.2 (90); next 0.15 (90) |
| SNGS-161 | 0.9 (12); next 1.0 (7) | 0.5 (48); next 0.25 (45) | 0.2 (80); next 0.15 (80) |
| SNGS-169 | 1.0 (5); next 0.9 (5) | 0.5 (47); next 0.25 (47) | 0.2 (80); next 0.25 (79) |
| SNGS-162 | 0.6 (56); next 0.55 (54) | 0.25 (83); next 0.2 (83) | 0.2 (97); next 0.25 (95) |
| SNGS-151 | 1.0 (9); next 0.9 (9) | 0.5 (57); next 0.25 (55) | 0.25 (87); next 0.2 (87) |
| SNGS-102 | 0.9 (16); next 0.8 (5) | 0.25 (51); next 0.5 (50) | 0.2 (84); next 0.25 (83) |
| SNGS-112 | 0.9 (24); next 0.8 (18) | 0.5 (60); next 0.25 (60) | 0.2 (91); next 0.25 (90) |

Held-out pooled result when the rule's penalty lam is changed (same rule, same raw results):

| lam | margins chosen per fold | held-out right / wrong / lost; links r / w / u; impure |
|---|---|---|
| 20 | 1x 0.6, 1x 0.9, 11x 1.2 | 9 / 4 / 124; 55 / 4 / 70; 1.28% |
| 15 | 1x 0.6, 7x 0.9, 1x 1.0, 4x 1.2 | 21 / 4 / 112; 114 / 4 / 122; 1.28% |
| 12 | 1x 0.6, 10x 0.9, 2x 1.0 | 33 / 4 / 100; 152 / 4 / 158; 1.28% |
| 11 | 1x 0.6, 10x 0.9, 2x 1.0 | 33 / 4 / 100; 152 / 4 / 158; 1.28% |
| 10 (strict) | 1x 0.6, 10x 0.9, 2x 1.0 | 33 / 4 / 100; 152 / 4 / 158; 1.28% |
| 9 | 1x 0.6, 1x 0.8, 11x 0.9 | 36 / 4 / 97; 159 / 4 / 177; 1.28% |
| 8 | 1x 0.6, 2x 0.8, 10x 0.9 | 40 / 4 / 93; 166 / 4 / 188; 1.28% |
| 7 | 1x 0.5, 1x 0.55, 2x 0.8, 9x 0.9 | 45 / 4 / 88; 172 / 5 / 208; 1.29% |
| 6 | 1x 0.25, 6x 0.5, 2x 0.8, 4x 0.9 | 59 / 6 / 72; 214 / 6 / 286; 1.55% |
| 5 | 1x 0.25, 10x 0.5, 1x 0.55, 1x 0.9 | 85 / 6 / 46; 263 / 8 / 338; 1.65% |
| 4 | 2x 0.25, 10x 0.5, 1x 0.55 | 90 / 7 / 40; 278 / 9 / 354; 1.73% |
| 3.3 | 4x 0.25, 9x 0.5 | 92 / 7 / 38; 283 / 10 / 369; 1.89% |
| 3 (balanced) | 6x 0.25, 7x 0.5 | 93 / 7 / 37; 290 / 10 / 376; 1.89% |
| 2.7 | 2x 0.2, 8x 0.25, 3x 0.5 | 101 / 7 / 29; 308 / 10 / 407; 1.91% |
| 2 | 2x 0.2, 11x 0.25 | 110 / 7 / 20; 313 / 10 / 429; 1.91% |
| 1.5 | 4x 0.2, 9x 0.25 | 110 / 7 / 20; 314 / 10 / 431; 1.91% |
| 1.1 | 10x 0.2, 3x 0.25 | 110 / 7 / 20; 314 / 10 / 437; 1.91% |
| 1 (recall) | 10x 0.2, 3x 0.25 | 110 / 7 / 20; 314 / 10 / 437; 1.91% |
| 0.9 | 1x 0.1, 1x 0.15, 11x 0.2 | 113 / 8 / 16; 317 / 12 / 445; 2.01% |
| 0.5 | 4x 0.1, 1x 0.15, 8x 0.2 | 114 / 8 / 15; 319 / 12 / 461; 2.09% |

## 3. Parameter transfer: each clip with the numbers of every fold (fixed margin)

'held-out' = the clip's own fold (fitted without it); 'in-fit' = the 12 folds whose fit set contains the clip; all13 = the frozen numbers.

| margin | held-out pooled right / wrong | all13 pooled | in-fit folds: pooled right (min - max over a random fold per clip is not shown; per-clip sums of min / mean / max) | wrong (sum of per-clip min / mean / max) | clips whose result differs between any two folds |
|---|---|---|---|---|---|
| 0.2 | 112 / 7 | 112 / 7 | 112 / 112.3 / 114 | 7 / 7.2 / 8 | 3: ['SNGS-163', 'SNGS-169', 'SNGS-162'] |
| 0.25 | 110 / 7 | 109 / 6 | 108 / 109.4 / 111 | 6 / 6.2 / 7 | 4: ['SNGS-107', 'SNGS-169', 'SNGS-162', 'SNGS-151'] |
| 0.5 | 91 / 4 | 90 / 4 | 85 / 90.3 / 93 | 4 / 4.2 / 6 | 7: ['SNGS-107', 'SNGS-067', 'SNGS-066', 'SNGS-161', 'SNGS-169', 'SNGS-162', 'SNGS-151'] |
| 0.8 | 51 / 1 | 51 / 1 | 43 / 50.6 / 55 | 1 / 1.5 / 2 | 9: ['SNGS-106', 'SNGS-097', 'SNGS-067', 'SNGS-161', 'SNGS-169', 'SNGS-162', 'SNGS-151', 'SNGS-102', 'SNGS-112'] |
| 0.9 | 31 / 0 | 37 / 0 | 22 / 35.8 / 46 | 0 / 0.0 / 0 | 11: ['SNGS-107', 'SNGS-106', 'SNGS-097', 'SNGS-067', 'SNGS-066', 'SNGS-161', 'SNGS-169', 'SNGS-162', 'SNGS-151', 'SNGS-102', 'SNGS-112'] |

Held-out minus the mean of the 12 in-fit folds, summed over clips (negative right = held-out is worse = the in-sample numbers are optimistic):

- margin 0.2: right -0.3, wrong -0.2
- margin 0.25: right +0.6, wrong +0.8
- margin 0.5: right +0.7, wrong -0.2
- margin 0.8: right +0.4, wrong -0.5
- margin 0.9: right -4.8, wrong +0.0

## 4. Pair model refitted here for two folds (weak labels of the 12 fit clips)

- fold SNGS-162: refit weights [8.2555, 0.6599, 1.1785], bias -7.2965, 15430 pairs (681 positive); largest difference to params_loco.json 0.00e+00; rows of the held-out clip in the fit table: False
- fold SNGS-107: refit weights [7.2372, 0.6783, 0.9558], bias -7.1499, 17724 pairs (733 positive); largest difference to params_loco.json 0.00e+00; rows of the held-out clip in the fit table: False
