# v_leak round 1: choices repeated inside the folds (cells: fuse_r1/cache/loco_raw.pkl)

Cells: right / wrong / lost of the 137 scored returns, pooled over the 13 held-out clips. Limits: strict 4, balanced 7, recall 14 wrong.

Check: n_common.choose_multi with a single variant against fuse_r1's rules.choose (2 linkers x 4 rules x 3 points x 13 folds = 312 choices): 0 differ.

## A. The cap of rule B uses the number of scored returns of all 13 clips (137)

floor(K x scored_fit / 137) = floor(K x (1 - scored_heldout / 137)): the cap of a fold depends on how many scored returns its held-out clip has.
Caps per fold under fuse_r1's formula, and the held-out result with caps that do not use the held-out clip:

| cap | strict | balanced | recall |
|---|---|---|---|
| fuse_r1: floor(K x scored_fit / 137) | 67 / 3 / 67 (caps {3: 13}; 13 of 13 folds choose the same setting as rule B) | 85 / 5 / 47 (caps {6: 13}; 13 of 13 folds choose the same setting as rule B) | 106 / 7 / 24 (caps {12: 8, 13: 5}; 13 of 13 folds choose the same setting as rule B) |
| floor(K x 12 / 13) | 67 / 3 / 67 (caps {3: 13}; 13 of 13 folds choose the same setting as rule B) | 85 / 5 / 47 (caps {6: 13}; 13 of 13 folds choose the same setting as rule B) | 106 / 7 / 24 (caps {12: 13}; 13 of 13 folds choose the same setting as rule B) |
| K - 1 | 67 / 3 / 67 (caps {3: 13}; 13 of 13 folds choose the same setting as rule B) | 85 / 5 / 47 (caps {6: 13}; 13 of 13 folds choose the same setting as rule B) | 106 / 7 / 24 (caps {13: 13}; 13 of 13 folds choose the same setting as rule B) |
| K | 67 / 3 / 67 (caps {4: 13}; 13 of 13 folds choose the same setting as rule B) | 85 / 5 / 47 (caps {7: 13}; 13 of 13 folds choose the same setting as rule B) | 106 / 7 / 24 (caps {14: 13}; 13 of 13 folds choose the same setting as rule B) |
| no cap at all (part (i) of rule B removed) | 67 / 3 / 67 (caps none; 13 of 13 folds choose the same setting as rule B) | 85 / 5 / 47 (caps none; 13 of 13 folds choose the same setting as rule B) | 106 / 7 / 24 (caps none; 13 of 13 folds choose the same setting as rule B) |

## B. The rule for the operating points

fuse_r1 fixed rule B in PREREG.txt knowing round 0's held-out tables (there 'J without the worst fit clip' had given 88 / 4 / 45 post hoc).
Held-out result of each rule of the family PREREG.txt names, and of the rule CHOSEN INSIDE each fold:

| rule | strict | balanced | recall |
|---|---|---|---|
| B | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 |
| R0 | 26 / 2 / 109 | 72 / 4 / 61 | 97 / 8 / 32 |
| Rcap | 78 / 8 / 51 | 98 / 9 / 30 | 112 / 9 / 16 |
| Rnotrim | 26 / 2 / 109 | 72 / 4 / 61 | 97 / 8 / 32 |
| rule chosen inside the fold: most right with the inner held-out wrong returns within the cap | 71 / 8 / 58 (11x B, 1x R0, 1x Rcap) | 87 / 8 / 42 (10x B, 2x R0, 1x Rcap) | 112 / 9 / 16 (13x Rcap) |
| rule chosen inside the fold: largest inner held-out right - lam x errors | 46 / 2 / 89 (8x R0, 5x B) | 75 / 4 / 58 (11x R0, 2x B) | 105 / 7 / 25 (8x Rcap, 3x R0, 2x B) |

Inner results of the four rules in the fold that holds SNGS-162 out and in the fold that holds SNGS-107 out (pooled over the 12 inner held-out clips):

- SNGS-162, strict: B 66 / 1 / 52; R0 63 / 1 / 55; Rcap 107 / 3 / 9; Rnotrim 63 / 1 / 55
- SNGS-162, balanced: B 84 / 2 / 33; R0 81 / 2 / 36; Rcap 107 / 3 / 9; Rnotrim 81 / 2 / 36
- SNGS-162, recall: B 98 / 2 / 19; R0 94 / 2 / 23; Rcap 107 / 3 / 9; Rnotrim 94 / 2 / 23
- SNGS-107, strict: B 58 / 3 / 60; R0 22 / 2 / 97; Rcap 69 / 7 / 45; Rnotrim 22 / 2 / 97
- SNGS-107, balanced: B 73 / 5 / 43; R0 62 / 3 / 56; Rcap 87 / 7 / 27; Rnotrim 62 / 3 / 56
- SNGS-107, recall: B 85 / 8 / 28; R0 75 / 6 / 40; Rcap 96 / 8 / 17; Rnotrim 75 / 6 / 40

## C. Design choices offered to rule B inside each fold

r0 = round 0's linker (no tiers); n3 / n10 = solid from 3 / 10 boxes; c5 = a rival must share >= 5 frames with the partner (m_appear fixed '>= 1' after looking at all 13 clips).
Rule B picks the (variant, leaves, margin) on the 12 fit clips; ties go to the method's own variant.

| offered | strict | balanced | recall |
|---|---|---|---|
| r1 only (the method) | 67 / 3 / 67 (9x r1 0.60 core, 3x r1 0.70 core, 1x r1 0.50 core) | 85 / 5 / 47 (4x r1 0.20 core, 4x r1 0.40 core, 3x r1 0.50 core, 1x r1 0.25 core, 1x r1 0.15 core) | 106 / 7 / 24 (10x r1 0.20 +leaves, 1x r1 0.25 core, 1x r1 0.20 core, 1x r1 0.25 +leaves) |
| r1 or r0 (tiers or not) | 70 / 5 / 62 (8x r1 0.60 core, 3x r1 0.70 core, 1x r1 0.50 core, 1x r0 0.60 core) | 108 / 7 / 22 (10x r0 0.25 core, 3x r0 0.50 core) | 111 / 8 / 18 (8x r0 0.20 core, 4x r0 0.25 core, 1x r0 0.10 core) |

Each variant alone under rule B (held-out), and at the method's frozen settings (0.60 core / 0.40 core / 0.20 +leaves; held-out weights, setting not held-out):

| variant | strict | balanced | recall | frozen strict | frozen balanced | frozen recall |
|---|---|---|---|---|---|---|
| r1 | 67 / 3 / 67 (9x r1 0.60 core, 3x r1 0.70 core, 1x r1 0.50 core) | 85 / 5 / 47 (4x r1 0.20 core, 4x r1 0.40 core, 3x r1 0.50 core, 1x r1 0.25 core, 1x r1 0.15 core) | 106 / 7 / 24 (10x r1 0.20 +leaves, 1x r1 0.25 core, 1x r1 0.20 core, 1x r1 0.25 +leaves) | 70 / 2 / 65 | 85 / 4 / 48 | 112 / 7 / 18 |
| r0 | 51 / 4 / 82 (12x r0 0.80 core, 1x r0 0.60 core) | 108 / 7 / 22 (10x r0 0.25 core, 3x r0 0.50 core) | 111 / 8 / 18 (8x r0 0.20 core, 4x r0 0.25 core, 1x r0 0.10 core) | 78 / 4 / 55 | 97 / 6 / 34 | 112 / 7 / 18 |

## D. Rule B with another penalty or without the trimming (the method's variant r1; held-out)

| lam | with the worst fit clip left out of J (rule B) | no clip left out |
|---|---|---|
| 1 | strict 78 / 7 / 52; balanced 99 / 7 / 31; recall 106 / 7 / 24 | strict 78 / 7 / 52; balanced 97 / 8 / 32; recall 97 / 8 / 32 |
| 2 | strict 74 / 5 / 58; balanced 93 / 6 / 38; recall 93 / 6 / 38 | strict 74 / 5 / 58; balanced 78 / 5 / 54; recall 78 / 5 / 54 |
| 3 | strict 73 / 5 / 59; balanced 85 / 5 / 47; recall 85 / 5 / 47 | strict 72 / 4 / 61; balanced 72 / 4 / 61; recall 72 / 4 / 61 |
| 5 | strict 72 / 4 / 61; balanced 73 / 4 / 60; recall 73 / 4 / 60 | strict 35 / 2 / 100; balanced 35 / 2 / 100; recall 35 / 2 / 100 |
| 7 | strict 71 / 3 / 63; balanced 71 / 3 / 63; recall 71 / 3 / 63 | strict 26 / 2 / 109; balanced 26 / 2 / 109; recall 26 / 2 / 109 |
| 10 | strict 67 / 3 / 67; balanced 67 / 3 / 67; recall 67 / 3 / 67 | strict 26 / 2 / 109; balanced 26 / 2 / 109; recall 26 / 2 / 109 |
| 15 | strict 58 / 2 / 77; balanced 58 / 2 / 77; recall 58 / 2 / 77 | strict 25 / 2 / 110; balanced 25 / 2 / 110; recall 25 / 2 / 110 |
| 20 | strict 58 / 2 / 77; balanced 58 / 2 / 77; recall 58 / 2 / 77 | strict 25 / 2 / 110; balanced 25 / 2 / 110; recall 25 / 2 / 110 |

(The cap K of a point stays 4 / 7 / 14; fuse_r1 uses lam 10 / 3 / 1 for strict / balanced / recall.)

## E. Rule B leaves 'the fit clip with the most errors' out of J: is that SNGS-162 by another name?

Rule B against the same rule with SNGS-162 named (left out of J whenever it is a fit clip; nothing left out in the fold that holds it out):

| point | rule B | 'J without SNGS-162' | folds with the same setting | clip rule B leaves out, per fold |
|---|---|---|---|---|
| strict | 67 / 3 / 67 | 67 / 3 / 67 | 13 of 13 | 12x SNGS-162, 1x SNGS-112 |
| balanced | 85 / 5 / 47 | 85 / 5 / 47 | 12 of 13 | 12x SNGS-162, 1x SNGS-066 |
| recall | 106 / 7 / 24 | 106 / 7 / 24 | 13 of 13 | 12x SNGS-162, 1x SNGS-100 |

## F. Fixed settings, held-out weights: the 12 clips other than SNGS-162 against SNGS-162 alone

right / wrong / lost; leaves off. (No setting is chosen here; this shows on which clips a design variant differs from the method.)

| margin | r1: 12 other clips | r1: SNGS-162 | r0: 12 other clips | r0: SNGS-162 |
|---|---|---|---|---|
| 0.40 | 81 / 1 / 37 | 4 / 3 / 11 | 90 / 1 / 28 | 7 / 5 / 6 |
| 0.50 | 78 / 1 / 40 | 3 / 3 / 12 | 84 / 0 / 35 | 7 / 4 / 7 |
| 0.55 | 73 / 1 / 45 | 3 / 2 / 13 | 77 / 0 / 42 | 7 / 4 / 7 |
| 0.60 | 67 / 0 / 52 | 3 / 2 / 13 | 72 / 0 / 47 | 6 / 4 / 8 |
| 0.70 | 55 / 0 / 64 | 3 / 2 / 13 | 58 / 0 / 61 | 6 / 4 / 8 |
| 0.80 | 37 / 0 / 82 | 3 / 1 / 14 | 45 / 0 / 74 | 6 / 1 / 11 |

