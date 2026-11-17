# r1: reproduction of fuse_r1's numbers

| file | score.evaluate here | claimed | same |
|---|---|---|---|
| METHOD_strict.json | 67 / 3 / 67 | 312 / 7 / 25 | 1.61% (0.01611) | (67, 3, 67, 312, 7, 25, 0.01611) | True |
| METHOD_balanced.json | 85 / 5 / 47 | 378 / 16 / 30 | 2.12% (0.02123) | (85, 5, 47, 378, 16, 30, 0.02123) | True |
| METHOD_recall.json | 106 / 7 / 24 | 414 / 17 / 260 | 2.17% (0.02175) | (106, 7, 24, 414, 17, 260, 0.02175) | True |
| METHOD_fixed_strict.json | 70 / 2 / 65 | 319 / 6 / 22 | 1.44% (0.01445) | (70, 2, 65, 319, 6, 22, 0.01445) | True |
| METHOD_fixed_balanced.json | 85 / 4 / 48 | 361 / 12 / 30 | 1.81% (0.01813) | (85, 4, 48, 361, 12, 30, 0.01813) | True |
| METHOD_fixed_recall.json | 112 / 7 / 18 | 415 / 17 / 311 | 2.18% (0.02177) | (112, 7, 18, 415, 17, 311, 0.02177) | True |
| METHOD_frozen_strict.json | 69 / 2 / 66 | 315 / 7 / 23 | 1.52% (0.0152) | (69, 2, 66, 315, 7, 23, 0.0152) | True |
| METHOD_frozen_balanced.json | 86 / 4 / 47 | 362 / 12 / 29 | 1.81% (0.01814) | (86, 4, 47, 362, 12, 29, 0.01814) | True |
| METHOD_frozen_recall.json | 112 / 7 / 18 | 415 / 17 / 311 | 2.18% (0.02178) | (112, 7, 18, 415, 17, 311, 0.02178) | True |
| METHOD_cuts_only.json | 0 / 0 / 137 | 0 / 0 / 0 | 1.00% (0.00999) | (0, 0, 137, 0, 0, 0, 0.00999) | True |
| baseline split | 0 / 4 / 133 | 0 / 0 / 0 | 2.61% | (0, 4, 133) | True |
| baseline plugin | 30 / 14 / 93 | 33 / 10 / 50 | 3.08% | (30, 14, 93) | True |
| baseline mutual | 26 / 6 / 105 | 60 / 7 / 2 | 3.03% | (26, 6, 105) | True |

Rebuild through fuse_r1's apply path (method.build / method.cuts_only) from params_loco.json, compared with the files:

- METHOD_strict.json: 344 links, 221 cuts rebuilt; clips that differ: none
- METHOD_balanced.json: 424 links, 221 cuts rebuilt; clips that differ: none
- METHOD_recall.json: 691 links, 221 cuts rebuilt; clips that differ: none
- METHOD_fixed_strict.json: 347 links, 221 cuts rebuilt; clips that differ: none
- METHOD_fixed_balanced.json: 403 links, 221 cuts rebuilt; clips that differ: none
- METHOD_fixed_recall.json: 743 links, 221 cuts rebuilt; clips that differ: none
- METHOD_frozen_strict.json: 345 links, 219 cuts rebuilt; clips that differ: none
- METHOD_frozen_balanced.json: 403 links, 219 cuts rebuilt; clips that differ: none
- METHOD_frozen_recall.json: 743 links, 219 cuts rebuilt; clips that differ: none
- METHOD_cuts_only.json: 0 links, 221 cuts rebuilt; clips that differ: none
