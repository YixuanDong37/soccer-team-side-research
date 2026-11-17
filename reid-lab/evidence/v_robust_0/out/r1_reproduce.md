# r1: reproduction of fuse's numbers

| file | score.evaluate here | claimed | same |
|---|---|---|---|
| METHOD_strict.json | 33 / 4 / 100 | 152 / 4 / 158 | 1.28% | (33, 4, 100, 152, 4, 158, 0.01284) | True |
| METHOD_balanced.json | 93 / 7 / 37 | 290 / 10 / 376 | 1.89% | (93, 7, 37, 290, 10, 376, 0.01887) | True |
| METHOD_recall.json | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | (110, 7, 20, 314, 10, 437, 0.01909) | True |
| METHOD_margin_0.50.json | 91 / 4 / 42 | 272 / 7 / 345 | 1.51% | (91, 4, 42, 272, 7, 345, 0.01511) | True |
| METHOD_frozen_strict.json | 37 / 0 / 100 | 152 / 2 / 152 | 1.05% | (37, 0, 100, 152, 2, 152, 0.01052) | True |
| METHOD_frozen_balanced.json | 109 / 6 / 22 | 310 / 10 / 424 | 1.91% | (109, 6, 22, 310, 10, 424, 0.01909) | True |
| METHOD_frozen_recall.json | 112 / 7 / 18 | 314 / 11 / 441 | 2.13% | (112, 7, 18, 314, 11, 441, 0.02132) | True |
| baseline split | 0 / 4 / 133 | 0 / 0 / 0 | 2.61% | | |
| baseline plugin | 30 / 14 / 93 | 33 / 10 / 50 | 3.08% | | |
| baseline mutual | 26 / 6 / 105 | 60 / 7 / 2 | 3.03% | | |
| cuts_only_heldout (cuts of METHOD_recall, no links) | 0 / 0 / 137 | 0 / 0 / 0 | 1.00% | | |
| cuts_only_frozen (cuts of METHOD_frozen_recall, no links) | 0 / 0 / 137 | 0 / 0 / 0 | 1.00% | | |

Rebuild through fuse's apply path (method.build) from params_loco.json, compared with the files:

- METHOD_strict.json: 314 links, 221 cuts rebuilt; clips that differ: none
- METHOD_balanced.json: 676 links, 221 cuts rebuilt; clips that differ: none
- METHOD_recall.json: 761 links, 221 cuts rebuilt; clips that differ: none
- METHOD_margin_0.50.json: 624 links, 221 cuts rebuilt; clips that differ: none
- METHOD_frozen_strict.json: 306 links, 219 cuts rebuilt; clips that differ: none
- METHOD_frozen_balanced.json: 744 links, 219 cuts rebuilt; clips that differ: none
- METHOD_frozen_recall.json: 766 links, 219 cuts rebuilt; clips that differ: none
