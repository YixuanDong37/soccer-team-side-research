# Blindfolded re-run of every LOCO fold against fuse's published files

| fold | pair model max abs diff | margins equal | links at the 3 points equal | cuts equal | held-out links equal at all 16 margins | fit-clip label files opened | held-out labels opened | fitted files of other agents read | blocked attempts |
|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-163 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-106 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-097 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-067 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-100 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-066 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-161 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-169 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-162 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-151 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-102 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |
| SNGS-112 | 0.0e+00 | True | True | True | True | 12 | False | 0 | 0 |

| point | blind re-run (score.evaluate) | fuse's METHOD file |
|---|---|---|
| strict | 33 / 4 / 100 | 152 / 4 / 158 | 1.28% | 33 / 4 / 100 | 152 / 4 / 158 | 1.28% |
| balanced | 93 / 7 / 37 | 290 / 10 / 376 | 1.89% | 93 / 7 / 37 | 290 / 10 / 376 | 1.89% |
| recall | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% | 110 / 7 / 20 | 314 / 10 / 437 | 1.91% |

| fixed margin | blind re-run, held-out |
|---|---|
| 0.1 | 116 / 9 / 12 | 330 / 15 / 482 | 2.31% |
| 0.15 | 114 / 7 / 16 | 326 / 12 / 459 | 2.19% |
| 0.2 | 112 / 7 / 18 | 316 / 10 / 438 | 1.91% |
| 0.25 | 110 / 7 / 20 | 315 / 10 / 424 | 1.91% |
| 0.3 | 105 / 6 / 26 | 308 / 10 / 403 | 1.91% |
| 0.35 | 104 / 6 / 27 | 301 / 11 / 392 | 1.86% |
| 0.4 | 97 / 6 / 34 | 288 / 11 / 369 | 1.84% |
| 0.45 | 94 / 5 / 38 | 278 / 9 / 356 | 1.66% |
| 0.5 | 91 / 4 / 42 | 272 / 7 / 345 | 1.51% |
| 0.55 | 84 / 4 / 49 | 262 / 6 / 326 | 1.42% |
| 0.6 | 78 / 4 / 55 | 254 / 5 / 304 | 1.41% |
| 0.7 | 64 / 4 / 69 | 222 / 4 / 263 | 1.29% |
| 0.8 | 51 / 1 / 85 | 196 / 3 / 216 | 1.20% |
| 0.9 | 31 / 0 / 106 | 145 / 2 / 147 | 1.05% |
| 1.0 | 21 / 0 / 116 | 101 / 2 / 85 | 1.05% |
| 1.2 | 2 / 0 / 135 | 16 / 0 / 19 | 1.00% |
