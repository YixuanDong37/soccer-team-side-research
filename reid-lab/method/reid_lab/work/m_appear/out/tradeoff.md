137 scored returns in the 13 dev clips (30 more are undetected). lam = price of one wrong identity
(extra wrong return or wrong link) in recovered returns, used to pick the parameters.

| method | right | wrong | lost | links right / wrong / unknown | link precision | impure rows |
|---|---|---|---|---|---|---|
| **baselines (reid_lab/score.py)** | | | | | | |
| gbf now: split, no links | 0 | 4 | 133 | 0 / 0 / 0 | - | 2.61 % |
| plugin: keep every plugin join | 30 | 14 | 93 | 33 / 10 / 50 | 0.767 | 3.08 % |
| mutual: appearance_match.py pairs, margin 0.05 | 26 | 6 | 105 | 60 / 7 / 2 | 0.895 | 3.03 % |
| **HELD-OUT (leave one clip out): agglomeration; kind, t_abs, margin picked on the other 12 clips** | | | | | | |
| agglo lam=1  (recall; not recommended) | 87 | 16 | 34 | 281 / 16 / 248 | 0.946 | 3.75 % |
| agglo lam=2  (balanced) | 74 | 7 | 56 | 239 / 8 / 197 | 0.968 | 3.10 % |
| agglo lam=3  (cautious) | 67 | 6 | 64 | 226 / 3 / 187 | 0.987 | 2.86 % |
| agglo lam=5 | 67 | 6 | 64 | 214 / 3 / 180 | 0.986 | 2.86 % |
| agglo lam=10 (strict) | 60 | 6 | 71 | 201 / 3 / 165 | 0.985 | 2.83 % |
| **HELD-OUT: agglomeration, only the margin picked on the other 12 clips (kind w, t_abs 0.5 fixed after seeing all 13)** | | | | | | |
| agglo-w-0.5 lam=1 | 78 | 14 | 45 | 260 / 14 / 220 | 0.949 | 3.57 % |
| agglo-w-0.5 lam=2 | 73 | 6 | 58 | 236 / 2 / 192 | 0.992 | 2.80 % |
| agglo-w-0.5 lam=3 | 71 | 6 | 60 | 232 / 2 / 189 | 0.992 | 2.80 % |
| agglo-w-0.5 lam=5 and 10 | 71 | 6 | 60 | 220 / 2 / 176 | 0.991 | 2.80 % |
| **HELD-OUT: pairwise mutual best + rival margin on the same features (no chains)** | | | | | | |
| pairwise lam=1 | 71 | 9 | 57 | 234 / 13 / 130 | 0.947 | 3.32 % |
| pairwise lam=2 | 58 | 9 | 70 | 205 / 11 / 117 | 0.949 | 3.08 % |
| pairwise lam=3, 5, 10 | 46 | 8 | 83 | 161 / 9 / 89 | 0.947 | 3.03 % |
| **HELD-OUT: as the first block, with the rival rule (share >= 1 or >= 5 frames) also picked inside the loop** | | | | | | |
| agglo nested-covis lam=2 | 75 | 14 | 48 | 240 / 10 / 204 | 0.960 | 3.27 % |
| agglo nested-covis lam=3 | 70 | 13 | 54 | 232 / 8 / 202 | 0.967 | 3.20 % |
| **IN-SAMPLE (parameters fitted on all 13 clips; secondary): frozen operating points of method.py** | | | | | | |
| frozen recall   (w, t_abs 0.40, margin 0.15) | 89 | 7 | 41 | 273 / 10 / 244 | 0.965 | 3.41 % |
| frozen balanced (w, t_abs 0.50, margin 0.25) | 75 | 6 | 56 | 238 / 2 / 198 | 0.992 | 2.80 % |
| frozen cautious (w, t_abs 0.50, margin 0.30) | 71 | 6 | 60 | 220 / 1 / 174 | 0.996 | 2.76 % |
| frozen strict   (w, t_abs 0.50, margin 0.35) | 61 | 6 | 70 | 205 / 1 / 147 | 0.995 | 2.76 % |
