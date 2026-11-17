# r6: behaviour by gap length (diagnosis with labels)

Annotated returns: 167; with a gap over 10 s: 44 (26%). Gap quantiles (frames) 10/50/90%: [37.0, 141.0, 345.0]; longest 722.

## Returns: right / wrong / lost (+ undetected) per gap bucket

| method | 0.4-2 s (10-49 fr) | 2-5 s (50-124) | 5-10 s (125-250) | > 10 s (251+) |
|---|---|---|---|---|
| strict | 11 / 0 / 7 (+7) | 15 / 2 / 25 (+7) | 7 / 2 / 34 (+6) | 0 / 0 / 34 (+10) |
| balanced | 17 / 0 / 1 (+7) | 33 / 3 / 6 (+7) | 26 / 4 / 13 (+6) | 17 / 0 / 17 (+10) |
| recall | 17 / 0 / 1 (+7) | 37 / 3 / 2 (+7) | 31 / 4 / 8 (+6) | 25 / 0 / 9 (+10) |
| margin_0.50 | 17 / 0 / 1 (+7) | 33 / 2 / 7 (+7) | 24 / 2 / 17 (+6) | 17 / 0 / 17 (+10) |
| plugin | 9 / 0 / 9 (+7) | 14 / 6 / 22 (+7) | 7 / 5 / 31 (+6) | 0 / 3 / 31 (+10) |
| split | 0 / 0 / 18 (+7) | 0 / 2 / 40 (+7) | 0 / 1 / 42 (+6) | 0 / 1 / 33 (+10) |
| mutual | 4 / 1 / 13 (+7) | 11 / 2 / 29 (+7) | 11 / 2 / 30 (+6) | 0 / 1 / 33 (+10) |

Right share and wrong share of the scored returns per bucket:

| method | 0.4-2 s (10-49 fr) | 2-5 s (50-124) | 5-10 s (125-250) | > 10 s (251+) |
|---|---|---|---|---|
| strict | 0.61 right, 0.000 wrong (n 18) | 0.36 right, 0.048 wrong (n 42) | 0.16 right, 0.047 wrong (n 43) | 0.00 right, 0.000 wrong (n 34) |
| balanced | 0.94 right, 0.000 wrong (n 18) | 0.79 right, 0.071 wrong (n 42) | 0.60 right, 0.093 wrong (n 43) | 0.50 right, 0.000 wrong (n 34) |
| recall | 0.94 right, 0.000 wrong (n 18) | 0.88 right, 0.071 wrong (n 42) | 0.72 right, 0.093 wrong (n 43) | 0.74 right, 0.000 wrong (n 34) |
| margin_0.50 | 0.94 right, 0.000 wrong (n 18) | 0.79 right, 0.048 wrong (n 42) | 0.56 right, 0.047 wrong (n 43) | 0.50 right, 0.000 wrong (n 34) |
| plugin | 0.50 right, 0.000 wrong (n 18) | 0.33 right, 0.143 wrong (n 42) | 0.16 right, 0.116 wrong (n 43) | 0.00 right, 0.088 wrong (n 34) |
| split | 0.00 right, 0.000 wrong (n 18) | 0.00 right, 0.048 wrong (n 42) | 0.00 right, 0.023 wrong (n 43) | 0.00 right, 0.029 wrong (n 34) |
| mutual | 0.22 right, 0.056 wrong (n 18) | 0.26 right, 0.048 wrong (n 42) | 0.26 right, 0.047 wrong (n 43) | 0.00 right, 0.029 wrong (n 34) |

Long-gap (> 10 s) returns per clip, recall point: SNGS-067 {'right': 1}, SNGS-100 {'lost': 1}, SNGS-102 {'right': 3, 'lost': 2, 'undetected': 2}, SNGS-106 {'lost': 1, 'undetected': 1}, SNGS-107 {'right': 8, 'undetected': 1, 'lost': 1}, SNGS-151 {'undetected': 2, 'right': 2}, SNGS-161 {'right': 5}, SNGS-162 {'lost': 1}, SNGS-163 {'right': 5, 'undetected': 4, 'lost': 3}, SNGS-169 {'right': 1}

## Links by the frames between the two pieces

judged = score.py (both pieces >= 5 matched rows); weak = majority person of >= 1 matched row.

| point | bucket | links | judged right / wrong / unknown | judged precision | weak agree / disagree / none | weak precision |
|---|---|---|---|---|---|---|
| strict | < 0.4 s (< 10 fr) | 159 | 55 / 0 / 104 | 1.000 | 128 / 1 / 30 | 0.992 |
| strict | 0.4-2 s (10-49 fr) | 112 | 68 / 2 / 42 | 0.971 | 104 / 4 / 4 | 0.963 |
| strict | 2-5 s (50-124) | 31 | 22 / 2 / 7 | 0.917 | 27 / 2 / 2 | 0.931 |
| strict | 5-10 s (125-250) | 11 | 6 / 0 / 5 | 1.000 | 10 / 0 / 1 | 1.000 |
| strict | > 10 s (251+) | 1 | 1 / 0 / 0 | 1.000 | 1 / 0 / 0 | 1.000 |
| balanced | < 0.4 s (< 10 fr) | 288 | 75 / 0 / 213 | 1.000 | 222 / 11 / 55 | 0.953 |
| balanced | 0.4-2 s (10-49 fr) | 219 | 114 / 4 / 101 | 0.966 | 196 / 10 / 13 | 0.951 |
| balanced | 2-5 s (50-124) | 95 | 57 / 4 / 34 | 0.934 | 82 / 6 / 7 | 0.932 |
| balanced | 5-10 s (125-250) | 46 | 25 / 2 / 19 | 0.926 | 39 / 4 / 3 | 0.907 |
| balanced | > 10 s (251+) | 28 | 19 / 0 / 9 | 1.000 | 26 / 1 / 1 | 0.963 |
| recall | < 0.4 s (< 10 fr) | 326 | 85 / 0 / 241 | 1.000 | 249 / 12 / 65 | 0.954 |
| recall | 0.4-2 s (10-49 fr) | 228 | 115 / 4 / 109 | 0.966 | 198 / 11 / 19 | 0.947 |
| recall | 2-5 s (50-124) | 107 | 60 / 4 / 43 | 0.938 | 91 / 8 / 8 | 0.919 |
| recall | 5-10 s (125-250) | 60 | 31 / 2 / 27 | 0.939 | 50 / 6 / 4 | 0.893 |
| recall | > 10 s (251+) | 40 | 23 / 0 / 17 | 1.000 | 34 / 3 / 3 | 0.919 |
| margin_0.50 | < 0.4 s (< 10 fr) | 270 | 70 / 0 / 200 | 1.000 | 211 / 7 / 52 | 0.968 |
| margin_0.50 | 0.4-2 s (10-49 fr) | 205 | 108 / 3 / 94 | 0.973 | 187 / 8 / 10 | 0.959 |
| margin_0.50 | 2-5 s (50-124) | 82 | 55 / 3 / 24 | 0.948 | 73 / 3 / 6 | 0.961 |
| margin_0.50 | 5-10 s (125-250) | 39 | 20 / 1 / 18 | 0.952 | 35 / 2 / 2 | 0.946 |
| margin_0.50 | > 10 s (251+) | 28 | 19 / 0 / 9 | 1.000 | 26 / 1 / 1 | 0.963 |
