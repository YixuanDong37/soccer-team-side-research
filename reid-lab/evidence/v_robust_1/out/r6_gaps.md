# r6: behaviour by gap length (diagnosis with labels)

Annotated returns: 167; with a gap over 10 s: 44 (26%). Gap quantiles (frames) 10/50/90%: [37.0, 141.0, 345.0]; longest 722.

## Returns: right / wrong / lost (+ undetected) per gap bucket

| method | 0.4-2 s (10-49 fr) | 2-5 s (50-124) | 5-10 s (125-250) | > 10 s (251+) |
|---|---|---|---|---|
| strict | 14 / 0 / 4 (+7) | 28 / 1 / 13 (+7) | 14 / 2 / 27 (+6) | 11 / 0 / 23 (+10) |
| balanced | 15 / 1 / 2 (+7) | 32 / 1 / 9 (+7) | 22 / 3 / 18 (+6) | 16 / 0 / 18 (+10) |
| recall | 17 / 1 / 0 (+7) | 36 / 3 / 3 (+7) | 31 / 3 / 9 (+6) | 22 / 0 / 12 (+10) |
| fixed_strict | 15 / 0 / 3 (+7) | 28 / 0 / 14 (+7) | 14 / 2 / 27 (+6) | 13 / 0 / 21 (+10) |
| fixed_balanced | 15 / 1 / 2 (+7) | 31 / 1 / 10 (+7) | 23 / 2 / 18 (+6) | 16 / 0 / 18 (+10) |
| fixed_recall | 17 / 1 / 0 (+7) | 37 / 3 / 2 (+7) | 32 / 3 / 8 (+6) | 26 / 0 / 8 (+10) |
| plugin | 9 / 0 / 9 (+7) | 14 / 6 / 22 (+7) | 7 / 5 / 31 (+6) | 0 / 3 / 31 (+10) |
| split | 0 / 0 / 18 (+7) | 0 / 2 / 40 (+7) | 0 / 1 / 42 (+6) | 0 / 1 / 33 (+10) |
| mutual | 4 / 1 / 13 (+7) | 11 / 2 / 29 (+7) | 11 / 2 / 30 (+6) | 0 / 1 / 33 (+10) |

Right share and wrong share of the scored returns per bucket:

| method | 0.4-2 s (10-49 fr) | 2-5 s (50-124) | 5-10 s (125-250) | > 10 s (251+) |
|---|---|---|---|---|
| strict | 0.78 right, 0.000 wrong (n 18) | 0.67 right, 0.024 wrong (n 42) | 0.33 right, 0.047 wrong (n 43) | 0.32 right, 0.000 wrong (n 34) |
| balanced | 0.83 right, 0.056 wrong (n 18) | 0.76 right, 0.024 wrong (n 42) | 0.51 right, 0.070 wrong (n 43) | 0.47 right, 0.000 wrong (n 34) |
| recall | 0.94 right, 0.056 wrong (n 18) | 0.86 right, 0.071 wrong (n 42) | 0.72 right, 0.070 wrong (n 43) | 0.65 right, 0.000 wrong (n 34) |
| fixed_strict | 0.83 right, 0.000 wrong (n 18) | 0.67 right, 0.000 wrong (n 42) | 0.33 right, 0.047 wrong (n 43) | 0.38 right, 0.000 wrong (n 34) |
| fixed_balanced | 0.83 right, 0.056 wrong (n 18) | 0.74 right, 0.024 wrong (n 42) | 0.53 right, 0.047 wrong (n 43) | 0.47 right, 0.000 wrong (n 34) |
| fixed_recall | 0.94 right, 0.056 wrong (n 18) | 0.88 right, 0.071 wrong (n 42) | 0.74 right, 0.070 wrong (n 43) | 0.76 right, 0.000 wrong (n 34) |
| plugin | 0.50 right, 0.000 wrong (n 18) | 0.33 right, 0.143 wrong (n 42) | 0.16 right, 0.116 wrong (n 43) | 0.00 right, 0.088 wrong (n 34) |
| split | 0.00 right, 0.000 wrong (n 18) | 0.00 right, 0.048 wrong (n 42) | 0.00 right, 0.023 wrong (n 43) | 0.00 right, 0.029 wrong (n 34) |
| mutual | 0.22 right, 0.056 wrong (n 18) | 0.26 right, 0.048 wrong (n 42) | 0.26 right, 0.047 wrong (n 43) | 0.00 right, 0.029 wrong (n 34) |

Long-gap (> 10 s) scored returns per clip, strict: SNGS-067 {'right': 1}, SNGS-100 {'lost': 1}, SNGS-102 {'right': 3, 'lost': 2}, SNGS-106 {'lost': 1}, SNGS-107 {'lost': 8, 'right': 1}, SNGS-151 {'lost': 1, 'right': 1}, SNGS-161 {'right': 3, 'lost': 2}, SNGS-162 {'lost': 1}, SNGS-163 {'lost': 7, 'right': 1}, SNGS-169 {'right': 1}
Long-gap (> 10 s) scored returns per clip, balanced: SNGS-067 {'right': 1}, SNGS-100 {'lost': 1}, SNGS-102 {'right': 3, 'lost': 2}, SNGS-106 {'lost': 1}, SNGS-107 {'lost': 7, 'right': 2}, SNGS-151 {'lost': 1, 'right': 1}, SNGS-161 {'right': 5}, SNGS-162 {'lost': 1}, SNGS-163 {'lost': 5, 'right': 3}, SNGS-169 {'right': 1}
Long-gap (> 10 s) scored returns per clip, recall: SNGS-067 {'right': 1}, SNGS-100 {'lost': 1}, SNGS-102 {'right': 3, 'lost': 2}, SNGS-106 {'lost': 1}, SNGS-107 {'right': 5, 'lost': 4}, SNGS-151 {'right': 2}, SNGS-161 {'right': 5}, SNGS-162 {'lost': 1}, SNGS-163 {'right': 5, 'lost': 3}, SNGS-169 {'right': 1}
Long-gap (> 10 s) scored returns per clip, plugin: SNGS-067 {'lost': 1}, SNGS-100 {'lost': 1}, SNGS-102 {'lost': 5}, SNGS-106 {'wrong': 1}, SNGS-107 {'lost': 8, 'wrong': 1}, SNGS-151 {'lost': 2}, SNGS-161 {'lost': 5}, SNGS-162 {'wrong': 1}, SNGS-163 {'lost': 8}, SNGS-169 {'lost': 1}
Gaps over 20 s (scored), strict: {'lost': 3}; longest gap scored right: 490 frames
Gaps over 20 s (scored), balanced: {'lost': 2, 'right': 1}; longest gap scored right: 591 frames
Gaps over 20 s (scored), recall: {'lost': 2, 'right': 1}; longest gap scored right: 591 frames

## Links by the frames between the two pieces

judged = score.py (both pieces >= 5 matched rows); weak = majority person of >= 1 matched row.

| point | bucket | links | judged right / wrong / unknown | judged precision | weak agree / disagree / none | weak precision |
|---|---|---|---|---|---|---|
| strict | < 0.4 s (< 10 fr) | 87 | 75 / 0 / 12 | 1.000 | 83 / 2 / 2 | 0.976 |
| strict | 0.4-2 s (10-49 fr) | 143 | 131 / 3 / 9 | 0.978 | 137 / 3 / 3 | 0.979 |
| strict | 2-5 s (50-124) | 70 | 64 / 3 / 3 | 0.955 | 67 / 3 / 0 | 0.957 |
| strict | 5-10 s (125-250) | 30 | 28 / 1 / 1 | 0.966 | 28 / 1 / 1 | 0.966 |
| strict | > 10 s (251+) | 14 | 14 / 0 / 0 | 1.000 | 14 / 0 / 0 | 1.000 |
| balanced | < 0.4 s (< 10 fr) | 98 | 83 / 0 / 15 | 1.000 | 94 / 2 / 2 | 0.979 |
| balanced | 0.4-2 s (10-49 fr) | 163 | 148 / 6 / 9 | 0.961 | 154 / 6 / 3 | 0.963 |
| balanced | 2-5 s (50-124) | 87 | 78 / 5 / 4 | 0.940 | 81 / 5 / 1 | 0.942 |
| balanced | 5-10 s (125-250) | 51 | 45 / 5 / 1 | 0.900 | 45 / 5 / 1 | 0.900 |
| balanced | > 10 s (251+) | 25 | 24 / 0 / 1 | 1.000 | 25 / 0 / 0 | 1.000 |
| recall | < 0.4 s (< 10 fr) | 224 | 95 / 0 / 129 | 1.000 | 193 / 7 / 24 | 0.965 |
| recall | 0.4-2 s (10-49 fr) | 229 | 150 / 6 / 73 | 0.962 | 205 / 10 / 14 | 0.953 |
| recall | 2-5 s (50-124) | 115 | 83 / 6 / 26 | 0.933 | 102 / 8 / 5 | 0.927 |
| recall | 5-10 s (125-250) | 77 | 51 / 5 / 21 | 0.911 | 62 / 12 / 3 | 0.838 |
| recall | > 10 s (251+) | 46 | 35 / 0 / 11 | 1.000 | 41 / 3 / 2 | 0.932 |
| fixed_strict | < 0.4 s (< 10 fr) | 87 | 76 / 0 / 11 | 1.000 | 83 / 2 / 2 | 0.976 |
| fixed_strict | 0.4-2 s (10-49 fr) | 145 | 134 / 3 / 8 | 0.978 | 139 / 3 / 3 | 0.979 |
| fixed_strict | 2-5 s (50-124) | 68 | 64 / 2 / 2 | 0.970 | 66 / 2 / 0 | 0.971 |
| fixed_strict | 5-10 s (125-250) | 30 | 28 / 1 / 1 | 0.966 | 28 / 1 / 1 | 0.966 |
| fixed_strict | > 10 s (251+) | 17 | 17 / 0 / 0 | 1.000 | 17 / 0 / 0 | 1.000 |
| fixed_balanced | < 0.4 s (< 10 fr) | 93 | 77 / 1 / 15 | 0.987 | 88 / 3 / 2 | 0.967 |
| fixed_balanced | 0.4-2 s (10-49 fr) | 158 | 145 / 4 / 9 | 0.973 | 151 / 4 / 3 | 0.974 |
| fixed_balanced | 2-5 s (50-124) | 80 | 72 / 4 / 4 | 0.947 | 75 / 4 / 1 | 0.949 |
| fixed_balanced | 5-10 s (125-250) | 48 | 44 / 3 / 1 | 0.936 | 44 / 3 / 1 | 0.936 |
| fixed_balanced | > 10 s (251+) | 24 | 23 / 0 / 1 | 1.000 | 24 / 0 / 0 | 1.000 |
| fixed_recall | < 0.4 s (< 10 fr) | 244 | 95 / 0 / 149 | 1.000 | 207 / 7 / 30 | 0.967 |
| fixed_recall | 0.4-2 s (10-49 fr) | 244 | 150 / 6 / 88 | 0.962 | 217 / 10 / 17 | 0.956 |
| fixed_recall | 2-5 s (50-124) | 125 | 83 / 6 / 36 | 0.933 | 109 / 8 / 8 | 0.932 |
| fixed_recall | 5-10 s (125-250) | 79 | 52 / 5 / 22 | 0.912 | 63 / 12 / 4 | 0.840 |
| fixed_recall | > 10 s (251+) | 51 | 35 / 0 / 16 | 1.000 | 46 / 3 / 2 | 0.939 |
