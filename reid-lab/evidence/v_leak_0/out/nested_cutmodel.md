# Nested choice of m_cuts' feature set (inner LOCO log-loss on the 12 fit clips of each fold)

| fold | chosen inside the fold | inner log-loss: chosen / m_cuts' set | held-out cuts: nested model | fuse's cuts | in both | only nested | only fuse |
|---|---|---|---|---|---|---|---|
| SNGS-107 | lr(cos,md,mr,lgap,ld) | 0.1610 / 0.1616 | 11 | 10 | 10 | 1 | 0 |
| SNGS-163 | lr(cos,md,mr,lgap,ld) | 0.1438 / 0.1450 | 20 | 20 | 19 | 1 | 1 |
| SNGS-106 | lr(cos,md,mr,lgap,ld) | 0.1487 / 0.1500 | 20 | 20 | 20 | 0 | 0 |
| SNGS-097 | lr(cos,md,mr,lgap,ld) | 0.1591 / 0.1598 | 6 | 6 | 6 | 0 | 0 |
| SNGS-067 | lr(cos,mm,lgap,ld) | 0.1442 / 0.1442 | 30 | 30 | 30 | 0 | 0 |
| SNGS-100 | lr(cos,mm,lgap,ld) | 0.1474 / 0.1474 | 25 | 25 | 25 | 0 | 0 |
| SNGS-066 | lr(cos,md,mr,lgap,ld) | 0.1275 / 0.1278 | 21 | 22 | 21 | 0 | 1 |
| SNGS-161 | lr(cos,md,mr,lgap,ld) | 0.1580 / 0.1584 | 6 | 6 | 6 | 0 | 0 |
| SNGS-169 | lr(cos,md,mr,lgap,ld) | 0.1495 / 0.1504 | 25 | 27 | 25 | 0 | 2 |
| SNGS-162 | lr(cos,md,mr,lgap,ld) | 0.1552 / 0.1555 | 32 | 32 | 32 | 0 | 0 |
| SNGS-151 | lr(cos,md,mr,lgap,ld) | 0.1551 / 0.1556 | 12 | 12 | 12 | 0 | 0 |
| SNGS-102 | lr(cos,md,mr,lgap,ld) | 0.1597 / 0.1604 | 5 | 5 | 5 | 0 | 0 |
| SNGS-112 | lr(cos,md,mr,lgap,ld) | 0.1615 / 0.1621 | 6 | 6 | 6 | 0 | 0 |
| **all** | 2 of 13 folds choose m_cuts' set | | | 221 | 217 | 2 | 4 |
