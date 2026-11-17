# What score.py's three outcomes hide (diagnosis, 137 scored returns)

| method | right: clean | right: via an earlier stretch of the same person | right: mixed (group also holds >= 5 rows of another person) | right: swallowed (group majority is another person) | wrong | lost: hidden wrong, joined by a LINK to 1-4 rows of another person | lost: hidden wrong, 1-4 rows of another person in B's own piece (no link involved) | lost: new identity, but its group holds >= 5 rows of another person later on | lost: new identity |
|---|---|---|---|---|---|---|---|---|---|
| baseline split | 0 | 0 | 0 | 0 | 4 | 0 | 4 | 6 | 123 |
| baseline plugin | 27 | 1 | 1 | 1 | 14 | 3 | 2 | 6 | 82 |
| baseline mutual | 19 | 5 | 2 | 0 | 6 | 0 | 4 | 5 | 96 |
| fuse strict (held-out) | 31 | 2 | 0 | 0 | 4 | 0 | 2 | 2 | 96 |
| fuse balanced (held-out) | 82 | 8 | 3 | 0 | 7 | 0 | 2 | 1 | 34 |
| fuse recall (held-out) | 93 | 11 | 6 | 0 | 7 | 0 | 1 | 0 | 19 |
| fuse margin_0.50 (held-out) | 81 | 8 | 2 | 0 | 4 | 0 | 2 | 1 | 39 |

Per clip, fuse recall point: 107: mixed 0, swallowed 0, wrong 0, hidden 0; 163: mixed 2, swallowed 0, wrong 0, hidden 0; 106: mixed 0, swallowed 0, wrong 0, hidden 0; 097: mixed 0, swallowed 0, wrong 0, hidden 0; 067: mixed 0, swallowed 0, wrong 0, hidden 0; 100: mixed 0, swallowed 0, wrong 0, hidden 0; 066: mixed 1, swallowed 0, wrong 1, hidden 0; 161: mixed 0, swallowed 0, wrong 0, hidden 0; 169: mixed 0, swallowed 0, wrong 0, hidden 0; 162: mixed 2, swallowed 0, wrong 6, hidden 0; 151: mixed 1, swallowed 0, wrong 0, hidden 0; 102: mixed 0, swallowed 0, wrong 0, hidden 0; 112: mixed 0, swallowed 0, wrong 0, hidden 0
