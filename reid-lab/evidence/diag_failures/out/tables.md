## T1 pooled scores
| what | right | wrong | lost | links | links right/wrong/unknown | impure rows % |
|---|---|---|---|---|---|---|
| split = gbf now (no links) | 0 | 4 | 133 | 0 | 0/0/0 | 2.61 |
| plugin (keep every plugin join) | 30 | 14 | 93 | 93 | 33/10/50 | 3.08 |
| mutual (appearance, margin 0.05) | 26 | 6 | 105 | 69 | 60/7/2 | 3.03 |
| ORACLE: perfect choice among appearance_match candidate pairs | 51 | 4 | 82 | 202 | 202/0/0 | 2.61 |
| ORACLE: link all pieces of known identity (>= 5 matched rows) | 105 | 7 | 25 | 505 | 505/0/0 | 2.61 |
| ORACLE: ... plus pieces with 1-4 matched rows | 128 | 8 | 1 | 775 | 398/0/377 | 2.61 |
| ORACLE: cut switches (runs >= 5 rows) + link known pieces | 108 | 3 | 26 | 583 | 583/0/0 | 0.53 |
| ORACLE: cut switches (runs >= 5 rows) + link every piece | 131 | 5 | 1 | 853 | 468/0/385 | 0.53 |
| ORACLE: cut at every change of person + link every piece | 137 | 0 | 0 | 1408 | 402/0/1006 | 0 |
| PROBE (label-free): cut every piece at every hole >= 10 frames, no links (599 cuts) | 0 | 0 | 137 | 0 | 0/0/0 | 0.76 |
| PROBE (label-free): cut every piece at every hole >= 15 frames, no links (383 cuts) | 0 | 0 | 137 | 0 | 0/0/0 | 0.98 |
| PROBE (label-free): cut every piece at every hole >= 25 frames, no links (238 cuts) | 0 | 0 | 137 | 0 | 0/0/0 | 1.41 |
| PROBE (label-free): cut every piece at every hole >= 50 frames, no links (152 cuts) | 0 | 0 | 137 | 0 | 0/0/0 | 1.77 |
| ORACLE on the H=10 hole cuts: links between pieces of known identity (>= 5 matched rows) | 108 | 1 | 28 | 739 | 739/0/0 | 0.76 |
| ORACLE on the H=10 hole cuts: links of every piece with a matched row | 134 | 3 | 0 | 1311 | 484/0/827 | 0.76 |

## T2 per clip
| clip | returns | scored | undetected | plugin R | plugin W | mutual R | mutual W | split W | oracle known R | oracle +fragments R | B row in fragment | B row in foreign piece | label suspects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 18 | 16 | 2 | 1 | 2 | 3 | 1 | 1 | 11 | 14 | 3 | 2 | 0 |
| SNGS-163 | 16 | 11 | 5 | 1 | 0 | 1 | 0 | 0 | 9 | 11 | 2 | 1 | 0 |
| SNGS-106 | 8 | 7 | 1 | 4 | 2 | 2 | 0 | 0 | 4 | 6 | 2 | 1 | 0 |
| SNGS-097 | 10 | 6 | 4 | 3 | 0 | 1 | 0 | 0 | 5 | 6 | 1 | 0 | 0 |
| SNGS-067 | 13 | 10 | 3 | 1 | 1 | 0 | 0 | 0 | 8 | 10 | 2 | 0 | 0 |
| SNGS-100 | 5 | 4 | 1 | 1 | 0 | 3 | 0 | 0 | 3 | 4 | 0 | 0 | 0 |
| SNGS-066 | 13 | 11 | 2 | 1 | 3 | 0 | 1 | 1 | 8 | 10 | 2 | 1 | 0 |
| SNGS-161 | 16 | 14 | 2 | 6 | 0 | 4 | 0 | 0 | 12 | 14 | 2 | 0 | 0 |
| SNGS-169 | 16 | 15 | 1 | 4 | 1 | 4 | 0 | 0 | 13 | 14 | 1 | 1 | 0 |
| SNGS-162 | 20 | 18 | 2 | 2 | 5 | 1 | 4 | 2 | 9 | 14 | 8 | 2 | 4 |
| SNGS-151 | 14 | 9 | 5 | 2 | 0 | 1 | 0 | 0 | 7 | 9 | 2 | 0 | 0 |
| SNGS-102 | 14 | 12 | 2 | 2 | 0 | 4 | 0 | 0 | 12 | 12 | 0 | 0 | 0 |
| SNGS-112 | 4 | 4 | 0 | 2 | 0 | 2 | 0 | 0 | 4 | 4 | 0 | 0 | 0 |
| all | 167 | 137 | 30 | 30 | 14 | 26 | 6 | 4 | 105 | 128 | 25 | 8 | 4 |

## T3 ceiling classes (167 returns)
| ceiling_class | returns |
|---|---|
| linkable: existing pieces, known identity | 105 |
| undetected (detection) | 30 |
| linkable only through a piece with < 5 matched rows | 23 |
| needs a cut around a stray run of < 5 rows | 6 |
| needs an in-track switch fix (cut of a run >= 5 rows) | 3 |

## T3b ceiling class x outcome now
| ceiling_class | plugin lost | plugin right | plugin wrong | mutual lost | mutual right | mutual wrong |
|---|---|---|---|---|---|---|
| linkable only through a piece with < 5 matched rows | 16 | 3 | 4 | 23 | 0 | 0 |
| linkable: existing pieces, known identity | 74 | 26 | 5 | 76 | 26 | 3 |
| needs a cut around a stray run of < 5 rows | 2 | 0 | 4 | 4 | 0 | 2 |
| needs an in-track switch fix (cut of a run >= 5 rows) | 1 | 1 | 1 | 2 | 0 | 1 |

## T4 piece under row A x piece under row B (137 scored)
| row A in | foreign | fragment | own | All |
|---|---|---|---|---|
| foreign | 1 | 2 | 5 | 8 |
| fragment | 2 | 3 | 13 | 18 |
| own | 4 | 18 | 89 | 111 |
| All | 7 | 23 | 107 | 137 |

## T4b same plugin id on both sides
| A and B share the plugin id | lost | right | wrong | All |
|---|---|---|---|---|
| False | 93 | 1 | 14 | 108 |
| True | 0 | 29 | 0 | 29 |
| All | 93 | 30 | 14 | 137 |

## T5 by gap length (all 167, annotated gap)
| gap | returns | undetected | scored | plugin R | plugin W | mutual R | mutual W | oracle known R | row pair is a candidate | own pieces on both sides | true pair beats every other person | ... by >= 0.05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <= 1 s | 9 | 3 | 6 | 3 | 0 | 1 | 0 | 5 | 3 | 8 | 8 | 6 |
| 1-2 s | 16 | 4 | 12 | 6 | 0 | 3 | 1 | 11 | 6 | 16 | 15 | 12 |
| 2-5 s | 50 | 7 | 43 | 14 | 6 | 11 | 2 | 33 | 18 | 45 | 40 | 30 |
| 5-10 s | 48 | 6 | 42 | 7 | 5 | 11 | 2 | 31 | 13 | 47 | 40 | 22 |
| > 10 s | 44 | 10 | 34 | 0 | 3 | 0 | 1 | 25 | 0 | 39 | 36 | 27 |
| all | 167 | 30 | 137 | 30 | 14 | 26 | 6 | 105 | 40 | 155 | 139 | 97 |

## T6 image side of exit and re-entry (annotation, 167)
| exit side | bottom | left | left+bottom | none | right | right+bottom | All |
|---|---|---|---|---|---|---|---|
| bottom | 1 | 1 | 0 | 0 | 1 | 0 | 3 |
| left | 0 | 80 | 2 | 0 | 1 | 0 | 83 |
| left+bottom | 0 | 0 | 1 | 0 | 0 | 0 | 1 |
| none | 0 | 0 | 0 | 0 | 4 | 0 | 4 |
| right | 0 | 0 | 0 | 2 | 71 | 2 | 75 |
| right+bottom | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| All | 1 | 81 | 3 | 2 | 77 | 3 | 167 |

same side: 154 of 167

annotated implied speed (m/s) quantiles 10/25/50/75/90/95/max: [1.02, 1.7, 2.8, 3.9, 4.81, 5.49, 35.85]
annotated distance exit->re-entry (m) quantiles 10/25/50/75/90/95/max: [4.0, 8.1, 13.3, 23.4, 36.1, 42.1, 67.8]
system (own pieces) implied speed quantiles 10/25/50/75/90/95/max: [0.94, 1.54, 2.45, 3.43, 4.14, 4.61, 31.97] n 150
own-piece pairs within appearance_match's reach (12 m/s + 4 m): {True: 148, nan: 5, False: 2}

## T7 mutual rule funnel (137 scored)
| funnel | lost | right | wrong | All |
|---|---|---|---|---|
| B row in a foreign piece (no link of known pieces helps) | 4 | 0 | 3 | 7 |
| B row in a fragment piece (no link of known pieces helps) | 23 | 0 | 0 | 23 |
| candidate generation | 53 | 0 | 2 | 55 |
| decision: linked | 0 | 26 | 0 | 26 |
| decision: margin < 0.05 | 10 | 0 | 0 | 10 |
| decision: no runner-up | 9 | 0 | 0 | 9 |
| decision: not mutual best | 5 | 0 | 1 | 6 |
| no earlier own piece of the person | 1 | 0 | 0 | 1 |
| All | 105 | 26 | 6 | 137 |

## T8 candidate generation losses: every gate failing for the nearest earlier own piece (55 returns)
| fn_gate_nearest_own | returns |
|---|---|
| gap > 10 s | 18 |
| A team unverified; B team unverified | 9 |
| B team unverified | 6 |
| A team unverified | 3 |
| B team unverified; B clean rows < 5 | 3 |
| A team unverified; gap > 10 s | 2 |
| A team unverified; B team unverified; gap > 10 s | 2 |
| A team unverified; B team unverified; overlap in time | 2 |
| B team unverified; overlap in time | 2 |
| A team unverified; B clean rows < 5; gap > 10 s | 1 |
| A team unverified; A clean rows < 5; gap > 10 s | 1 |
| out of reach | 1 |
| B team unverified; gap > 10 s | 1 |
| A clean rows < 5 | 1 |
| A clean rows < 5; B clean rows < 5; overlap in time | 1 |
| overlap in time | 1 |
| A team unverified; A clean rows < 5 | 1 |

| index | returns |
|---|---|
| gap > 10 s (alone or with others) | 25 |
| team unverified on A or B | 33 |
| fewer than 5 clean rows on A or B | 8 |
| overlap in time | 6 |
| out of reach | 1 |
| only one gate fails | 39 |

## T8b the same gates on the 155 pairs of own pieces (scored and undetected returns)
| index | pairs |
|---|---|
| pairs | 155 |
| candidate | 62 |
| gap > 10 s | 42 |
| team unverified | 57 |
| clean rows < 5 | 18 |
| overlap in time | 4 |
| out of reach | 2 |

| anchors_mutual_status | pairs |
|---|---|
| not a candidate pair | 93 |
| linked | 27 |
| margin < 0.05 | 13 |
| not mutual best | 12 |
| no runner-up | 10 |

path G on the same 155 pairs:
| anchors_pathG | pairs |
|---|---|
| G:gap_too_long | 65 |
| not an audit pair (gap > 10 s or overlap) | 46 |
| veto:piece_jump | 20 |
| G:rivals | 12 |
| G:no_clean_row | 4 |
| veto:team_conflict | 4 |
| G:no_anchor | 2 |
| G:no_decision_frames | 2 |

descriptor kinds (earlier/later): {'strict/strict': 137, 'strict/relaxed': 12, 'relaxed/strict': 6}
cos of the true pair, quantiles 5/25/50/75/95: [0.863, 0.934, 0.957, 0.97, 0.983]
true pair beats the best other-person piece on both sides: 139 of 155
... by >= 0.02: 126 of 155
... by >= 0.03: 121 of 155
... by >= 0.05: 97 of 155
... by >= 0.1: 46 of 155
without SNGS-162: beats by > 0: 129 of 135; by >= 0.05: 87
plugin team labels of the two own pieces differ: 0 ; a label is missing: 7
sides (system boxes): own piece A ends at an edge 118 ; own piece B starts at an edge 107 ; both and same side 80 of 155
lead of own piece A before the exit (frames) quantiles 50/75/90/max: [4.0, 7.0, 30.599999999999994, 114.0] ; delay of own piece B after re-entry: [2.0, 6.5, 25.0, 215.0]

## T9 decision losses: who is the rival (25 returns with a true candidate pair, not linked)
| funnel /  | another person / another person | another person / no rival | another person / same person | no rival / another person | same person / another person | same person / no rival |
|---|---|---|---|---|---|---|
| decision: margin < 0.05 | 3 | 0 | 4 | 0 | 3 | 0 |
| decision: no runner-up | 0 | 6 | 0 | 2 | 0 | 1 |
| decision: not mutual best | 4 | 0 | 2 | 0 | 0 | 0 |

## T10 row pair (piece A, piece B) of the 137 scored returns under the mutual rule
| mutual_status | lost | right | wrong | All |
|---|---|---|---|---|
| linked | 0 | 21 | 0 | 21 |
| margin < 0.05 | 7 | 0 | 0 | 7 |
| no runner-up | 8 | 0 | 0 | 8 |
| not a candidate pair | 86 | 5 | 6 | 97 |
| not mutual best | 4 | 0 | 0 | 4 |
| All | 105 | 26 | 6 | 137 |

candidate row pairs: 40 ; true partner rank 1 for A: 36 ; rank 1 for B: 38 ; rank 1 on both: 36
cos of candidate row pairs quantiles 5/25/50/75/95: [0.906, 0.943, 0.961, 0.977, 0.987]

path G on the 137 row pairs:
| pathG | returns |
|---|---|
| G:gap_too_long | 46 |
| not an audit pair (gap > 10 s or overlap) | 38 |
| veto:piece_jump | 14 |
| G:no_anchor | 11 |
| G:rivals | 10 |
| veto:team_conflict | 9 |
| G:no_clean_row | 6 |
| G:no_decision_frames | 2 |
| veto:role | 1 |

## T11 mutual links that are not right
| clip | earlier | later | label | cos | min_margin | gap_s | a_clean | b_clean | a_person | b_person | a_purity | b_purity | a_true_partner | b_true_partner |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-163 | 21 | 54 | unknown | 0.9263 | 0.2085 | 1.08 | 91 | 7 | 12 | 12 | 0.989 | 1 | own piece 79 exists but is no candidate: gap > 10 s | candidate 21 cos 0.9263 (lost by 0.0000) |
| SNGS-106 | 7 | 23 | unknown | 0.8902 | 0.0773 | 0.52 | 74 | 6 | 1 | 1 | 1 | 1 | own piece 24 exists but is no candidate: B team unverified | candidate 7 cos 0.8902 (lost by -0.0000) |
| SNGS-106 | 9 | 46 | wrong | 0.8917 | 0.0769 | 9.28 | 123 | 30 | 15 | 24 | 0.965 | 0.618 | own piece 40 exists but is no candidate: B team unverified | no own piece before it (person annotated on 33 frames there) |
| SNGS-100 | 27 | 63 | wrong | 0.9209 | 0.0584 | 7.48 | 17 | 210 | 17 | 25 | 1 | 1 | no own piece after it (person annotated on 6 frames there) | no own piece before it (person annotated on 5 frames there) |
| SNGS-169 | 85 | 110 | wrong | 0.8127 | 0.1271 | 8 | 61 | 49 | 16 | 18 | 1 | 1 | no own piece after it (person annotated on 4 frames there) | own piece 88 exists but is no candidate: A team unverified |
| SNGS-162 | 4 | 96 | wrong | 0.9915 | 0.0902 | 3.92 | 425 | 156 | 14 | 12 | 1 | 1 | candidate 104 cos 0.9013 (lost by 0.0902) | own piece 18 exists but is no candidate: A team unverified |
| SNGS-162 | 23 | 116 | wrong | 0.9703 | 0.1464 | 1.6 | 341 | 14 | 5 | 8 | 0.957 | 1 | no own piece after it (person annotated on 3 frames there) | candidate 95 cos 0.7865 (lost by 0.1838) |
| SNGS-162 | 27 | 114 | wrong | 0.9695 | 0.0593 | 9.28 | 113 | 55 | 9 | 18 | 0.995 | 1 | candidate 93 cos 0.8642 (lost by 0.1053) | candidate 42 cos 0.8747 (lost by 0.0948) |
| SNGS-162 | 55 | 111 | wrong | 0.9541 | 0.073 | 8.24 | 54 | 49 | 21 | 13 | 0.963 | 1 | candidate 97 cos 0.8699 (lost by 0.0842) | candidate 6 cos 0.8811 (lost by 0.0730) |

right links: same plugin id on both sides: {False: 49, True: 11}
min margin >= 0.05: {'right': 60, 'wrong': 7, 'unknown': 2}
min margin >= 0.06: {'right': 57, 'wrong': 5, 'unknown': 2}
min margin >= 0.08: {'right': 51, 'wrong': 3, 'unknown': 1}
min margin >= 0.1: {'right': 46, 'wrong': 2, 'unknown': 1}
min margin >= 0.15: {'right': 28, 'unknown': 1}
cos >= 0.9: {'right': 60, 'wrong': 5, 'unknown': 1}
cos >= 0.93: {'right': 56, 'wrong': 4}
cos >= 0.95: {'right': 44, 'wrong': 4}

## T12 plugin joins (93)
| label | different people | one piece matches nobody | same person | All |
|---|---|---|---|---|
| right | 0 | 0 | 33 | 33 |
| unknown | 22 | 15 | 13 | 50 |
| wrong | 10 | 0 | 0 | 10 |
| All | 32 | 15 | 46 | 93 |

| index | right | unknown | wrong |
|---|---|---|---|
| joins | 33 | 50 | 10 |
| gap_s | 0.68/1.56/3/4.68/9.44 | 0.44/0.73/2.14/4.91/9.08 | 1.52/2.89/5.4/7.9/10.04 |
| rows_earlier | 10/34/81/120/668 | 1/1/3/27/232 | 11/51.75/103.5/172.5/336 |
| rows_later | 17/44/133/208/653 | 1/1/2.5/12.25/206 | 10/12.5/40/142.5/378 |
| cos_any | 0.92/0.946/0.967/0.979/0.988 | 0.549/0.739/0.803/0.89/0.992 | 0.498/0.736/0.819/0.906/0.94 |
| other_pieces_scoring_higher | 0/0/0/0/1 | 0/1.25/5/13.5/47 | 0/1.75/11/14/52 |
| speed_mps | 0.01/1.49/1.99/2.74/4.14 | 0.1/1.68/3.14/4.44/26.27 | 0.64/2.145/3.615/4.468/7.52 |
| height_ratio | 0.78/0.95/0.99/1.11/1.25 | 0.48/0.822/1.005/1.148/1.71 | 0.79/0.862/0.9/0.988/1.16 |
| same_edge_side | 33 | 34 | 10 |
| team_label_equal | 29 | 21 | 6 |
| within_reach | 33 | 48 | 10 |
| shortest_piece_le_4_rows | 0 | 46 | 0 |

wrong joins:
| clip | earlier | later | gap_s | a_rows | b_rows | a_clean | b_clean | a_person | b_person | cos_any | desc_kinds | pieces_scoring_higher | pool | a_end_side | speed_mps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 46 | 100046 | 10.04 | 63 | 36 | 34 | 1 | 8 | 3 | 0.7167 | strict/relaxed | 14 | 54 | left | 3.85 |
| SNGS-106 | 6 | 100006 | 2.84 | 153 | 102 | 87 | 94 | 14 | 9 | 0.6994 | strict/strict | 10 | 45 | right | 0.64 |
| SNGS-097 | 56 | 100056 | 3.04 | 92 | 11 | 65 | 8 | 7 | 22 | 0.9401 | strict/strict | 0 | 33 | left | 3.38 |
| SNGS-066 | 62 | 100062 | 6.04 | 179 | 44 | 163 | 40 | 6 | 10 | 0.7943 | strict/strict | 12 | 74 | right | 2.05 |
| SNGS-169 | 19 | 100019 | 5.24 | 48 | 17 | 32 | 9 | 9 | 7 | 0.8056 | strict/strict | 15 | 50 | left | 5.04 |
| SNGS-162 | 100031 | 200031 | 1.92 | 115 | 11 | 95 | 5 | 17 | 2 | 0.9217 | strict/strict | 0 | 44 | left | 1.34 |
| SNGS-151 | 2 | 100002 | 8.52 | 336 | 201 | 317 | 146 | 2 | 1 | 0.8319 | strict/strict | 14 | 86 | right | 3.98 |
| SNGS-151 | 7 | 100007 | 9.04 | 236 | 10 | 45 | 0 | 3 | 12 | 0.4976 | strict/relaxed | 52 | 61 | right | 2.43 |
| SNGS-112 | 100015 | 200015 | 5.56 | 34 | 378 | 32 | 300 | 16 | 20 | 0.8575 | strict/strict | 7 | 42 | right | 7.52 |
| SNGS-112 | 49 | 100049 | 1.52 | 11 | 156 | 3 | 97 | 6 | 25 | 0.9369 | relaxed/strict | 0 | 56 | left | 4.63 |

cos_any >= 0.9: {'right': 33, 'wrong': 3} | unknown joins kept: {'one piece matches nobody': 9, 'same person': 2, 'different people': 1}
cos_any >= 0.92: {'right': 32, 'wrong': 3} | unknown joins kept: {'one piece matches nobody': 9, 'same person': 2, 'different people': 1}
cos_any >= 0.95: {'right': 23} | unknown joins kept: {'one piece matches nobody': 6}
no other piece scores higher: {'right': 29, 'wrong': 3} ; <= 1 higher: {'right': 33, 'wrong': 3}
cos_any >= 0.90 and <= 1 other piece higher: {'right': 33, 'wrong': 3} | unknown joins kept: {'one piece matches nobody': 9}

## T13 undetected: one row per missing side (34 sides of 30 returns)
| reason | A | B | All |
|---|---|---|---|
| a fitting box exists but is matched to another person | 1 | 3 | 4 |
| box found but IoU < 0.5 (partial body), matched later | 2 | 4 | 6 |
| never matched in this visible stretch | 2 | 2 | 4 |
| never matched in this visible stretch (stretch <= 12 frames) | 0 | 2 | 2 |
| no box at all near the gap, matched later | 4 | 5 | 9 |
| only a poorly overlapping box, matched later | 7 | 2 | 9 |
| All | 16 | 18 | 34 |

box height at the gap < 40 px (720p): 18 of 34 ; nearest matched row of the person, frames from the gap, quantiles 25/50/75/max: [20.0, 30.5, 38.0, 66.0]
undetected returns with an own piece on both sides: 25 of 30 ; of those, true pair beats every other-person piece: 23 ; is an appearance_match candidate: 4

| clip | ret | ref_id | missing_side | gap_s | side | stretch_frames | best_iou_in_window | frames_fitting_box_taken_by_other | ann_box_h_720p | ann_box_w_720p | nearest_matched_row_frames_from_gap | own_piece_frames_from_gap | reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 7 | 9 | A | 6.32 | left | 503 | 0.23 | 0 | 50.7 | 20.7 | 13 | 13 | only a poorly overlapping box, matched later |
| SNGS-107 | 7 | 9 | B | 6.32 | left | 89 | 0.39 | 0 | 57.3 | 23.3 | 14 | 14 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-107 | 10 | 14 | B | 15.4 | left | 69 | 0.05 | 0 | 76 | 28.7 | 32 | 55 | only a poorly overlapping box, matched later |
| SNGS-163 | 7 | 8 | A | 10.4 | left | 302 | 0.29 | 0 | 61.3 | 8 | 14 | 14 | only a poorly overlapping box, matched later |
| SNGS-163 | 9 | 10 | A | 11.4 | left | 293 | 0.37 | 0 | 54.7 | 37.3 | 31 | 31 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-163 | 10 | 11 | A | 12.08 | left | 294 | 0.18 | 0 | 26 | 10 | 34 | 34 | only a poorly overlapping box, matched later |
| SNGS-163 | 12 | 13 | B | 6.04 | right | 478 | 0.07 | 0 | 27.3 | 17.3 | 27 | 38 | only a poorly overlapping box, matched later |
| SNGS-163 | 13 | 14 | A | 10.76 | left | 319 | 0.27 | 0 | 52.7 | 13.3 | 19 | 54 | only a poorly overlapping box, matched later |
| SNGS-106 | 6 | 14 | B | 13.72 | right | 234 | 0 | 0 | 22.7 | 6.7 | 66 | 66 | no box at all near the gap, matched later |
| SNGS-097 | 0 | 3 | B | 2.72 | left+bottom | 20 | 0 | 0 | 7.3 | 3.3 |  |  | never matched in this visible stretch |
| SNGS-097 | 1 | 3 | A | 2.16 | left+bottom | 20 | 0 | 0 | 4 | 2 |  |  | never matched in this visible stretch |
| SNGS-097 | 1 | 3 | B | 2.16 | left+bottom | 4 | 0 | 0 | 11.3 | 10.7 |  |  | never matched in this visible stretch (stretch <= 12 frames) |
| SNGS-097 | 2 | 7 | B | 0.8 | left | 6 | 0 | 0 | 12.7 | 6.7 |  |  | never matched in this visible stretch (stretch <= 12 frames) |
| SNGS-097 | 7 | 15 | B | 1.84 | left | 80 | 0.83 | 4 | 59.3 | 12.7 | 17 | 25 | a fitting box exists but is matched to another person |
| SNGS-067 | 5 | 9 | B | 4.44 | right | 42 | 0.56 | 2 | 45.3 | 32.7 | 36 |  | a fitting box exists but is matched to another person |
| SNGS-067 | 6 | 10 | B | 3.96 | right | 168 | 0.45 | 0 | 42 | 6.7 | 64 | 64 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-067 | 9 | 14 | B | 3.04 | right | 190 | 0.42 | 0 | 40.7 | 4.7 | 21 | 21 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-100 | 1 | 3 | A | 1.68 | right | 103 | 0 | 0 | 2 | 2 | 43 | 43 | no box at all near the gap, matched later |
| SNGS-066 | 1 | 4 | B | 2.08 | right | 470 | 0.41 | 0 | 94 | 47.3 | 13 | 13 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-066 | 8 | 12 | B | 4.72 | right | 465 | 0 | 0 | 37.3 | 21.3 | 20 | 20 | no box at all near the gap, matched later |
| SNGS-161 | 7 | 11 | A | 1.24 | left | 53 | 0 | 0 | 5.3 | 5.3 | 30 | 30 | no box at all near the gap, matched later |
| SNGS-161 | 11 | 17 | A | 0.8 | left | 55 | 0 | 0 | 42 | 12 | 36 | 36 | only a poorly overlapping box, matched later |
| SNGS-169 | 7 | 9 | A | 1.08 | none | 307 | 0.52 | 1 | 17.3 | 2.7 | 41 | 41 | a fitting box exists but is matched to another person |
| SNGS-162 | 3 | 6 | A | 7.04 | none | 447 | 0 | 0 | 47.3 | 30.7 | 36 | 36 | only a poorly overlapping box, matched later |
| SNGS-162 | 17 | 20 | A | 6.8 | none | 450 | 0.4 | 0 | 62.7 | 23.3 | 25 | 30 | box found but IoU < 0.5 (partial body), matched later |
| SNGS-151 | 0 | 1 | A | 14.64 | bottom | 183 | 0 | 0 | 3.3 | 13.3 | 27 | 27 | no box at all near the gap, matched later |
| SNGS-151 | 3 | 4 | B | 10.28 | right | 235 | 0.57 | 2 | 44.7 | 15.3 | 26 | 37 | a fitting box exists but is matched to another person |
| SNGS-151 | 5 | 7 | B | 5.24 | right | 245 | 0 | 0 | 33.3 | 21.3 | 64 | 64 | no box at all near the gap, matched later |
| SNGS-151 | 10 | 15 | A | 0.52 | right | 189 | 0 | 0 | 32.7 | 4 | 20 | 20 | no box at all near the gap, matched later |
| SNGS-151 | 10 | 15 | B | 0.52 | right | 150 | 0 | 0 | 43.3 | 4 | 37 | 37 | no box at all near the gap, matched later |
| SNGS-151 | 11 | 15 | B | 6.48 | right | 236 | 0 | 0 | 26.7 | 12.7 | 43 | 48 | no box at all near the gap, matched later |
| SNGS-102 | 3 | 6 | A | 14.8 | left | 311 | 0.19 | 0 | 25.3 | 10 | 53 | 53 | only a poorly overlapping box, matched later |
| SNGS-102 | 11 | 15 | A | 11.92 | left | 355 | 0 | 0 | 36 | 16.7 |  |  | never matched in this visible stretch |
| SNGS-102 | 11 | 15 | B | 11.92 | left | 97 | 0 | 0 | 32 | 11.3 |  |  | never matched in this visible stretch |

## T14 holes >= 10 frames inside gbf pieces (not split: no box at the edge next to the hole)
| hole frames | different people | same person | unknown (no matched row on one side) | All |
|---|---|---|---|---|
| 10-14 | 38 | 165 | 13 | 216 |
| 15-24 | 37 | 97 | 11 | 145 |
| 25-49 | 29 | 44 | 13 | 86 |
| 50-99 | 33 | 23 | 8 | 64 |
| >= 100 | 57 | 17 | 14 | 88 |
| All | 194 | 346 | 59 | 599 |

| label | cos < 0.90 | cos >= 0.90 | no cosine (a side has < 5 clean rows) |
|---|---|---|---|
| different people | 54 | 10 | 130 |
| same person | 11 | 161 | 174 |

switch events with >= 5 matched rows on both sides: 78 in 64 pieces; a hole >= 10 frames lies between the two runs in 47; segment cosine known in 68, below 0.90 in 51

returns that need a cut:
| clip | ret | ref_id | piece_b | b_kind | piece_rows | majority_person | frame_b | person_run_rows_from_b | frames_missing_before_b | centre_jump_px | implied_speed_mps | gbf_edge_touch_prev_or_b | out_split | out_plugin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-107 | 8 | 10 | 36 | foreign | 197 | 9 | 642 | 3 | 152 | 33.3 | 2.3 | False | wrong | wrong |
| SNGS-107 | 14 | 18 | 73 | foreign | 65 | 17 | 675 | 1 |  |  |  |  | lost | lost |
| SNGS-106 | 1 | 5 | 100011 | foreign | 110 | 4 | 230 | 37 |  |  |  |  | lost | right |
| SNGS-066 | 7 | 11 | 48 | foreign | 375 | 17 | 300 | 17 | 0 | 7.5 | 18.6 | False | lost | lost |
| SNGS-169 | 8 | 10 | 45 | foreign | 7 | 22 | 305 | 1 |  |  |  |  | lost | lost |
| SNGS-162 | 0 | 1 | 53 | foreign | 71 | 10 | 636 | 1 | 7 | 123.3 | 14 | True | wrong | wrong |
| SNGS-162 | 1 | 2 | 100033 | fragment | 4 | 10 | 431 | 1 | 76 | 58.5 | 5.4 | False | lost | wrong |
| SNGS-162 | 2 | 2 | 100115 | fragment | 3 | 8 | 733 | 1 |  |  |  |  | lost | wrong |
| SNGS-162 | 15 | 18 | 53 | foreign | 71 | 10 | 628 | 1 | 203 | 83.4 | 1.5 | False | wrong | wrong |

## T15 label suspects
| clip | person_i | jersey_i | person_j | jersey_j | team | before_i | after_i | before_j | after_j | cos_annotated_i | cos_annotated_j | cos_exchanged_i_to_j | cos_exchanged_j_to_i | clean_rows | exit_i | reentry_i | exit_j | reentry_j |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-162 | 9 | 10 | 18 |  | left | 27 | 93 | 35 | 114 | 0.864 | 0.852 | 0.969 | 0.957 | 113/74/6/55 | 438 | 585 | 480 | 622 |
| SNGS-162 | 12 | 10 | 14 | 28 | right | 18 | 96 | 4 | 104 | 0.901 | 0.901 | 0.965 | 0.992 | 283/156/425/14 | 431 | 591 | 496 | 622 |

| clip | id_first | id_second | team | jersey_first | jersey_second | first_ends | second_starts | gap_s | end_side | start_side | dist_m | piece_first | piece_second | cos | best_other_later | best_other_earlier | margin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SNGS-097 | 7 | 22 | left | 24 |  | 694 | 737 | 1.68 | left | left | 8.1 | 56 | 100056 | 0.9401 | 0.8142 | 0.7904 | 0.1259 |
| SNGS-100 | 17 | 25 | left |  | 62 | 254 | 430 | 7 | left | left+bottom | 36.6 | 27 | 63 | 0.9209 | 0.8089 | 0.8626 | 0.0584 |
| SNGS-112 | 6 | 25 | left | 33 |  | 424 | 458 | 1.32 | left | left | 7.2 | 49 | 100049 | 0.9369 | 0.6556 | 0.7915 | 0.1454 |

returns whose own pieces look less alike than one of them and another person's piece, by >= 0.05: 13 by clip: {'SNGS-162': 8, 'SNGS-066': 2, 'SNGS-102': 2, 'SNGS-163': 1}

## T16 ranking ceiling
| earlier candidates | returning B | top-1 is the person | top-1 another person | top-1 a fragment | no candidate | true piece removed | new B | new B with a top-1 | cos>=0.93 & margin>=0.05: right/wrong/new | cos>=0.95 & margin>=0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| all earlier pieces | 403 | 358 | 20 | 25 | 0 | 0 | 182 | 153 | 263/5/13 | 218/5/5 |
| strict descriptors only (>= 5 clean rows both) | 403 | 324 | 34 | 4 | 41 | 15 | 182 | 130 | 255/8/3 | 210/8/0 |
| earlier piece ended at an image edge | 403 | 151 | 195 | 42 | 15 | 221 | 182 | 132 | 113/11/5 | 92/9/2 |
| ended at an edge, B starts at the same side | 403 | 87 | 23 | 6 | 287 | 23 | 182 | 23 | 70/3/1 | 55/2/0 |
| within reach (8 m/s + 4 m) | 403 | 357 | 21 | 25 | 0 | 2 | 182 | 136 | 270/6/13 | 224/6/5 |
| gap <= 10 s | 403 | 313 | 48 | 42 | 0 | 49 | 182 | 153 | 238/5/12 | 193/4/5 |

## T17 fragments and foreign pieces under row A / row B
| side | kind | linkable only through a piece with < 5 matched rows | linkable: existing pieces, known identity | needs a cut around a stray run of < 5 rows | needs an in-track switch fix (cut of a run >= 5 rows) |
|---|---|---|---|---|---|
| A | foreign | 2 | 5 | 1 | 0 |
| A | fragment | 5 | 11 | 1 | 1 |
| B | foreign | 0 | 0 | 4 | 3 |
| B | fragment | 21 | 0 | 2 | 0 |

fragments: 41; <= 4 rows: 38; 0 clean rows: 21; start or end at the image edge: 29; a gbf re-entry piece (k > 0): 16; person has an own piece on that side: 34; within 5 frames of it: 21; within 12: 23
B-side fragments: 23; own piece follows: 19; within 5 frames: 11; box IoU across >= 0.3: 3; centre shift <= 60 px: 9; cos to own piece quantiles 25/50/75: [0.831, 0.928, 0.948]
