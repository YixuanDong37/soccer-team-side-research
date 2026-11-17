# fuse_r1: round 1 of the fused re-linker (changes after the verifiers' findings on work/fuse)

Everything here works on the 13 SN-GSR dev clips only, reads the tracking outputs through
`reid_lab/data.py` and the other agents' label-free modules, and writes only into this directory.
No image or video was opened, no git, no network. Python: gbf's venv, always with `-I` (I added `-B`
so that importing other agents' modules leaves no bytecode in their folders).

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/fuse_r1
```

`PREREG.txt` (written 07:28 US Central on 2026-11-15, before any fuse_r1 code ran) fixes the method
changes and the rule for the operating points. It also lists what I already knew from round 0, so
this is a protocol fixed in advance, not a blind one.

## The rule in plain words

1. **Cut first** (m_cuts, P(switch) >= 0.7, unchanged): where the plugin's track moved onto another
   person inside a piece, the piece is cut.
2. **Two kinds of pieces** (new): a piece with at least 5 boxes is *solid*, a shorter one is *short*.
3. **Join solid pieces** (fuse's rule, on solid pieces only): two groups of solid pieces of the same
   kit that are never in the same frame are joined, most alike first, when
   - they look alike at all (appearance cosine >= 0.3; m_appear's team-centred, whitened prototypes),
   - each is the other's best partner,
   - appearance + position bonus beats by the **margin** both 0.5 ("nobody") and the score to any
     group that is seen at the same time as the partner (certainly someone else), on both sides.
   The position bonus is m_space's "the later piece starts where the earlier one can have got to",
   now computed **among solid pieces only**, so a 1-4 box fragment cannot absorb the position
   evidence between two long pieces. The plugin's own join is an input of the pair model as before;
   on solid pieces its fitted weight is about zero (-0.37 to +0.33 over the folds), so it does nothing.
4. **Short pieces as leaves** (new, only at the recall point): a short piece may be attached to at
   most one group, on its own appearance + position evidence and the same rule. It never connects
   two groups and never changes a group's appearance.
5. **It abstains** when the two do not look alike, when a candidate visible together with the partner
   is nearly as good, when the pieces share a frame or differ in team, and for referees. With leaves
   off, pieces under 5 boxes are left alone.

One number sets how sure a join must be (the margin), and one switch says whether short pieces are
attached. Frozen settings (rule B on all 13 clips): strict margin 0.60, leaves off; balanced 0.40,
leaves off; recall 0.20, leaves on.

## Results (137 scored returns, 13 dev clips, leave one clip out, pooled)

Numbers of `reid_lab/score.py`'s CLI (`out/cli_scores.txt`, verbatim):

| file | right | wrong | lost | links right / wrong / unknown | link precision | impure rows |
|---|---|---|---|---|---|---|
| gbf now (split) | 0 | 4 | 133 | 0 / 0 / 0 | - | 2.61% |
| plugin joins | 30 | 14 | 93 | 33 / 10 / 50 | 0.767 | 3.08% |
| mutual baseline | 26 | 6 | 105 | 60 / 7 / 2 | 0.895 | 3.03% |
| cuts only `METHOD_cuts_only.json` (baseline of the linking step) | 0 | 0 | 137 | 0 / 0 / 0 | - | 1.00% |
| round 0 (fuse) strict / balanced / recall | 33 / 93 / 110 | 4 / 7 / 7 | 100 / 37 / 20 | 152/4/158, 290/10/376, 314/10/437 | 0.974, 0.967, 0.969 | 1.28%, 1.89%, 1.91% |
| **strict** `METHOD_strict.json` | 67 | 3 | 67 | 312 / 7 / 25 | 0.9781 | 1.61% |
| **balanced** `METHOD_balanced.json` | 85 | 5 | 47 | 378 / 16 / 30 | 0.9594 | 2.12% |
| **recall** `METHOD_recall.json` | 106 | 7 | 24 | 414 / 17 / 260 | 0.9606 | 2.18% |
| frozen strict setting, held-out weights `METHOD_fixed_strict.json` | 70 | 2 | 65 | 319 / 6 / 22 | 0.9815 | 1.45% |
| frozen balanced setting, held-out weights `METHOD_fixed_balanced.json` | 85 | 4 | 48 | 361 / 12 / 30 | 0.9678 | 1.81% |
| frozen recall setting, held-out weights `METHOD_fixed_recall.json` | 112 | 7 | 18 | 415 / 17 / 311 | 0.9606 | 2.18% |
| in-sample, frozen strict `METHOD_frozen_strict.json` | 69 | 2 | 66 | 315 / 7 / 23 | 0.9783 | 1.52% |
| in-sample, frozen balanced `METHOD_frozen_balanced.json` | 86 | 4 | 47 | 362 / 12 / 29 | 0.9679 | 1.81% |
| in-sample, frozen recall `METHOD_frozen_recall.json` | 112 | 7 | 18 | 415 / 17 / 311 | 0.9606 | 2.18% |

The three points keep the limits (wrong <= 4 / 7 / 14). The headline rows are `METHOD_strict / balanced /
recall`: each clip was processed with a setting chosen without it. The `METHOD_fixed_*` rows show what the
frozen setting does on each clip with weights fitted without that clip; their setting was chosen on all
13 clips, so they are not held-out for the choice of the setting.

### How the operating points were chosen (rule B of PREREG.txt, `rules.py`)

Candidates: leaves off / on x 16 margins. Per fold, on the 12 fit clips:
(i) wrong returns <= floor(K x scored_fit / 137) at the candidate's margin and every stricter one
(K = 4 / 7 / 14); (ii) among those the largest J = right - lam x (wrong returns + links judged wrong +
leaf links with different weak-label persons), summed over the fit clips except the one with the most
such errors (lam = 10 / 3 / 1, round 0's values); (iii) ties: larger margin, then leaves off.

| point | settings chosen by the 13 folds | all 13 clips (frozen) | folds equal to frozen |
|---|---|---|---|
| strict | 9x 0.60 core, 3x 0.70 core, 1x 0.50 core | 0.60 core | 9 / 13 |
| balanced | 4x 0.40, 4x 0.20, 3x 0.50, 1x 0.25, 1x 0.15 (all core) | 0.40 core | 4 / 13 |
| recall | 10x 0.20 +leaves, 1x 0.25 +leaves, 1x 0.20 core, 1x 0.25 core | 0.20 +leaves | 10 / 13 |

In 12 folds the clip left out of J is SNGS-162; in the fold that holds SNGS-162 out it is SNGS-112,
SNGS-066 or SNGS-100, and that fold chooses strict 0.60 core like most folds, balanced 0.15 core (the loosest
of the 13) and recall 0.25 +leaves.

**The points depend on the rule.** The same runs under the other rules that PREREG.txt names
(`out/tradeoff.md`):

| rule | strict | balanced | recall |
|---|---|---|---|
| **B** (the METHOD files) | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 |
| R0 = round 0's rule (J, no cap, no clip left out) | 26 / 2 / 109 | 72 / 4 / 61 | 97 / 8 / 32 |
| Rcap = cap only (most right under the cap) | 78 / 8 / 51 (limit broken) | 98 / 9 / 30 (limit broken) | 112 / 9 / 16 |
| Rnotrim = B without leaving a clip out | 26 / 2 / 109 | 72 / 4 / 61 | 97 / 8 / 32 |
| rule B on round 0's linker (no tiers) | 51 / 4 / 82 | 108 / 7 / 22 | 111 / 8 / 18 |

Rule B was fixed before any fuse_r1 result existed, but I knew round 0's held-out tables when I wrote it
(in round 0 "J without the worst fit clip" was computed after the fact and gave 88 / 4 / 45). Read the
operating points as "what this rule does", and the fixed-setting table below as the stable finding.

### Trade-off: fixed setting (nothing chosen; each clip with the weights fitted without it)

right / wrong / lost | links right / wrong / unknown | impure rows | without SNGS-162 | SNGS-162 alone.
Full table with in-sample columns and round 0's linker: `out/tradeoff.md`.

| leaves | margin | held-out | without SNGS-162 | SNGS-162 |
|---|---|---|---|---|
| off | 0.10 | 96 / 7 / 34, 432 / 20 / 38, 2.25% | 92 / 2 / 25 | 4 / 5 / 9 |
| off | 0.20 | 95 / 5 / 37, 415 / 17 / 36, 2.14% | 91 / 1 / 27 | 4 / 4 / 10 |
| off | 0.30 | 91 / 5 / 41, 395 / 16 / 34, 2.14% | 87 / 1 / 31 | 4 / 4 / 10 |
| off | 0.35 | 88 / 4 / 45, 382 / 14 / 34, 2.05% | 84 / 1 / 34 | 4 / 3 / 11 |
| off | 0.40 | 85 / 4 / 48, 361 / 12 / 30, 1.81% | 81 / 1 / 37 | 4 / 3 / 11 |
| off | 0.50 | 81 / 4 / 52, 345 / 9 / 29, 1.78% | 78 / 1 / 40 | 3 / 3 / 12 |
| off | 0.55 | 76 / 3 / 58, 333 / 7 / 26, 1.61% | 73 / 1 / 45 | 3 / 2 / 13 |
| off | 0.60 | 70 / 2 / 65, 319 / 6 / 22, 1.44% | 67 / 0 / 52 | 3 / 2 / 13 |
| off | 0.70 | 58 / 2 / 77, 270 / 5 / 21, 1.30% | 55 / 0 / 64 | 3 / 2 / 13 |
| off | 0.80 | 40 / 1 / 96, 230 / 4 / 17, 1.23% | 37 / 0 / 82 | 3 / 1 / 14 |
| off | 0.90 | 25 / 0 / 112, 176 / 2 / 9, 1.07% | 23 / 0 / 96 | 2 / 0 / 16 |
| on | 0.10 | 113 / 9 / 15, 432 / 20 / 334, 2.28% | 105 / 2 / 12 | 8 / 7 / 3 |
| on | 0.20 | 112 / 7 / 18, 415 / 17 / 311, 2.18% | 104 / 1 / 14 | 8 / 6 / 4 |
| on | 0.30 | 104 / 7 / 26, 395 / 16 / 286, 2.17% | 97 / 1 / 21 | 7 / 6 / 5 |
| on | 0.40 | 96 / 6 / 35, 361 / 12 / 253, 1.83% | 89 / 1 / 29 | 7 / 5 / 6 |
| on | 0.50 | 92 / 6 / 39, 345 / 9 / 215, 1.79% | 86 / 1 / 32 | 6 / 5 / 7 |
| on | 0.60 | 78 / 4 / 55, 319 / 6 / 174, 1.46% | 72 / 0 / 47 | 6 / 4 / 8 |
| on | 0.80 | 44 / 1 / 92, 230 / 4 / 110, 1.23% | 39 / 0 / 80 | 5 / 1 / 12 |

With leaves off every margin from 0.35 up has at most 4 wrong returns. Compared with round 0's linker
at equal wrong counts the curve is about the same (round 0: 51 / 1 at 0.80, 91 / 4 at 0.50, 112 / 7 at
0.20; here 70 / 2, 88 / 4, 112 / 7); the gain of the tiers is between 2 and 3 wrong, where round 0 had
no setting at all, and in what the links are (below).

### Per clip (held-out, rule B; right / wrong / lost, links right / wrong / unknown, setting)

| clip | strict | balanced | recall |
|---|---|---|---|
| SNGS-107 | 8 / 0 / 8, 24 / 0 / 3 (0.60 core) | 9 / 0 / 7, 26 / 0 / 3 (0.50 core) | 12 / 0 / 4, 33 / 0 / 5 (0.25 core) |
| SNGS-163 | 2 / 0 / 9, 13 / 0 / 1 (0.70 core) | 4 / 0 / 7, 18 / 0 / 2 (0.50 core) | 8 / 0 / 3, 26 / 0 / 21 (0.20 +leaves) |
| SNGS-106 | 3 / 0 / 4, 24 / 0 / 3 (0.60 core) | 4 / 0 / 3, 35 / 0 / 3 (0.20 core) | 4 / 0 / 3, 35 / 0 / 3 (0.20 core) |
| SNGS-097 | 4 / 0 / 2, 22 / 0 / 0 (0.70 core) | 5 / 0 / 1, 26 / 2 / 0 (0.25 core) | 6 / 0 / 0, 26 / 2 / 15 (0.20 +leaves) |
| SNGS-067 | 5 / 0 / 5, 35 / 0 / 4 (0.60 core) | 8 / 0 / 2, 47 / 0 / 6 (0.20 core) | 9 / 0 / 1, 47 / 0 / 26 (0.20 +leaves) |
| SNGS-100 | 2 / 0 / 2, 31 / 1 / 0 (0.60 core) | 3 / 0 / 1, 37 / 2 / 0 (0.20 core) | 3 / 0 / 1, 37 / 2 / 15 (0.20 +leaves) |
| SNGS-066 | 5 / 1 / 5, 22 / 1 / 5 (0.50 core) | 5 / 1 / 5, 23 / 1 / 5 (0.40 core) | 6 / 1 / 4, 29 / 2 / 14 (0.20 +leaves) |
| SNGS-161 | 8 / 0 / 6, 24 / 0 / 1 (0.70 core) | 13 / 0 / 1, 30 / 0 / 1 (0.20 core) | 14 / 0 / 0, 30 / 0 / 13 (0.20 +leaves) |
| SNGS-169 | 11 / 0 / 4, 27 / 0 / 1 (0.60 core) | 11 / 0 / 4, 31 / 0 / 1 (0.40 core) | 15 / 0 / 0, 38 / 0 / 38 (0.20 +leaves) |
| SNGS-162 | 3 / 2 / 13, 38 / 4 / 1 (0.60 core) | 4 / 4 / 10, 48 / 10 / 3 (0.15 core) | 8 / 6 / 4, 45 / 10 / 42 (0.25 +leaves) |
| SNGS-151 | 3 / 0 / 6, 8 / 0 / 2 (0.60 core) | 5 / 0 / 4, 11 / 0 / 2 (0.50 core) | 7 / 0 / 2, 16 / 0 / 29 (0.20 +leaves) |
| SNGS-102 | 10 / 0 / 2, 21 / 0 / 4 (0.60 core) | 10 / 0 / 2, 22 / 0 / 4 (0.40 core) | 10 / 0 / 2, 23 / 0 / 17 (0.20 +leaves) |
| SNGS-112 | 3 / 0 / 1, 23 / 1 / 0 (0.60 core) | 4 / 0 / 0, 24 / 1 / 0 (0.40 core) | 4 / 0 / 0, 29 / 1 / 22 (0.20 +leaves) |

## What the verifiers found and what was done

| finding (round 0) | status |
|---|---|
| v_leak major 1: PRTReid was probably trained on the SN-GSR train split, which all dev clips come from | **Not fixable here; not verified either.** The plugin loads `prtreid-soccernet-baseline.pth.tar` (tools/fetch_weights.sh, gsr/provenance-gsr.json); gbf's registry says of its training data "Not reviewed". Added: `stress.py` (what happens with degraded embeddings) and `selfcheck.py` (a label-free check to run on new footage). Needs footage outside SN-GSR train to settle. |
| v_leak major 2: the fixed-margin row 0.5 (91 / 4) was a pick on the pooled table | No row is picked. The points come from rule B, fixed in advance; the frozen settings' rows are reported as such; the same runs under three other rules are shown. |
| v_leak major 3 / v_robust major 2: strict point weak (33 right) and fragile; the pooled result described a mix of margins, not the frozen one | Strict is now 67 / 3 / 67 held-out; 9 of 13 folds choose the frozen setting (margin 0.60), three choose 0.70 and one 0.50; the frozen setting's own held-out row is 70 / 2 / 65. Each threshold +-10%: 57-74 right, 3 wrong; 40 random joint draws: 58-72 right, always 3 wrong (`out/sensitivity_heldout.md`). Right against the plugin: better in 12 clips, worse in 1; 11 of the 34 returns after more than 10 s are right (round 0: 0). Under round 0's rule the strict point would be 26 / 2 / 109, under the cap-only rule 78 / 8 / 51. |
| v_robust major 1: the wrong count cannot be certified, it sits in one clip | **Still true.** Strict: 2 of 3 wrong returns are in SNGS-162; clip bootstrap P(wrong share <= 4/137) = 0.73 (0.88 for the frozen setting's row), wrong share 5%-95%: 0-4.6%. Against gbf now the wrong share is not above it in 100% of the resamples (strict), 46% (balanced), 36% (recall). `out/uncertainty.md`. |
| v_robust major 3: "impure rows below gbf now" was carried by the cuts, not by the links | Reported against cuts only now (table below). Links add impure rows: strict 1.00% -> 1.61%, balanced 2.12%, recall 2.18%; SNGS-162 and SNGS-112 end above gbf now at every point. |
| v_robust minor: 1-4 box pieces bridged two people | Removed by construction: two solid pieces are joined only on their own evidence; a short piece has at most one link (checked on every METHOD file by `check_method.py`). |

## What the linking adds to the cuts (diagnosis with labels, `out/diag_points.md`, `out/diag_baselines.json`)

| | impure rows | added by links | identities with >= 5 boxes of two people | a person's boxes in his largest identity | wrong + "right but mixed" returns |
|---|---|---|---|---|---|
| gbf now (split) | 2.61% | - | 76 | 0.68 | 4 + 0 |
| cuts only | 1.00% | 0 | 35 | 0.68 | 0 |
| strict | 1.61% | +673 rows | 42 | 0.84 | 3 + 2 |
| balanced | 2.12% | +1236 | 48 | 0.88 | 5 + 4 |
| recall | 2.18% | +1293 | 49 | 0.92 | 7 + 7 |
| frozen strict setting | 1.45% | +491 | 41 | 0.84 | 2 + 2 |

Impure rows added by links per clip at the strict point: SNGS-162 +349, SNGS-066 +182, SNGS-100 +96,
SNGS-112 +44, the other nine 0 or +1. Clips above gbf now: SNGS-162 (4.31% against 3.76%) and SNGS-112
(0.64% against 0.47%) at strict; SNGS-097 too at balanced and recall, where SNGS-162 reaches 8.4-8.6%.

Wrong returns: strict SNGS-066 person 21, SNGS-162 persons 9 and 12; balanced adds SNGS-162 persons 13
and 21; recall adds SNGS-162 persons 2 and 8. Links judged wrong at the strict point (7): SNGS-162 4>96,
10000030>93, 10000023>116, 10000021>43; SNGS-100 27>63; SNGS-112 49>100049; SNGS-066 14>10000028.
The first three SNGS-162 pairs and the SNGS-100 and SNGS-112 pairs are on the diagnosis agents' lists of
numerically suspect annotations (persons 12/14, 9/18, 5/8 exchanged; two annotation ids for one
player); SNGS-162 person 13 is the annotated return that is physically impossible. Nobody has looked at
the frames, so the wrong count of SNGS-162 is uncertain in both directions.

Links score.py cannot judge (a piece with < 5 matched boxes): strict 25, balanced 30, recall 260. By
weak labels (majority person of >= 1 matched box): strict 17 agree / 2 disagree / 6 no matched box;
recall 189 / 23 / 48; of the leaf links alone about one in ten of the judgeable ones disagrees (20 at the frozen recall setting). Round 0 had 158 / 376 / 437
such links.

## Sensitivity (`out/sensitivity_heldout.md`, `out/sensitivity_fixed.md`)

Held-out settings, one change at a time (right / wrong / lost):

| change | strict | balanced | recall |
|---|---|---|---|
| base | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 |
| margin x 0.9 / x 1.1 | 74 / 3 / 60, 57 / 3 / 77 | 86 / 5 / 46, 85 / 5 / 47 | 107 / 7 / 23, 105 / 7 / 25 |
| floor 0.27 / 0.33 | 67 / 3, 67 / 3 | 85 / 5, 85 / 5 | 106 / 7, 105 / 7 |
| "nobody" 0.45 / 0.55 | 70 / 3, 65 / 3 | 85 / 5, 85 / 5 | 106 / 7, 105 / 7 |
| bonus x 0.9 / x 1.1 | 64 / 3, 69 / 3 | 85 / 5, 85 / 5 | 105 / 8, 106 / 7 |
| solid from 4 / 6 boxes | 67 / 3, 67 / 3 | 85 / 5, 85 / 5 | 107 / 7, 106 / 7 |
| solid from 3 / 10 boxes | 69 / 5, 66 / 3 | 88 / 6, 83 / 5 | 108 / 7, 105 / 7 |
| cuts at P 0.63 / 0.77 | 67 / 3, 66 / 3 | 86 / 5, 84 / 5 | 105 / 7, 105 / 7 |
| all looser / all stricter by 10% | 73 / 4, 48 / 3 | 86 / 5, 83 / 5 | 107 / 7, 103 / 7 |
| 40 random joint draws of +-10% | 58-72 right, 3 wrong in all 40 | 83-86, 5 wrong in all 40 | 104-107, 7 wrong in all 40 |

Frozen settings: strict 63-76 right with 2 wrong in 28 draws and 3 in 12; balanced 82-87 / 4; recall
111-113 / 7. The one constant that matters is the tier boundary: with "solid from 3 boxes" the strict
point has 5 wrong returns (limit broken), with 4, 5, 6 or 10 boxes it has 3.

## The risk that the embeddings have seen these clips (`out/stress.md`, `out/selfcheck.md`)

`stress.py` re-runs the whole held-out pipeline (cut detector, tables, linker, the folds' settings;
nothing refitted) on degraded embeddings. Synthetic, not a result; "none" reproduces the METHOD files.

| embeddings | cuts made | strict | balanced | recall | self-check: same person 10% / 50% | different people 99% |
|---|---|---|---|---|---|---|
| none | 221 | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 | 0.65 / 0.83 | 0.51 |
| only 64 / 32 / 16 strongest directions kept | 212 / 199 / 179 | 69 / 3, 73 / 4, 70 / 3 | 86 / 5, 87 / 6, 85 / 5 | 107 / 7, 105 / 7, 105 / 9 | 0.71-0.81 / 0.88-0.94 | 0.58, 0.66, 0.75 |
| only 8 directions | 139 | 47 / 2 / 88 | 73 / 5 / 59 | 98 / 8 / 31 | 0.76 / 0.96 | 0.86 |
| drift 0.5 between 2 s stretches | 243 | 41 / 2 / 94 | 73 / 5 / 59 | 95 / 8 / 34 | 0.49 / 0.69 | 0.36 |
| drift 1.0 | 545 | 13 / 0 / 124 | 39 / 2 / 96 | 56 / 7 / 74 | 0.36 / 0.55 | 0.33 |
| drift 2.0 | 2082 | 8 / 1 / 128 | 13 / 3 / 121 | 23 / 6 / 108 | 0.32 / 0.62 | 0.47 |

Reading: when appearance gets worse the method mostly loses returns instead of making wrong ones (the
strict point never exceeds 4 wrong), but the wrong count does not fall with the right count, so the
share of wrong among the joined returns grows at the loose points (recall 7 of 113 -> 6 of 29), and the
cut detector either misses switches (8 directions: cuts only has 1 wrong return, 1.63% impure) or cuts
far too often (drift). `selfcheck.py` needs no labels and moves with both kinds of damage: on the dev
clips the same-person median is 0.76-0.90 per clip and the different-people 99% quantile 0.36-0.71;
drift 0.5 (0.69) and 8 directions (0.86) fall outside those ranges (so does 16 directions with 0.75, where the results are still unchanged: the alarm errs on the cautious side). Its AUC hardly moves (0.997 -> 0.992
at drift 1.0), so read the quantiles. It can raise an alarm on new footage; it cannot certify the thresholds.

## Commands (in this order)

```
$PY -I -B prep.py 5                  # label-free tables on the cut pieces, incl. the solid-only position tables -> cache/
                                     #   (I copied fuse's app_/space_/ends_ tables into cache/ first to save 3 min; without them
                                     #   prep.py builds everything; check_labelfree.py rebuilds two clips from nothing and gets the same links)
$PY -I -B check_linker2.py           # label-free: linker2 with every piece solid = fuse's linker (78 comparisons); tier structure
$PY -I -B fit_loco.py 5              # LOCO: pair model + rule B per fold -> params_loco.json, out/tradeoff.md, out/loco_report.json (9 min)
$PY -I -B check_r0.py                # the variant "r0" computed here reproduces fuse's fixed-margin rows and pair models exactly
$PY -I -B method.py                  # label-free: METHOD_strict/balanced/recall, METHOD_frozen_*, METHOD_fixed_*, METHOD_cuts_only
$PY -I -B check_method.py            # method.py reproduces the links fit_loco.py scored; groups consistent
$PY -I -B check_labelfree.py         # audit hook: the apply path opens no annotation and imports no label module
for p in strict balanced recall fixed_strict fixed_balanced fixed_recall frozen_strict frozen_balanced frozen_recall cuts_only; do
  $PY -I C:/Research/plugins/tools/reid_lab/score.py METHOD_$p.json OUT_$p.json; done        # out/cli_scores.txt
$PY -I -B diag_points.py ; $PY -I -B diag_baselines.py      # diagnosis with labels -> out/diag_points.md, out/diag_baselines.json
$PY -I -B uncertainty.py             # clip bootstrap, paired with the baselines -> out/uncertainty.md
$PY -I -B sensitivity.py 4 heldout ; $PY -I -B sensitivity.py 4 fixed     # out/sensitivity_*.md (3 min each)
$PY -I -B selfcheck.py               # label-free -> out/selfcheck.md
$PY -I -B stress.py all 4            # out/stress.md (about 20 min; writes cache_stress/, which can be deleted afterwards)
$PY -I -B method.py cand 0.5 off     # METHOD_cand_m0.50_off.json: any fixed setting of the trade-off table
```

A second complete run of `fit_loco.py` gave byte-identical `params_loco.json`, `out/loco_report.json` and
`out/tradeoff.md` (the first run's copies are in `out/run1/`). Under load two diagnosis commands returned exit
code 127 with no output (the interpreter did not start) and ran normally when repeated; every command above
finished with exit code 0, and loops should check the exit code.

## Files

| file | labels? | what |
|---|---|---|
| `PREREG.txt` | no | method changes and rule fixed before the first run, with what I knew |
| `prep.py`, `evidence.py` | no | fuse's files + solid-only tables (`n_solid`) |
| `linker.py`, `pairmodel.py` | no / fit yes | fuse's files, unchanged (path string only) |
| `linker2.py` | no | the two-stage linker |
| `method.py` | no | apply path, METHOD files from `params_loco.json` |
| `rules.py` | no (works on scored counts) | rule B and the alternative rules |
| `fuse_labels.py` | yes | identities from `reid_lab.score`'s matching: fit clips and diagnosis only |
| `fit_loco.py` | yes | the LOCO run (phases A evidence, B fit, C links, D scoring) |
| `selfcheck.py` | no | label-free check of the appearance evidence on a clip |
| `stress.py`, `sensitivity.py`, `uncertainty.py`, `diag_points.py`, `diag_baselines.py` | yes (scoring / diagnosis) | analyses above |
| `check_linker2.py`, `check_method.py`, `check_labelfree.py`, `check_r0.py` | no | checks |
| `params_loco.json` | fitted | per fold and `all13`: pair model, chosen settings |
| `METHOD_*.json`, `OUT_*.json`, `out/` | | METHOD files, the CLI's full output, tables and logs |
| `report-fuse_r1.html` | | visual summary (Chinese) for the project lead |

For a new clip: `method.clip_method(seq, margin, leaves, params["all13"], None)` with the frozen setting
of a point (`params["all13"]["choice"]`). `prep.py` only accepts dev clips today because `reid_lab.data` does.

## Caveats

- 13 clips, and one of them (SNGS-162) holds most of the wrong returns at every point. No limit can be
  certified from this sample; the 37 reserved clips are the clean test and were not touched.
- LOCO protects the fitted numbers (pair model, settings, m_cuts' and m_space's weights), not the design.
  The tiers, the boundary of 5 boxes, the solid-only position table, leaves and rule B were fixed in
  advance for this round, but with round 0's held-out tables and v_robust's guard tables known. The
  operating points move with the rule (table above).
- Balanced is unstable: the folds choose margins from 0.15 to 0.50 and only 4 choose the frozen 0.40.
- The tiers do not move the trade-off curve as a whole; at the loose end round 0's linker gives about the
  same counts. What they change is the strict end, the bridging through fragments, and that the links of
  the strict and balanced points are nearly all between pieces score.py can judge.
- Leaves bring 17 more right returns at the recall point and 2 more wrong ones, both in SNGS-162, and
  about one in ten of the judgeable leaf links disagrees by weak labels.
- Not done: choosing the pair-model inputs inside the folds (v_leak found a different set wins the inner
  log-loss with a comparable curve); doubly held-out stacking (m_cuts' and m_space's weights of a fold are
  in-sample for its 12 fit clips); m_space's 30-second "time in clip" prior is unchanged; shot cuts are not
  handled (none in the dev clips).
- The suspected annotation errors in SNGS-162 (and SNGS-100, SNGS-112) are numeric signs only.
- The stress test is synthetic: it shows the direction of failure, not its size on real unseen footage.
- No bug found in `reid_lab/score.py` or `data.py`. Nothing outside this directory was written.
