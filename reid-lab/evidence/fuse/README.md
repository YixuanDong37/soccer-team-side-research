# fuse: one re-linking method from the evidence of m_cuts, m_appear, m_space and m_keep

Everything here works on the 13 SN-GSR dev clips only, reads the tracking outputs through
`reid_lab/data.py` and the other agents' label-free modules, and writes only into this directory.
No image or video was opened. Python: gbf's venv, always with `-I`.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/fuse
```

## The rule in plain words

1. **Cut first** (m_cuts, P(switch) >= 0.7): wherever the plugin's track moved onto another person
   inside a piece, the piece is cut. Everything below works on the cut pieces.
2. **Evidence for two pieces** that could be one person (same kit, never in the same frame, not a
   referee):
   - how alike they look: m_appear's cosine of the two prototypes after the team's average look
     is removed and the clip's frame-to-frame variation is whitened;
   - a **position bonus** when one piece would directly follow the other: m_space's assignment
     score for "the later piece starts where the earlier one can have got to" (it is high when few
     other pieces fit), turned into cosine units;
   - a small **plugin bonus** when the plugin itself tracked the two as one id and gbf separated them.
   The weights of the two bonuses come from a logistic regression on pairs (3 weights + bias).
3. **Join**, most alike first, two groups of pieces when
   - they look alike at all (appearance cosine >= 0.3),
   - each is the other's best partner,
   - the fused score (appearance + bonus) beats by the **margin** both 0.5 ("nobody") and the score
     to any group that is seen at the same time as the partner (such a group is certainly someone
     else), on both sides.
   Joined pieces pool their appearance and the search continues.
4. **It abstains** when another candidate who is visible together with the partner is nearly as
   good, when the two do not look alike, when position gives no support and the look alone is not
   clearly best, when the pieces share a frame or differ in team, and for referees.

One number sets how sure a join must be: the margin (large = strict).

## Results (137 scored returns, 13 dev clips, leave one clip out, pooled)

Numbers of `reid_lab/score.py`'s CLI on the METHOD files (`out/cli_scores.txt`):

| operating point (rule fixed in advance) | right | wrong | lost | links right / wrong / unknown | link precision | impure rows |
|---|---|---|---|---|---|---|
| gbf now (split) | 0 | 4 | 133 | 0 / 0 / 0 | - | 2.61% |
| plugin joins | 30 | 14 | 93 | 33 / 10 / 50 | 0.767 | 3.08% |
| mutual baseline | 26 | 6 | 105 | 60 / 7 / 2 | 0.895 | 3.03% |
| **strict** `METHOD_strict.json` (lam 10) | 33 | 4 | 100 | 152 / 4 / 158 | 0.974 | 1.28% |
| **balanced** `METHOD_balanced.json` (lam 3) | 93 | 7 | 37 | 290 / 10 / 376 | 0.967 | 1.89% |
| **recall** `METHOD_recall.json` (lam 1) | 110 | 7 | 20 | 314 / 10 / 437 | 0.969 | 1.91% |
| in-sample, frozen strict `METHOD_frozen_strict.json` | 37 | 0 | 100 | 152 / 2 / 152 | 0.987 | 1.05% |
| in-sample, frozen balanced `METHOD_frozen_balanced.json` | 109 | 6 | 22 | 310 / 10 / 424 | 0.969 | 1.91% |
| in-sample, frozen recall `METHOD_frozen_recall.json` | 112 | 7 | 18 | 314 / 11 / 441 | 0.966 | 2.13% |

All three points keep the asked limits (wrong <= 4 / 7 / 14). The strict point is far more cautious
than its limit needs, and the balanced point is dominated by the recall point (same 7 wrong); see
"How the operating points were chosen" and the fixed-margin table.

### How the operating points were chosen (fixed before the pooled held-out result was looked at)

Per fold, on the 12 fit clips, the margin with the largest
`J = right - lam * (wrong returns + wrong links)`, ties to the larger margin; `lam` = 10 (strict),
3 (balanced), 1 (recall). This is m_appear's and m_keep's objective with their penalties. The pair
model is fitted on the same 12 clips; m_cuts' and m_space's weights are the ones those agents fitted
without the held-out clip (`m_cuts/loco_params.json[clip]`, `m_space/params_cache/excl_<clip>.json`)
and are used for all 13 clips of the fold, so nothing in a fold has seen its held-out clip.
Margins chosen: strict 10x 0.9, 2x 1.0, 1x 0.6; balanced 7x 0.5, 6x 0.25; recall 10x 0.2, 3x 0.25.
On all 13 clips (the frozen method): 0.9 / 0.25 / 0.2.

What the rule did, honestly:
- **strict** lands at margin 0.9 because the links in SNGS-162 that the annotation calls wrong cost
  10 each; only the fold that holds SNGS-162 out chooses 0.6, and its 4 wrong returns are the 4 of
  the pooled result.
- **balanced** flips between 0.5 and 0.25 (J differs by 1-5); the mix gives 93 / 7.
- The secondary rule "most right with wrong <= floor(cap * scored_fit / 137) on the fit clips"
  (computed as a check, `out/tradeoff.md`) gives strict 53 / 6 / 78 (limit broken: the fold that holds
  SNGS-162 out sees 12 clean clips and picks the loosest margin), balanced 111 / 7 / 19, recall 116 / 9 / 12.
- POST HOC, not a result: the same J without the fit clip that has the most wrong (returns + links)
  gives strict 88 / 4 / 45 (margins 0.5-0.6), balanced 110 / 7 / 20, recall 112 / 8 / 17 (`out/d7_after.txt`).

### Trade-off: fixed margin (nothing chosen; each clip with the weights fitted without it)

| margin | right | wrong | lost | links right / wrong / unknown | impure rows | in-sample (all-13 weights) |
|---|---|---|---|---|---|---|
| 0.10 | 116 | 9 | 12 | 330 / 15 / 482 | 2.31% | 116 / 11 / 10 |
| 0.15 | 114 | 7 | 16 | 326 / 12 / 459 | 2.19% | 113 / 8 / 16 |
| 0.20 | 112 | 7 | 18 | 316 / 10 / 438 | 1.91% | 112 / 7 / 18 |
| 0.25 | 110 | 7 | 20 | 315 / 10 / 424 | 1.91% | 109 / 6 / 22 |
| 0.30 | 105 | 6 | 26 | 308 / 10 / 403 | 1.91% | 103 / 6 / 28 |
| 0.35 | 104 | 6 | 27 | 301 / 11 / 392 | 1.86% | 101 / 6 / 30 |
| 0.40 | 97 | 6 | 34 | 288 / 11 / 369 | 1.84% | 98 / 6 / 33 |
| 0.45 | 94 | 5 | 38 | 278 / 9 / 356 | 1.66% | 93 / 6 / 38 |
| 0.50 | 91 | 4 | 42 | 272 / 7 / 345 | 1.51% | 90 / 4 / 43 |
| 0.55 | 84 | 4 | 49 | 262 / 6 / 326 | 1.42% | 83 / 4 / 50 |
| 0.60 | 78 | 4 | 55 | 254 / 5 / 304 | 1.41% | 79 / 4 / 54 |
| 0.70 | 64 | 4 | 69 | 222 / 4 / 263 | 1.29% | 64 / 4 / 69 |
| 0.80 | 51 | 1 | 85 | 196 / 3 / 216 | 1.20% | 51 / 1 / 85 |
| 0.90 | 31 | 0 | 106 | 145 / 2 / 147 | 1.05% | 37 / 0 / 100 |
| 1.00 | 21 | 0 | 116 | 101 / 2 / 85 | 1.05% | 20 / 0 / 117 |
| 1.20 | 2 | 0 | 135 | 16 / 0 / 19 | 1.00% | 2 / 0 / 135 |

These rows are honest held-out numbers for a threshold fixed beforehand; picking the best row now
would be a choice made on the held-out result. `python -I method.py margin 0.5` writes the METHOD file
of any row.

### What each part adds (same LOCO, same rule; `out/tradeoff.md` has every row)

| variant | fixed margin, held-out | strict / balanced / recall (lam rule), held-out |
|---|---|---|
| **fuse** (cuts + appearance + position + plugin) | 0.5: 91 / 4 / 42; 0.25: 110 / 7 / 20 | 33 / 4, 93 / 7, 110 / 7 |
| fuse_pos (no plugin bonus) | 0.5: 89 / 4 / 44; 0.25: 110 / 7 / 20; 0.8: 43 / 1 (fuse 51 / 1) | 29 / 4, 91 / 7, 110 / 7 |
| app_cut (cuts + m_appear's rule, no bonus) | 0.30: 86 / 5 / 46; 0.20: 97 / 8 / 32 | 68 / 5, 86 / 7, 92 / 10 |
| fuse_nocut (no cuts) | 0.5: 63 / 5 / 69; 0.30: 91 / 8 / 38, impure 2.97% | 46 / 10, 85 / 11, 92 / 11 |
| app_nocut (= m_appear, margin only) | 0.30: 71 / 6 / 60; 0.25: 75 / 6 / 56 | 71 / 6, 71 / 6, 76 / 10 |

app_nocut reproduces m_appear's frozen points exactly (71 / 6 / 60 with links 220 / 1 / 174 and
75 / 6 / 56 with 238 / 2 / 198).

### Per clip (held-out; right / wrong / lost, links right / wrong / unknown, margin used)

| clip | strict | balanced | recall |
|---|---|---|---|
| SNGS-107 | 4 / 0 / 12, 13 / 0 / 16 (0.9) | 10 / 0 / 6, 20 / 0 / 35 (0.5) | 15 / 0 / 1, 22 / 0 / 45 (0.25) |
| SNGS-163 | 1 / 0 / 10, 8 / 0 / 8 (0.9) | 5 / 0 / 6, 17 / 0 / 22 (0.5) | 8 / 0 / 3, 23 / 0 / 29 (0.25) |
| SNGS-106 | 3 / 0 / 4, 6 / 0 / 11 (0.9) | 6 / 0 / 1, 24 / 0 / 38 (0.25) | 6 / 0 / 1, 24 / 0 / 38 (0.2) |
| SNGS-097 | 3 / 0 / 3, 13 / 0 / 8 (0.9) | 6 / 0 / 0, 20 / 1 / 22 (0.25) | 6 / 0 / 0, 20 / 1 / 23 (0.2) |
| SNGS-067 | 2 / 0 / 8, 14 / 0 / 13 (0.9) | 8 / 0 / 2, 31 / 0 / 33 (0.5) | 9 / 0 / 1, 35 / 0 / 40 (0.2) |
| SNGS-100 | 2 / 0 / 2, 19 / 0 / 1 (0.9) | 3 / 0 / 1, 30 / 2 / 18 (0.25) | 3 / 0 / 1, 29 / 2 / 20 (0.2) |
| SNGS-066 | 2 / 0 / 9, 8 / 0 / 5 (0.9) | 6 / 1 / 4, 22 / 1 / 20 (0.25) | 6 / 1 / 4, 24 / 1 / 20 (0.2) |
| SNGS-161 | 4 / 0 / 10, 13 / 0 / 6 (0.9) | 13 / 0 / 1, 28 / 0 / 9 (0.5) | 14 / 0 / 0, 27 / 0 / 14 (0.2) |
| SNGS-169 | 2 / 0 / 13, 6 / 0 / 15 (1.0) | 12 / 0 / 3, 20 / 0 / 47 (0.5) | 15 / 0 / 0, 26 / 0 / 59 (0.2) |
| SNGS-162 | 6 / 4 / 8, 26 / 3 / 40 (0.6) | 7 / 6 / 5, 33 / 5 / 53 (0.25) | 7 / 6 / 5, 32 / 5 / 56 (0.2) |
| SNGS-151 | 0 / 0 / 9, 1 / 0 / 12 (1.0) | 4 / 0 / 5, 6 / 0 / 35 (0.5) | 7 / 0 / 2, 9 / 0 / 44 (0.25) |
| SNGS-102 | 2 / 0 / 10, 10 / 0 / 10 (0.9) | 10 / 0 / 2, 21 / 0 / 18 (0.25) | 10 / 0 / 2, 21 / 0 / 19 (0.2) |
| SNGS-112 | 2 / 0 / 2, 15 / 1 / 13 (0.9) | 3 / 0 / 1, 18 / 1 / 26 (0.5) | 4 / 0 / 0, 22 / 1 / 30 (0.2) |

Clip bootstrap of the held-out shares (13 clips resampled, 5% / 50% / 95%): strict right 0.19 / 0.24 /
0.30, wrong 0 / 0.03 / 0.08; balanced right 0.58 / 0.68 / 0.78, wrong 0 / 0.05 / 0.12; recall right
0.69 / 0.81 / 0.91, wrong 0 / 0.05 / 0.12.

### SNGS-162 (read with care; diagnosis, `out/d7_after.txt`, `out/wrong_links.txt`)

- Without SNGS-162 (12 clips, 119 scored returns): strict 27 / 0 / 92, balanced 86 / 1 / 32,
  recall 103 / 1 / 15; fixed margin 0.5: 84 / 0 / 35, 0.25: 103 / 1 / 15, 0.10: 108 / 3 / 8.
  SNGS-162 alone (18 scored): 7 right, 4-6 wrong at every margin from 0.10 to 0.70.
- The wrong returns: strict 4, all SNGS-162 (annotated persons 2, 8, 9, 12); balanced and recall 7:
  those 4, SNGS-162 persons 19 and 21, SNGS-066 person 21.
- Persons 9, 12 and 8 (and the spot of person 2 at frames 732-734) are among the cases that
  diag_failures and m_appear list as numerically suspect annotations (identities exchanged after
  the gap). Nobody has looked at the frames. Of the 10 links score.py calls wrong at the recall
  point, 5 are on those suspect lists (SNGS-162 4>96, 10000030>93, 27>114; SNGS-100 27>63;
  SNGS-112 49>100049).
- Impure rows added by linking over cuts alone (1.00%) sit in few clips: SNGS-162 +552 rows at the
  recall point, SNGS-097 +172, SNGS-100 +116, SNGS-066 +83, SNGS-112 +70, the other eight clips
  +0 to +4 (`out/impure_per_clip.txt`).

### The links score.py cannot judge (a piece with < 5 matched rows)

Judged by the majority person of however few matched rows (diagnosis): strict 118 agree, 3 disagree,
37 with a piece that matches nobody; balanced 275 / 22 / 79; recall 308 / 30 / 99. So about 92% of
the judgeable short-piece links agree, against 97% for the links score.py judges. Their rows are
inside impure_row_share.

### Sensitivity to the constants set by hand (`out/sensitivity.md`; held-out, fixed margin)

| setting | margin 0.25 | margin 0.5 | margin 0.9 |
|---|---|---|---|
| base | 110 / 7 / 20 | 91 / 4 / 42 | 31 / 0 / 106 |
| no appearance floor | 110 / 8 / 19 | 91 / 4 / 42 | 31 / 0 / 106 |
| floor 0.4 | 108 / 7 / 22 | 90 / 4 / 43 | 31 / 0 / 106 |
| no "nobody" rival | 110 / 7 / 20 | 92 / 4 / 41 | 45 / 0 / 92 |
| bonus x 0.5 | 105 / 6 / 26 | 59 / 4 / 74 | 0 / 0 / 137 |
| bonus x 1.5 | 110 / 5 / 22 | 88 / 5 / 44 | 55 / 1 / 81 |
| cuts at P >= 0.5 | 110 / 8 / 19 | 93 / 4 / 40 | 38 / 0 / 99 |
| cuts at P >= 0.9 | 107 / 6 / 24 | 86 / 5 / 46 | 27 / 1 / 109 |
| pair model on firm labels (>= 5 rows) | 109 / 6 / 22 | 75 / 4 / 58 | 4 / 0 / 133 |
| pair model with the raw position score | 98 / 5 / 34 | 69 / 4 / 64 | 24 / 0 / 113 |

The floor, the "nobody" rival and the cut level hardly matter. The size and form of the bonus move
a result along the margin axis (the margin is in units of appearance + bonus), and the raw position
score is clearly worse than its log-odds.

## Commands (in this order)

```
$PY -I prep.py 6                 # label-free tables on the cut pieces -> cache/ (about 3 min the first time)
$PY -I check_linker.py           # label-free: linker.py without a bonus = m_appear's agglomeration, all clips
$PY -I check_space.py native ; $PY -I check_space.py compare    # label-free: m_space on the uncut pieces through fuse = m_space natively
$PY -I fit_loco.py 6             # LOCO: pair model + margin per fold, every variant -> params_loco.json, out/tradeoff.md, out/loco_report.json (about 12 min)
$PY -I method.py                 # label-free: METHOD_strict/balanced/recall.json and METHOD_frozen_*.json from params_loco.json
$PY -I check_method.py           # method.py reproduces exactly the links fit_loco.py scored
$PY -I check_groups.py           # label-free: no group has two pieces in one frame, mixed teams, a referee or a second partner
for p in strict balanced recall frozen_strict frozen_balanced frozen_recall; do
  $PY -I C:/Research/plugins/tools/reid_lab/score.py METHOD_$p.json OUT_$p.json; done      # out/cli_scores.txt
$PY -I method.py margin 0.5      # METHOD_margin_0.50.json: any fixed-margin row
$PY -I sensitivity.py 6          # out/sensitivity.md (about 2 min)
$PY -I d7_after.py ; $PY -I t7_impure.py ; $PY -I t8_wronglinks.py     # diagnosis after the run
```

A second complete run of `fit_loco.py` gave byte-identical `params_loco.json`, `out/loco_report.json`
and `out/tradeoff.md`.

## Files

| file | labels? | what |
|---|---|---|
| `prep.py` | no | cut lists (m_cuts), m_appear's and m_space's pair tables on the cut pieces, cached; imports the other agents' modules under private names |
| `evidence.py` | no | one table per clip with all evidence per pair of pieces (`pair_table`) |
| `pairmodel.py` | apply no / fit yes | logistic pair model: `design`, `logit` (apply); `fit` gets labelled pairs from the caller |
| `linker.py` | no | the agglomeration on appearance + junction bonus |
| `method.py` | no | the apply path: `clip_method`, METHOD files from `params_loco.json` |
| `fuse_labels.py` | yes | identities of (sub-)pieces from `reid_lab.score`'s matching: fit clips and diagnosis only |
| `fit_loco.py` | yes | the LOCO run, the choice rule, tables, `params_loco.json` |
| `sensitivity.py` | yes | constants changed one at a time |
| `check_*.py` | no | checks listed above |
| `d1_look.py` ... `d7_after.py`, `t*.py`, `explore_engine.py` | yes | diagnosis and the exploration that led to the design (below); numbers are not method results |
| `params_loco.json` | fitted | per fold and `all13`: pair model, margins, which m_cuts / m_space weights |
| `METHOD_*.json`, `OUT_*.json` | | METHOD files and the CLI's full output |
| `report-fuse.html` | | visual summary (Chinese) for the project lead |

For a new clip: `method.clip_method(seq, margin, params["all13"], None)` (frozen numbers: m_cuts'
`final_params.json`, m_space's `params_all13.json`, the all-13 pair model). `prep.py` only accepts dev
clips today because `reid_lab.data` does.

## How the design came about (what was tried, what failed)

- Unions of the agents' held-out METHOD files (`t1_union_probe.py`): m_appear cautious + m_space
  strict + m_keep balanced = 86 / 6 on the uncut pieces. The evidence is complementary.
- Pair model inputs (`d2_pairmodel.py`, `d4_table2d.py`, LOCO log-loss on pairs): position adds
  (precision at recall 0.85: 0.56 -> 0.80, short pieces 0.42 -> 0.70); log-odds of m_space's
  assignment score beats the raw score (0.0412 against 0.0441); the plugin flag, a low-quality
  flag, an "out of reach" flag, the gap and a second cosine add nothing measurable. The plugin flag
  was kept because it adds returns at strict margins (0.8: 51 against 43).
- FAILED: the first engine searched m_appear's threshold range (t_abs 0.4-0.8, margin 0.1-0.4) for the
  fused similarity (`explore_engine.py`, `t3_onefold.py`). The bonus is often 0.3-0.8, so those
  thresholds were all loose: 41 wrong merges of 685 against 13 of 570 for appearance alone
  (`d5_merges.py`). The fused similarity needs margins of 0.2-1.0 (`d6_calib.py`, `t5_gridprobe.py`).
- FAILED as a cure: an absolute bar on the fused similarity (t_abs 1.0-1.2) or a high appearance
  floor (0.5-0.6) lose returns without removing wrong ones (`out/gridprobe_107_pos.txt`).
- Plugin joins without appearance or position are poor for short pieces (9 same against 16 different,
  `d3_fragments.py`); position at the border without appearance support is too (sink >= 0.5 and
  cosine < 0.2: 5 same, 14 different).
- Not tried: a full conditional-logit model over all candidates with appearance inside (m_space's
  linker extended), group-level position scores recomputed after every merge, jersey reads.

## Caveats

- 13 clips. LOCO protects the fitted numbers (pair model, margin, and the other agents' weights), not
  the design: the form of the bonus, the floor 0.3, the "nobody" rival at 0.5, the margin grid and the
  objective were chosen while looking at pair-level diagnosis on all 13 clips and at the 12 fit clips
  of the fold that holds SNGS-107 out (which are the held-out clips of the other folds). The 37
  reserved clips are the clean test.
- The limits "wrong <= 4 / 7 / 14" are kept, but the wrong count is ruled by one clip: SNGS-162 gives
  4-6 wrong at every useful margin and the other 12 clips together 0-1 (3 at the loosest margin).
- About 55% of the links are "unknown" to score.py; by weak labels they are less precise (about 92%)
  than the judged ones (97%).
- The margin lives on the scale of appearance + bonus; if the pair model is refitted on other data
  the scale moves with it (see the sensitivity rows "bonus x 0.5 / x 1.5").
- The position evidence needs per-frame calibration; pieces without a ground position get no bonus
  and are linked by appearance only. Shot cuts are not handled (none in the dev clips).
- m_space has no `cuts` argument: `prep.py` hands its modules a tracks table whose track_id holds the
  sub-piece ids by substituting `reid_lab.data.tracks` inside the process for the duration of the
  call. On the uncut pieces the result equals m_space's own (`check_space.py`). m_space's position
  weights were fitted on uncut pieces and are applied to cut pieces.
- Importing m_space's modules created `m_space/__pycache__` (Python bytecode); I removed it again
  and switched bytecode writing off in `prep.py`. Nothing else outside this directory was written.
- No bug found in `reid_lab/score.py` or `data.py`.
