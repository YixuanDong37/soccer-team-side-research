# v_leak (round 1): leakage and protocol check of work/fuse_r1

Question: does anything in fuse_r1 (and the modules of m_cuts, m_appear, m_space it imports) let the
held-out clip of a leave-one-clip-out fold influence its own result, and how much of the reported
held-out result rests on choices made with the pooled held-out results of these 13 clips in view?

Everything here is on the 13 dev clips only, numbers only, written only into this directory (checked at
the end: no file under `reid_lab/` outside the round-1 verifier folders is newer than the start of this
work). Python: gbf's venv with `-I -B` (`-B` and `sys.dont_write_bytecode`, so importing the other agents'
modules leaves no `.pyc` in their folders). No git, no network, no image or video, no GPU.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_leak_1
```

## Verdict

**No label leak.** A re-run of all 13 folds in processes that cannot read the held-out clip's annotation
and that read no fitted or cached file of fuse_r1 or of any method agent reproduces fuse_r1 exactly: pair
models, the settings chosen by every rule, the held-out links at all 32 candidate settings, and the three
headline results (strict 67 / 3 / 67, balanced 85 / 5 / 47, recall 106 / 7 / 24).

**But the strict point's "wrong <= 4" does not survive when two choices that were made with SNGS-162's
held-out errors in view are handed back to the folds:**

- the tiers (solid from 5 boxes) remove wrong returns in SNGS-162 only; a fold that does not see SNGS-162
  prefers round 0's linker, and the pooled strict result is then 70 / 5 / 62 (limit 4 broken);
- rule B is in effect "largest J on the fit clips other than SNGS-162": its cap never binds, and naming
  SNGS-162 gives the same settings. With the rule chosen inside the folds the strict point is 71 / 8 / 58
  or 46 / 2 / 89, depending on the criterion.

fuse_r1 says that its design and rule were informed by round 0's held-out tables and calls the size of
the optimism unknown; the numbers below are that size.

## 1. Static check (`s0_static.py` -> `out/static.md`)

```
$PY -I -B s0_static.py
```

- Import graph (ast) of the apply path: fuse_r1 `prep.py`, `evidence.py`, `linker.py`, `linker2.py`,
  `method.py`, `pairmodel.py`, `selfcheck.py`; m_cuts `features.py`; m_appear `features.py`, `core.py`,
  `agglo.py`; m_space `features.py`, `geom.py`, `linker.py`. None imports `reid_lab.score`, `gbf.evaluate`,
  `fuse_labels` or a labels module at module level, and none names an annotation file. The one label
  import is m_space `linker.py`'s `fitdata`, inside `fit` and `heldout_nll` (fit functions; the apply path
  calls `posteriors` and `kind`).
- `reid_lab.data` inputs carry no annotation column (tracks: boxes, plugin role / team / jersey, pitch
  position; calibration; refine.json; embeddings). The pipeline behind them (`tools/gsr/provenance-gsr.json`)
  uses no ground truth.
- m_appear's team centring and whitening are computed per clip from that clip's own unlabelled rows; the
  pair model's and m_cuts' standardisation come from the fit clips.
- The fitted files a fold reads (`m_cuts/loco_params.json[fold]`, `m_space/params_cache/excl_<fold>.json`)
  are equal (largest difference 0.0) to round 0's blindfolded re-fits (`v_leak_0/refit/`) and have not been
  modified since 04:15 / 04:28; the cut lists in `fuse_r1/cache/cuts.json` equal the re-fit's for all
  13 x 13 (fold, clip) pairs.
- Time line: `PREREG.txt` 07:28:49, `rules.py` 07:34:38, `fit_loco.py` 07:35:37, first log of a
  label-using fuse_r1 script 07:45:15, METHOD files 07:46; `rules.py` and `fit_loco.py` were not changed
  after the first result. Consistent with "the rule was fixed before the first fuse_r1 result"; file times
  cannot prove that nothing was run and discarded before.
- No clip name or clip-specific number in method code (only in report code of `fit_loco.py`).

## 2. Blindfolded re-run of all 13 folds (the main test)

`blind.py` (round 0's, unchanged) makes every door to one clip's annotation raise in the running process
(`open` / `io.open` / pandas / numpy readers of its `Labels-GameState.json` or `reentry-eval.json`,
`score.clip`, `score.score_clip`, `gsr.load_reference`; a self-test proves the doors are closed) and logs
every file that is opened.

```
$PY -I -B s2_tables.py 4                              # evidence tables from nothing with EVERY clip's annotation blindfolded -> cache/tables (83 s; 32 distinct (clip, cut list) tables x app / space / ends for all pieces and solid from 5, 3, 10 boxes)
$PY -I -B driver.py s3_blind_r1.py s3_blind_r1 4      # one process per fold, held-out clip blindfolded -> out/blind/<fold>.pkl (3.5 to 27 min per fold, depending on the load of the machine)
$PY -I -B s4_compare.py                               # against fuse_r1's published files + scoring -> out/compare.md, out/METHOD_blind_*.json, out/blind_folds.pkl
for p in strict balanced recall; do $PY -I -B C:/Research/plugins/tools/reid_lab/score.py out/METHOD_blind_$p.json out/OUT_blind_$p.json; done   # out/cli_blind.txt
$PY -I -B s5_repeat_check.py                          # two folds were run twice: identical -> out/repeat_check.txt
```

A fold's process (`s3_blind_r1.py`) takes the cuts and position weights from round 0's blindfolded re-fits
of m_cuts and m_space (`v_leak_0/refit/`, shown above to equal what fuse_r1 reads), the tables from
`cache/tables`, and runs fuse_r1's own code unchanged: `evidence.pair_table`, `pairmodel.fit` on the fit
clips' weak labels, `linker2.link` / `linker.agglomerate` for all 13 clips, `fit_loco._scored` on the 12 fit
clips, `rules.choose`. The held-out clip gets links for every candidate setting and is not scored there.

Result (`out/compare.md`), the same for every one of the 13 folds:

| check | result |
|---|---|
| pair model (tiers) and round 0's pair model against `params_loco.json` / `loco_raw.pkl` | max abs difference 0.0 |
| settings chosen by rule B (strict / balanced / recall) | equal |
| settings chosen by all four rules, both linkers, against `out/loco_report.json` | equal |
| held-out links at the three points against `METHOD_strict / balanced / recall.json`, and the cuts | equal |
| held-out links at all 32 settings (16 margins x leaves off / on) and round 0's linker at 16 margins against `cache/loco_raw.pkl` | equal |
| scored cells of the 12 fit clips (every setting) against `cache/loco_raw.pkl` | equal |
| annotation files opened by the fold's process | 12 (the fit clips); the held-out clip's: 0; blocked attempts: 0 |
| fitted or cached files of other agents read by the fold's process | 0 (only source code, `reid_lab.data` inputs, `v_leak_0/refit/`, this directory's tables) |

score.py CLI on the blind METHOD files (`out/cli_blind.txt`): strict 67 / 3 / 67 | 312 / 7 / 25 | 1.611%;
balanced 85 / 5 / 47 | 378 / 16 / 30 | 2.123%; recall 106 / 7 / 24 | 414 / 17 / 260 | 2.175%: fuse_r1's
numbers. Folds SNGS-162 and SNGS-112 were each run twice in separate processes (once with a per-fold table
cache, once after the other folds): identical models, links, cells and choices.

So fuse_r1's cached tables (copied from round 0), its cut lists, its fold weights and its LOCO run hold no
information from the held-out clip's labels.

## 3. Choices made with the pooled held-out results in view, repeated inside the folds

```
$PY -I -B n1_nested.py blind      # -> out/nested.md, out/nested.json   (cells of the blindfolded re-run; variants r1, r0, n3, n10, c5)
$PY -I -B n1_nested.py fuse_r1    # -> out/nested_fuse_r1.md            (same analysis on fuse_r1/cache/loco_raw.pkl, variants r1 and r0: same numbers)
```

The blind folds also ran four design alternatives next to the method (r1), each with its own pair model
fitted on the fit clips: **r0** round 0's linker (no tiers), **n3 / n10** solid from 3 / 10 boxes (the
values PREREG.txt names), **c5** "a rival must share >= 5 frames with the partner" (m_appear fixed ">= 1"
after looking at all 13 clips). `n_common.choose_multi` is fuse_r1's rule with candidates from several
variants; with r1 alone it reproduces `rules.choose` (same settings in all folds). The held-out clip's cell
is read only after the fold's choice. Right / wrong / lost of 137, pooled over the 13 held-out clips.

### 3a. The tiers (major)

| offered to rule B inside each fold | strict (limit 4) | balanced (limit 7) | recall (limit 14) |
|---|---|---|---|
| r1 only = fuse_r1 | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 |
| r1 or r0 (tiers or not) | **70 / 5 / 62** (12 folds r1; the fold that holds SNGS-162 out takes r0 0.60 and gets 6 / 4 / 8 there instead of 3 / 2 / 13) | 108 / 7 / 22 (r0 in all 13 folds) | 111 / 8 / 18 (r0 in all 13 folds) |
| solid from 5 / 3 / 10 boxes | 65 / 3 / 69 (r1 in 11 folds, n10 in 2) | 88 / 6 / 43 (n3 in all 13 folds) | 108 / 9 / 20 (n3 in all 13 folds) |
| rival shares >= 1 or >= 5 frames | 73 / 3 / 61 (c5 in all 13 folds) | 75 / 4 / 58 (c5 in 12 folds) | 107 / 8 / 22 (r1 in 12 folds) |
| all five | **76 / 5 / 56** (c5 in 12 folds, r0 in the fold that holds SNGS-162 out) | 108 / 7 / 22 (r0 in all 13) | 111 / 8 / 18 (r0 in all 13) |

Why: at fixed settings with held-out weights (leaves off) the variants differ like this:

| margin | r1: 12 other clips | r1: SNGS-162 | r0: 12 other clips | r0: SNGS-162 | n3: 12 other clips | n3: SNGS-162 |
|---|---|---|---|---|---|---|
| 0.40 | 81 / 1 / 37 | 4 / 3 / 11 | 90 / 1 / 28 | 7 / 5 / 6 | 84 / 1 / 34 | 4 / 4 / 10 |
| 0.50 | 78 / 1 / 40 | 3 / 3 / 12 | 84 / 0 / 35 | 7 / 4 / 7 | 79 / 1 / 39 | 4 / 4 / 10 |
| 0.60 | 67 / 0 / 52 | 3 / 2 / 13 | 72 / 0 / 47 | 6 / 4 / 8 | 68 / 0 / 51 | 4 / 4 / 10 |
| 0.70 | 55 / 0 / 64 | 3 / 2 / 13 | 58 / 0 / 61 | 6 / 4 / 8 | 52 / 0 / 67 | 4 / 4 / 10 |

On the 12 clips other than SNGS-162 the tiers remove no wrong return at any margin and cost 3 to 9 right
returns; "solid from 3" and "solid from 5" have the same wrong counts there. Every wrong return the tiers
and the 5-box boundary save (4 -> 2 at margins 0.55 to 0.70) is in SNGS-162, the clip whose held-out
errors in round 0 (v_robust_0: two wrong returns through one 3-box piece; guard table "rows >= 5") led to
the design. In the fold that holds SNGS-162 out the fit clips cannot show that (r0: 66 right, 0 wrong
returns, 1 wrong link on the fit clips at margin 0.60; r1: 63, 0, 1), so rule B takes r0. The held-out
3 / 2 / 13 of SNGS-162 in fuse_r1's strict result is therefore in-sample for the design, and the nested
value of the strict point is 5 wrong, one over the limit. At balanced and recall the nested design keeps
the limits (7 and 8 wrong) and finds more returns than fuse_r1's points, as fuse_r1 reports for "rule B on
round 0's linker".

"A rival shares >= 1 frame" is not a problem in this method: offered to the folds, c5 is taken at the
strict point and gives 73 / 3 / 61.

### 3b. The rule for the operating points (major)

| what | strict | balanced | recall |
|---|---|---|---|
| rule B (PREREG.txt) | 67 / 3 / 67 | 85 / 5 / 47 | 106 / 7 / 24 |
| rule B without its cap (part (i) removed) | 67 / 3 / 67, same setting in 13 of 13 folds | 85 / 5 / 47, 13 of 13 | 106 / 7 / 24, 13 of 13 |
| "largest J without SNGS-162" (the clip named; nothing left out in its own fold) | 67 / 3 / 67, same setting in 13 of 13 folds | 85 / 5 / 47, 12 of 13 | 106 / 7 / 24, 13 of 13 |
| R0 = Rnotrim (J, no clip left out) | 26 / 2 / 109 | 72 / 4 / 61 | 97 / 8 / 32 |
| Rcap (most right under the cap) | 78 / 8 / 51 | 98 / 9 / 30 | 112 / 9 / 16 |
| rule chosen inside each fold by an inner leave-one-clip-out over its 12 fit clips: most right with the inner wrong returns within the cap | 71 / 8 / 58 (11x B, 1x R0, 1x Rcap: the fold that holds SNGS-162 out) | 87 / 8 / 42 | 112 / 9 / 16 (13x Rcap) |
| the same, criterion largest inner right - lam x errors | 46 / 2 / 89 (8x R0, 5x B) | 75 / 4 / 58 | 105 / 7 / 25 |

- The cap (i) of rule B ("wrong returns on the fit clips <= floor(K x scored_fit / 137)") never binds: with
  the cap removed no fold changes its setting at any point. The limits 4 / 7 / 14 are therefore not enforced
  by the rule; they hold in the pooled result because of the penalty lam and the trimming. (For the same
  reason the constant 137 in the cap, which contains the held-out clip's number of scored returns, has no
  effect: caps of K x 12 / 13, K - 1 or K give the same settings.)
- The clip rule B leaves out of J is SNGS-162 in all 12 folds where it is a fit clip, and the rule gives
  the same settings as one that names SNGS-162. So in those 12 folds the errors of SNGS-162 have no
  influence on the threshold, and the frozen thresholds (all 13 clips) are tuned on the 12 clean clips.
  The rule is generic in wording and clip-specific in effect on these data.
- With the trimming the strict point needs lam between 5 and 10 to stay within 4 wrong (lam 5: 72 / 4,
  7: 71 / 3, 10: 67 / 3, 15: 58 / 2; lam 3: 73 / 5, 2: 74 / 5, 1: 78 / 7); without it lam >= 5 gives
  25-35 right / 2 wrong (`out/nested.md`, section D).
- The inner selection uses the fold's weights for the inner clips too (only the choice of the setting is
  nested), so it is an approximation; it needs nothing fuse_r1 did not already compute.

Rule B was written down before any fuse_r1 result existed (section 1), so it was not picked on fuse_r1's
pooled table; it was taken from round 0's post-hoc analysis of the same 13 clips with a linker whose
fixed-margin curve is about the same. The 37 reserved clips are the only clean test of it.

## 4. What remains open or minor

- **PRTReid's training data (major, unverified, unchanged from round 0).** The plugin loads
  `prtreid-soccernet-baseline.pth.tar`; gbf's registry says "training_data = Not reviewed". If it was
  trained on the SN-GSR train split, as the upstream baseline's was to my knowledge, the embeddings (and
  the plugin's role and team labels) have seen these players, and neither LOCO nor the 37 reserved train
  clips can show it. Nothing on this machine that I may read settles it.
- Design constants fixed by their authors with all 13 clips in view and not nested here: the pair-model
  inputs (round 0: nested, comparable curve), m_cuts' feature set and P >= 0.7 (round 0: negligible),
  m_space's model configuration and hand-set constants, m_appear's whitening shrinkage and 5 clean rows,
  floor 0.3 and "nobody" 0.5 (probed in round 0 on the fit clips of one fold).
- Within a fold m_cuts' and m_space's weights are in-sample for the 12 fit clips and out-of-sample for the
  held-out clip (no doubly held-out stacking). This cannot leak the held-out clip.
- `fuse_labels.py` takes labels from `score.clip` and the private `score._ids_after_cuts` (fit and
  diagnosis code, not apply code).
- m_space's "nobody" terms use the time since the clip started (30 s clips).
- I found no bug in `reid_lab/score.py` or `data.py`.

## What went wrong on the way

- Under load (other agents' jobs, commit memory nearly exhausted) the interpreter sometimes did not start
  (exit code 127, no output); the first background attempt of the test fold produced nothing. Commands were
  repeated until they ran; every result above comes from a run with exit code 0.
- The first test of `s3_blind_r1.py` (fold SNGS-162, its own table cache; the script then built the tables
  itself, one line differs from the final script) took 12 minutes; the tables were then moved to one
  label-free stage (`s2_tables.py`). That test's result is kept in `out/test_per_fold_cache/` and equals
  the final run's.
- To save time fold SNGS-112 was started by a second driver call while the first was busy
  (`out/s3_driver_extra.stdout`); the first driver ran it again at its end. Both results are identical
  (`out/repeat_check.txt`); `s4_compare.py` and `n1_nested.py` were re-run after the last fold and gave
  byte-identical outputs.

## Files

| file | labels? | what |
|---|---|---|
| `blind.py`, `driver.py` | no | round 0's blindfold / file-open audit and one-process-per-fold driver (path changed) |
| `s0_static.py` | no | import graph, fitted files against round 0's re-fits, time line |
| `s2_tables.py` | no (all clips blindfolded) | evidence tables from nothing |
| `s3_blind_r1.py` | fit clips only (held-out clip blindfolded) | one fold of fuse_r1 and of the design alternatives |
| `s4_compare.py` | scoring | comparison with fuse_r1, scoring of the blind held-out links, cells for the nested analyses |
| `s5_repeat_check.py` | no | repeated folds identical |
| `n_common.py`, `n1_nested.py` | no (scored cells) | choices repeated inside the folds |
| `out/static.md`, `out/compare.md`, `out/nested.md`, `out/nested_fuse_r1.md` | | the tables quoted above (`.json` next to them) |
| `out/METHOD_blind_<point>.json`, `out/OUT_blind_<point>.json`, `out/cli_blind.txt` | | fuse_r1's held-out links reproduced blind, scored by the CLI |
| `out/blind/<fold>.pkl`, `out/blind_folds.pkl` | | per fold: models, links of the held-out clip, scored cells, choices, audit |
| `out/logs/`, `out/*.stdout`, `out/s2_tables.log`, `out/s3_blind_r1.log` | | logs |
| `out/test_per_fold_cache/`, `out/preview_162.txt` | | the first test fold (per-fold table cache), the first run of fold SNGS-112, and an early look at fold SNGS-162 from an inline snippet (superseded by `out/nested.md`) |

No METHOD.json of my own: `out/METHOD_blind_*.json` are fuse_r1's links, and the nested rows are analyses,
not proposed methods. `cache/tables/` (174 MB) was deleted after the run; `s2_tables.py` rebuilds it.
