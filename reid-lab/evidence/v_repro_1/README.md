# v_repro_1: reproduction check of work/fuse_r1 (verification round 1)

Everything here was written by v_repro. Nothing outside this directory was changed: `snapshot_before.tsv` /
`snapshot_after.tsv` hold size and mtime of every file under `reid_lab/` outside this directory; apart from the
two other verifiers' own directories (v_leak_1, v_robust_1, which were being written by them) there is no
difference (`out/snapshot_diff.txt` is empty), and no new `__pycache__` directory appeared. Only the 13 dev
clips were read. No image or video was opened, no git, no network, no install, no GPU.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_repro_1
```

## Verdict

**fuse_r1 reproduces exactly.** Rebuilt from an empty cache, every result file is byte-identical to fuse_r1's,
its METHOD files re-score to the numbers it reported, and an independent recomputation of the outcomes from
`tracks.parquet` and the annotation agrees with `score.py` on all 167 annotated returns.

| point | fuse_r1 reported | re-score of its METHOD file | regenerated here | by hand (no reid_lab, no gbf) |
|---|---|---|---|---|
| strict | 67 / 3 / 67, links 312 / 7 / 25, 1.611% | same | same, file identical | 67 / 3 / 67, 312 / 7 / 25, 1771 impure rows |
| balanced | 85 / 5 / 47, links 378 / 16 / 30, 2.123% | same | same, file identical | 85 / 5 / 47, 378 / 16 / 30, 2334 impure rows |
| recall | 106 / 7 / 24, links 414 / 17 / 260, 2.175% | same | same, file identical | 106 / 7 / 24, 414 / 17 / 260, 2391 impure rows |
| frozen strict setting, held-out weights | 70 / 2 / 65, 319 / 6 / 22, 1.445% | same | same, identical | 70 / 2 / 65, 319 / 6 / 22 |
| frozen balanced setting, held-out weights | 85 / 4 / 48, 361 / 12 / 30, 1.813% | same | same, identical | 85 / 4 / 48, 361 / 12 / 30 |
| frozen recall setting, held-out weights | 112 / 7 / 18, 415 / 17 / 311, 2.177% | same | same, identical | 112 / 7 / 18, 415 / 17 / 311 |
| in-sample frozen strict / balanced / recall | 69 / 2 / 66, 86 / 4 / 47, 112 / 7 / 18 | same | same, identical | not run |
| cuts only | 0 / 0 / 137, 0.999% | same | same, identical | 0 / 0 / 137, 1098 impure rows |

(right / wrong / lost of 137 scored returns; links right / wrong / unknown; impure rows.)

## What was done, in order

1. **Copy of fuse_r1's scripts** (`src_fuse_r1_original/` = untouched copies; the working copies here differ only
   in the path string `reid_lab/work/fuse_r1` -> `reid_lab/work/v_repro_1`). No cache, parameter, METHOD or
   result file of fuse_r1 was copied. The inputs fuse_r1 takes from the other agents were used as they are:
   m_cuts `loco_params.json` / `final_params.json`, m_space `params_cache/` / `params_all13.json`, and the
   modules of m_appear, m_space, m_cuts.
2. **Re-score of fuse_r1's own METHOD files** with the lab's CLI -> `rescore_fuse_r1/` (10 files, all `OUT_*.json`
   byte-identical to fuse_r1's):
   `for p in strict balanced recall fixed_strict fixed_balanced fixed_recall frozen_strict frozen_balanced frozen_recall cuts_only; do $PY -I -B C:/Research/plugins/tools/reid_lab/score.py C:/Research/plugins/tools/reid_lab/work/fuse_r1/METHOD_$p.json rescore_fuse_r1/OUT_$p.json; done`
3. **fuse_r1's README commands from nothing** (logs in `logs/`, each ends with its exit code):
   ```
   $PY -I -B prep.py 5                  # 106.6 s; 160 tables + cuts.json + cut_index.json
   $PY -I -B compare_cache.py           # (mine) all 160 tables, cut_index.json and the 182 cut lists equal fuse_r1's cache exactly
   $PY -I -B check_linker2.py           # ALL CHECKS PASS, log identical to fuse_r1's (78 comparisons)
   $PY -I -B fit_loco.py 5              # 820.6 s
   $PY -I -B check_r0.py                # r0 REPRODUCES fuse (32 rows equal, pair-model difference 0.0)
   $PY -I -B method.py                  # 10 METHOD files
   $PY -I -B check_method.py ; $PY -I -B check_labelfree.py
   for p in strict balanced recall fixed_strict fixed_balanced fixed_recall frozen_strict frozen_balanced frozen_recall cuts_only; do $PY -I C:/Research/plugins/tools/reid_lab/score.py METHOD_$p.json OUT_$p.json; done    # out/cli_scores.txt
   $PY -I -B diag_points.py ; $PY -I -B diag_baselines.py ; $PY -I -B uncertainty.py
   $PY -I -B sensitivity.py 4 heldout ; $PY -I -B sensitivity.py 4 fixed
   $PY -I -B selfcheck.py ; $PY -I -B stress.py all 4      # cache_stress/ (587 MB) deleted afterwards
   $PY -I -B method.py cand 0.5 off ; $PY -I -B method.py cand 0.55 off ; $PY -I -B method.py cand 0.2 off   # + score.py CLI -> out/cand_scores.txt
   ```
4. **Own checks** (scripts written here):
   ```
   $PY -I -B hand_check.py C:/Research/plugins/tools/reid_lab/work/fuse_r1/METHOD_balanced.json out/hand_balanced.json 2026111501 10   # out/hand_balanced.log
   $PY -I -B compare_hand.py C:/Research/plugins/tools/reid_lab/work/fuse_r1/METHOD_balanced.json out/hand_balanced.json              # out/compare_hand_balanced.log
   $PY -I -B trace_picks.py C:/Research/plugins/tools/reid_lab/work/fuse_r1/METHOD_balanced.json out/hand_balanced.picks.json         # out/trace_picks_balanced.log
   $PY -I -B trace_picks.py C:/Research/plugins/tools/reid_lab/work/fuse_r1/METHOD_balanced.json out/hand_balanced.wrong.json         # out/trace_wrong_balanced.log (the 5 wrong returns)
   # hand_check.py + compare_hand.py also for strict, recall, fixed_strict, fixed_balanced, fixed_recall, cuts_only: out/hand_<p>.log, out/compare_hand_<p>.log
   $PY -I -B check_rule_b.py            # out/check_rule_b.log: rule B written again from PREREG.txt, on this run's fold results
   $PY -I -B check_ids.py <the ten METHOD files of fuse_r1>     # out/check_ids.log: structure of the METHOD files, label-free
   $PY -I -B hand_offset.py SNGS-162 SNGS-107 SNGS-151          # out/hand_offset.log: frame alignment and scale the scorer assumes
   $PY -I -B hand_roles.py              # out/hand_roles.log: no return comes from a role change
   $PY -I -B C:/Research/plugins/tools/reid_lab/check_score.py  # out/check_score.log: baselines 0/4/133, 30/14/93, 26/6/105, problems: none
   ```
   `hand_check.py`, `trace_picks.py`, `hand_offset.py`, `hand_roles.py` read the annotation: they check the
   scorer and are not methods. They import neither `reid_lab` nor `gbf`.

## Results

**Reproduction.** Byte-identical to fuse_r1's files: `params_loco.json`, `out/loco_report.json`, `out/tradeoff.md`,
all ten `METHOD_*.json` and `OUT_*.json`, `out/cli_scores.txt`, `out/diag_points.md` / `.json`,
`out/diag_baselines.json`, `out/uncertainty.md` / `.json`, `out/sensitivity_heldout.md` / `.json`,
`out/sensitivity_fixed.md` / `.json`, `out/selfcheck.md`, `out/stress.md` and the eight `out/stress/*.json`.
The 160 evidence tables built here from nothing equal fuse_r1's cache exactly, so its shortcut (copying
fuse's tables instead of rebuilding them) changed nothing. This is the third LOCO run with identical output.
The three extra fixed-setting files score as the trade-off table says: 0.50 off 81 / 4 / 52 | 345 / 9 / 29 |
1.78%, 0.55 off 76 / 3 / 58 | 333 / 7 / 26 | 1.61%, 0.20 off 95 / 5 / 37 | 415 / 17 / 36 | 2.14%.

**Checks of fuse_r1 re-run here.** check_linker2: all pass. check_r0: reproduces fuse. check_method: method.py
reproduces the scored links, all groups consistent. check_labelfree: 993 files opened, no annotation file,
no label module, the links of two clips rebuilt from nothing equal the METHOD files.

**Scorer, by hand.** `hand_check.py` reads `Labels-GameState.json` and `refine/tracks.parquet` itself, matches
boxes per frame (IoU >= 0.5 at 1080p, Hungarian), finds the returns, applies cuts and links and the outcome
rules. For the balanced point and six more METHOD files its outcome equals `score.py`'s for every one of
the 167 annotated returns (137 scored, 30 undetected), with the same rows A and B, the same matched pairs,
the same link verdicts and the same impure rows on every clip.
The 10 random scored returns (numpy `default_rng(2026111501)` over the 137 scored returns in clip / person /
frame order), balanced point: SNGS-107 person 6 lost, SNGS-163 person 18 lost, SNGS-066 person 9 right,
SNGS-161 person 5 right, SNGS-169 person 1 right, SNGS-169 person 23 right, SNGS-162 person 2 lost,
SNGS-162 person 19 lost, SNGS-151 person 5 right, SNGS-102 person 14 right: 10 of 10 equal to `score.py`.
`trace_picks.py` recomputes the same 10 a second way (no Hungarian, no union-find, raw numbers printed):
10 of 10 again; the 5 wrong returns of the balanced point too (5 of 5).

**Rule B.** Written again from PREREG.txt's text and applied to this run's fold results, it chooses the setting
in `params_loco.json` in all 42 cases (13 folds + all 13 clips, three points), using the 12 fit clips only,
and pooling each clip at its fold's setting gives 67 / 3 / 67, 85 / 5 / 47, 106 / 7 / 24. The clip left out of J
is SNGS-162 in 12 folds and SNGS-112 / SNGS-066 / SNGS-100 in the fold that holds SNGS-162 out, as reported.

**METHOD files, structure (label-free).** Every link names two sub-pieces that have rows after the cuts, no
link is repeated or closes a cycle, no identity has two boxes in one frame, no linked piece is a referee,
every cut leaves two non-empty parts.

**Scorer assumptions.** Shifting the annotation by -2 .. +2 frames, offset 0 has the most matched pairs and
the highest IoU on three clips; scale 1.0 instead of 1.5 matches almost nothing. None of the 281 annotated
players / goalkeepers changes role inside a clip, so no "return" is a role change.

## Things a reader should know (none changes a reported number)

- **Exit code 127 under load.** While the three verifiers ran together (CPU 100%, 2.7 GB of 41 GB commit left,
  28 python processes) a python command returned exit code 127 with no output more than 20 times here
  (`logs/exit127.txt` lists 16; six more in the first hand-check loop and the first run of the diagnosis
  chain). Every one ran normally when repeated and gave identical output. fuse_r1's README mentions this;
  its scoring loop as written does not check the exit code, so a missing OUT file can go unnoticed.
- **What "right" contains.** Strict: 57 of the 67 right returns have rows A and B in one identity; in the
  other 10 row B joined an earlier stretch of the same person while the piece of row A stayed outside.
  Balanced 73 + 12, recall 92 + 14 (`out/right_kinds.log`). That is the lab's scoring rule, not a fault of
  fuse_r1, but "right" is not always "the two sides of the gap were joined".
- **One tie.** At the recall point one right return (SNGS-162 person 22, frames 447 > 616) rests on a tie:
  the identity of row B held 1 earlier box of this person and 1 of person 19, and the earlier one wins.
  With the other order it would be lost (105 / 7 / 25). No tie at the other points.
- **Rounding.** The same number is printed as 1.44% (out/tradeoff.md) and 1.45% (README headline row) for
  0.01445, and as 2.17% and 2.18% for 0.02175.
- **fuse_r1's `logs/prep.log`** is from a run with every table and the cut lists already in the cache (10 s,
  "0.0 s for the cuts"); from nothing it takes 107 s here and gives the same tables and cut lists.
- Not checked: that the Chinese HTML report was sent; the fits of m_cuts and m_space (inputs here);
  whether PRTReid saw these clips in training.

## Files

| file | labels? | what |
|---|---|---|
| `src_fuse_r1_original/` | | fuse_r1's scripts and PREREG.txt as copied |
| `prep.py` ... `uncertainty.py`, `check_*.py` of fuse_r1 | as in fuse_r1 | working copies (path string changed) |
| `compare_cache.py` | no | rebuilt tables against fuse_r1's cache |
| `hand_check.py`, `compare_hand.py`, `trace_picks.py` | yes (scorer check) | outcomes by hand, comparison with score.py, raw trace |
| `hand_offset.py`, `hand_roles.py` | yes (scorer check) | frame alignment / scale, role changes |
| `check_rule_b.py` | no (scored counts) | rule B from PREREG.txt's text |
| `check_ids.py` | no | structure of METHOD files |
| `rescore_fuse_r1/` | | CLI output for fuse_r1's METHOD files |
| `METHOD_*.json`, `OUT_*.json`, `params_loco.json`, `out/`, `logs/`, `cache/` | | regenerated here |
| `snapshot_*.tsv`, `pycache_dirs_*.txt`, `out/snapshot_diff.txt` | | proof that nothing else was changed |
