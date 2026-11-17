# v_repro_0: reproduction check of work/fuse (verification round 0)

Everything here was written by v_repro. Nothing outside this directory was changed
(`snapshot_before.tsv` / `snapshot_after.tsv`: size and mtime of every file of fuse, m_appear, m_cuts,
m_space, m_keep, diag_cues, diag_failures and of reid_lab/*.py, identical before and after; no new
`__pycache__` file anywhere). Only the 13 dev clips were read. No image or video was opened.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_repro_0
```

## What was done, in order

1. Copy of fuse's scripts (`src_fuse_original/` = untouched copies; the working copies in this
   directory differ only in the path string `reid_lab/work/fuse` -> `reid_lab/work/v_repro_0` and in
   line ends). No cache, parameter or METHOD file of fuse was copied: everything below was rebuilt.
   m_cuts' `loco_params.json` / `final_params.json`, m_space's `params_cache/` / `params_all13.json`
   and the modules of m_appear, m_space, m_cuts are fuse's inputs and were used as they are.
   Every command was run with `-I -B` (`-B`: write no bytecode; it does not change any result).
2. Re-score of fuse's own METHOD files with the lab's CLI -> `rescore_fuse/`:
   `for p in strict balanced recall frozen_strict frozen_balanced frozen_recall margin_0.50; do $PY -I -B C:/Research/plugins/tools/reid_lab/score.py C:/Research/plugins/tools/reid_lab/work/fuse/METHOD_$p.json rescore_fuse/OUT_$p.json; done`
3. fuse's README commands, from nothing:
   ```
   $PY -I -B prep.py 6                                   # out/prep.log (161 s)
   $PY -I -B check_linker.py ; $PY -I -B check_space.py native ; $PY -I -B check_space.py compare    # out/checks_1.log
   $PY -I -B fit_loco.py 6                               # out/fit_loco.log (1083 s; the machine was shared with other agents)
   $PY -I -B method.py ; $PY -I -B method.py margin 0.5 ; $PY -I -B method.py margin 0.8 ; $PY -I -B method.py margin 0.1    # out/method.log
   $PY -I -B check_method.py ; $PY -I -B check_groups.py # out/checks_2.log
   for p in strict balanced recall frozen_strict frozen_balanced frozen_recall margin_0.50 margin_0.80 margin_0.10; do $PY -I -B C:/Research/plugins/tools/reid_lab/score.py METHOD_$p.json OUT_$p.json; done    # out/cli_scores.txt
   $PY -I -B sensitivity.py 6                            # out/sensitivity.md, out/sensitivity.log (492 s)
   $PY -I -B d7_after.py > out/d7_after.txt ; $PY -I -B t7_impure.py > out/impure_per_clip.txt ; $PY -I -B t8_wronglinks.py > out/wrong_links.txt
   ```
   Exploration scripts behind numbers quoted in fuse's report (diagnosis, labels):
   ```
   $PY -I -B d1_look.py > out/d1_look.txt ; $PY -I -B d2_pairmodel.py > out/d2_pairmodel.txt
   $PY -I -B d3_fragments.py > out/d3_fragments.txt ; $PY -I -B d4_table2d.py > out/d4_table2d.txt
   $PY -I -B t3_onefold.py SNGS-107 > out/t3_onefold_107.txt
   $PY -I -B d5_merges.py SNGS-107 pos 0.5 0.30 > out/d5_merges_pos.txt ; $PY -I -B d5_merges.py SNGS-107 app 0.5 0.30 > out/d5_merges_app.txt
   ```
4. Own checks (scripts written here):
   ```
   $PY -I -B audit_method.py        # out/audit_method.log: audit hook on every file opened by the apply path
   $PY -I -B hand_check.py C:/Research/plugins/tools/reid_lab/work/fuse/METHOD_balanced.json out/hand_balanced.json     # out/hand_balanced.log
   $PY -I -B compare_hand.py C:/Research/plugins/tools/reid_lab/work/fuse/METHOD_balanced.json out/hand_balanced.json  # out/compare_hand_balanced.log
   $PY -I -B hand_links.py C:/Research/plugins/tools/reid_lab/work/fuse/METHOD_balanced.json OUT_balanced.json          # out/hand_links_balanced.log
   # the same two for strict and recall: out/hand_strict.*, out/hand_recall.*, out/compare_hand_strict.log, out/compare_hand_recall.log
   $PY -I -B hand_offset_check.py SNGS-162 SNGS-107       # out/hand_offset_check.log
   $PY -I -B C:/Research/plugins/tools/reid_lab/check_score.py   # out/check_score.log (baselines)
   ```
   `hand_check.py` imports neither reid_lab nor gbf: it reads Labels-GameState.json and
   refine/tracks.parquet itself, matches boxes per frame (IoU >= 0.5, Hungarian), finds the returns and
   applies the outcome rules. It reads labels: it is a check of the scorer, not a method.
   The 10 random scored returns: numpy default_rng(20261115) on the pooled list of scored returns.

## Result

- Byte-identical to fuse's files: `params_loco.json`, `out/loco_report.json`, `out/tradeoff.md`,
  `out/sensitivity.md`, `METHOD_{strict,balanced,recall,frozen_strict,frozen_balanced,frozen_recall,margin_0.50}.json`
  and the seven `OUT_*.json` (both the re-score of fuse's METHOD files and the score of the regenerated
  ones). `cache/cut_index.json` identical; all 135 evidence tables equal fuse's cache exactly; the cut
  lists equal for every key. `out/d7_after.txt`, `out/impure_per_clip.txt`, `out/wrong_links.txt`
  identical to fuse's up to line ends.
- Held-out: strict 33 / 4 / 100 (links 152 / 4 / 158, impure 1.284%), balanced 93 / 7 / 37 (290 / 10 / 376,
  1.887%), recall 110 / 7 / 20 (314 / 10 / 437, 1.909%); in-sample frozen 37 / 0 / 100, 109 / 6 / 22,
  112 / 7 / 18; fixed margin 0.5: 91 / 4 / 42, 0.8: 51 / 1 / 85, 0.1: 116 / 9 / 12. All as reported.
- Scorer check: hand outcome = score.py outcome for all 167 annotated returns (137 scored, 30 undetected)
  at the balanced, the strict and the recall point, including the 10 random ones; link verdicts and
  impure rows of the balanced point recomputed independently: equal on every clip.
- Apply path label-free at run time: no annotation file, no reid_lab/score.py, no gbf.evaluate opened;
  the METHOD dicts rebuilt under the audit hook equal the files.
- Two score.py CLI calls inside shell loops returned without output on the first attempt (once no exit
  code captured, once exit 127 = the interpreter was not started) while other agents loaded the machine;
  both gave the expected output when repeated. This is the shell, not the scorer.
- Small wording slips in fuse's report (no number of a result is affected): see the verdict returned to
  the orchestrator.
