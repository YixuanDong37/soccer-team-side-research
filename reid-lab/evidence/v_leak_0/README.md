# v_leak (round 0): leakage and protocol check of work/fuse

Question: does anything in the fused re-linker (work/fuse and the modules of m_cuts, m_appear, m_space it
imports) let the held-out clip of a leave-one-clip-out fold influence its own result, and how much of the
reported held-out result rests on choices made with all 13 clips in view?

Everything here is on the 13 dev clips only, numbers only, written only into this directory. Python: gbf's
venv with `-I -B` (`-B` and `sys.dont_write_bytecode` so that importing the other agents' modules leaves no
`.pyc` in their folders; checked afterwards: no file outside this directory is newer than the start of this
work).

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_leak_0
```

## Verdict

No label leak found. The three headline points and the whole fixed-margin table reproduce exactly from a
re-run in which each fold's process cannot read the held-out clip's annotation and reads no fitted file of
any agent. What remains are caveats about design choices made on all 13 clips, the fragility of the strict
point, and a domain question that cannot be settled on these clips.

## 1. Static check (reading the code)

Apply path = fuse `prep.py`, `evidence.py`, `linker.py`, `method.py`, `pairmodel.design/logit`; m_cuts
`features.py`; m_appear `features.py`, `core.py`; m_space `features.py`, `geom.py`, `linker.py`
(`admissible`, `posteriors`, `kind`). None of them imports `reid_lab.score`, a labels module, or opens
`Labels-GameState.json` / `reentry-eval.json` (grep in `work/`: the only direct reads of the annotation are
four diagnosis scripts of m_appear and m_cuts). Labels enter through `fuse_labels.py` (fit clips), m_cuts
`truth.py`, m_space `labels.py` / `fitdata.py` (imported lazily inside `linker.fit`). `reid_lab/data.py`
inputs (tracks, calibration, refine.json, embeddings) carry no annotation column. The dev clips are in the
pre-registered hash order of `tools/gsr/prepare_gsr.py`, i.e. not picked by their labels.

## 2. Blindfolded re-run of all 13 folds (the main test)

`blind.py` makes every door to one clip's annotation raise in the running process (open / io.open /
pandas / numpy readers of its `Labels-GameState.json` or `reentry-eval.json`, `score.clip`, `score.score_clip`,
`gsr.load_reference`; a self-test proves the doors are closed) and logs every file that is opened.

```
$PY -I -B driver.py s1_refit_cuts.py  s1_refit_cuts  5   # m_cuts' detector re-fitted per fold with m_cuts' code, held-out clip blindfolded  (3 min)
$PY -I -B driver.py s2_refit_space.py s2_refit_space 5   # m_space's position model re-fitted per fold (linker.fit, posscore.CONFIG), blindfolded      (2 min)
$PY -I -B driver.py s3_blind_fuse.py  s3_blind_fuse  5   # fuse's fold end to end from those weights: own evidence tables, pair model, margins, links  (12 min, 455 MB cache/)
$PY -I -B s4_compare.py                                  # against fuse's published files + scoring -> out/compare.md
for p in blind_strict blind_balanced blind_recall; do $PY -I -B C:/Research/plugins/tools/reid_lab/score.py out/METHOD_$p.json out/OUT_$p.json; done   # out/cli_blind.txt
```

`driver.py` starts one process per fold and retries a process that dies without output (random native
Windows crashes, exit 0xC000070A, happened twice under parallel load; `xargs -P` lost processes silently).

Result (`out/compare.md`, `out/s1_refit_cuts.log`, `out/s2_refit_space.log`), for every one of the 13 folds:

| check | result |
|---|---|
| m_cuts weights re-fitted without the clip vs `m_cuts/loco_params.json[clip]` | max abs difference 0.0; cuts of all 13 clips identical |
| m_space weights re-fitted without the clip vs `m_space/params_cache/excl_<clip>.json` | max abs difference 0.0 |
| pair model re-fitted vs `fuse/params_loco.json[clip]` | max abs difference 0.0 |
| margins chosen on the 12 fit clips (strict / balanced / recall) | equal |
| held-out links at the three points vs `METHOD_strict/balanced/recall.json` | equal |
| held-out links at all 16 margins vs fuse's `cache/loco_raw.pkl` | equal |
| annotation files opened by the fold's process | 12 (the fit clips); the held-out clip's: 0; blocked attempts: 0 |
| fitted files of any agent read by the fold's process | 0 (only source code, `reid_lab.data` inputs, this directory's stage outputs) |

Scored (score.py): blind strict 33 / 4 / 100, balanced 93 / 7 / 37, recall 110 / 7 / 20, links 152/4/158,
290/10/376, 314/10/437, impure rows 1.28% / 1.89% / 1.91%: fuse's numbers. The fixed-margin table is also
identical row by row (0.10: 116/9/12 ... 0.50: 91/4/42 ... 1.20: 2/0/135). The CLI on fuse's own METHOD
files gives the reported numbers too (`out/cli_rescore.txt`).

## 3. Design choices made with all 13 clips in view: how much optimism?

LOCO protects fitted numbers, not the design. Three choices were made on pooled held-out evidence; each was
repeated inside the folds.

```
$PY -I -B n1_nested_variants.py                          # from fuse/cache/loco_raw.pkl -> out/nested_variants.md
$PY -I -B driver.py n2_nested_cols.py n2_nested_cols 5   # blindfolded; -> out/nested_cols/<fold>.json (needs stages 1-2; rebuilds tables if cache/ is gone)
$PY -I -B n2b_score_nested_cols.py                       # -> out/nested_cols.md, out/METHOD_nestedcols_*.json
$PY -I -B n3_nested_cutmodel.py                          # -> out/nested_cutmodel.md
```

| choice | how fuse / the agent made it | nested inside each fold | effect on the held-out result |
|---|---|---|---|
| plugin-join bonus kept (fuse vs fuse_pos) | README: "kept because it adds returns at strict margins (0.8: 51 against 43)" = pooled held-out rows | variant chosen by J on the 12 fit clips: fuse in 9 / 6 / 13 folds | strict 32/4/101 (33/4/100), balanced 92/7/38 (93/7/37), recall unchanged |
| variant among all five (incl. no cuts, appearance only) | fuse declared the method | chosen by J on the fit clips | strict 36/6/95, balanced 91/8/38, recall 110/7/20 |
| pair-model inputs (cos_w, sink_lo, plug) | LOCO log-loss on the pairs of all 13 clips (d2_pairmodel.py, d4_table2d.py) | inner LOCO log-loss on the 12 fit clips: `cos_w+unreach+sink_lo+lgap` wins in all 13 folds (fuse's set ranks 2-4 of 13, gap 0.0006-0.0011) | points: strict 16/4/117, balanced 98/5/34, recall 108/9/20. Fixed margin: 0.25: 102/4/31, 0.20: 106/6/25, 0.15: 112/8/17, 0.10: 116/10/11 (fuse: 0.50: 91/4/42, 0.30: 105/6/26, 0.15: 114/7/16, 0.10: 116/9/12). A comparable curve; no measurable optimism |
| m_cuts' feature set | "chosen among 9 by held-out log-loss" (m_cuts README) | inner LOCO log-loss: another set in 11 of 13 folds | held-out cuts: 219 with the nested choice, 221 in fuse, 217 shared, 2 only nested, 4 only fuse (out/nested_cutmodel.md): negligible |

So the curve "right against wrong" is not an artefact of these choices. What does move is where a rule
lands on it:

- **Picking a fixed-margin row afterwards.** Best row with wrong <= 4 on the pooled held-out table: margin
  0.5, 91 / 4 / 42. The same criterion applied inside each fold: 53 / 6 / 78. The pre-fixed J rule: 33 / 4 / 100.
  Quoting 91 / 4 as an operating point would overstate by 38 right returns and 2 wrong ones against the
  honest version of the same pick. (cap 7: 114/7/16 picked afterwards vs 111/7/19 nested; cap 14: same row.)
- **The strict point is fragile.** Its 4 wrong returns all come from the one fold (SNGS-162) whose margin
  is 0.6; the other 12 folds use 0.9-1.0 and have 27 right, 0 wrong. The frozen method uses 0.9, whose
  held-out row is 31 / 0 / 106: the pooled 33 / 4 / 100 describes the selection procedure, not the frozen
  threshold. Under the equally defensible designs above the strict point is 16/4, 32/4, 36/6 or (cap rule)
  53/6, so "wrong <= 4" holds only just and only for this one rule.
- In-sample against held-out at the chosen margins (out/nested_variants.md, section 2): right share 0.276 vs
  0.241 (strict), 0.729 vs 0.679 (balanced), 0.817 vs 0.803 (recall); wrong share 0.000 vs 0.029, 0.036 vs
  0.051, 0.050 vs 0.051.

## 4. What the scoring rule does not show (diagnosis)

```
$PY -I -B d1_hidden_wrong.py      # -> out/hidden_wrong.md
```

score.py calls a return wrong only with >= 5 rows of another person before B, and right as soon as A and B
share a group. For fuse: no return is filed under "lost" although a link joined it to 1-4 rows of another
person (0 at all points). But some "right" returns sit in a group that also holds >= 5 matched rows of
another person: 0 (strict), 3 (balanced), 6 (recall), 2 (margin 0.5). Wrong + mixed: 4 / 10 / 13 / 6.

## 5. Things a real system could not rely on (not quantified here)

- The dev clips (and the 37 reserved ones) are SN-GSR **train** clips (tools/gsr/provenance-gsr.json), and
  the plugin's re-id model is `prtreid-soccernet-baseline.pth.tar` (tools/fetch_weights.sh). If that model
  was trained on the SoccerNet-GSR train split, as the upstream baseline's was to my knowledge, the
  appearance embeddings have seen these players, and cos_w separates people better here than on unseen
  footage. gbf's registry says "training_data = Not reviewed". I did not read the plugin's own folder (not
  in my read scope), so this is unverified. Neither LOCO nor the reserved clips can detect it; only the
  test-split holdout or an unseen clip can.
- m_space's "nobody" terms use the time since the clip started and to its end (cap 30 s), fitted on 30 s
  clips; on a long video these priors mean something else.
- Within a fold, m_cuts' and m_space's weights are in-sample for the 12 fit clips (the pair model and the
  margin are fitted on that evidence) and out-of-sample for the held-out clip. m_space's posscore docstring
  recommends excluding (fit clip, held-out clip) for this; fuse did not. This cannot leak the held-out clip;
  it can only make the fit-clip evidence look slightly better than the held-out clip's.

## 6. Small hygiene notes

- `fuse/t4_perclip.py` is described as a "fit-clip view" but prints all 13 clips, the fold's held-out clip
  included (marked `*`). Only the exploration fold SNGS-107 has a one-fold cache; whether the script was run
  cannot be told.
- `fuse_labels.py` takes the labels from `score.clip` and the private `score._ids_after_cuts` (fit code, not
  apply code).

## Files

| file | labels? | what |
|---|---|---|
| `blind.py` | no | the blindfold and the file-open audit |
| `driver.py` | no | one process per fold, with retry |
| `s1_refit_cuts.py`, `s2_refit_space.py`, `s3_blind_fuse.py` | fit clips only (blindfolded for the held-out clip) | stages 1-3 |
| `s4_compare.py` | scoring | comparison and scoring of the blind links |
| `n1_nested_variants.py`, `n2_nested_cols.py`, `n2b_score_nested_cols.py`, `n3_nested_cutmodel.py` | fit clips for the choices; scoring | nested design choices |
| `d1_hidden_wrong.py` | diagnosis | outcomes split further |
| `refit/` | fitted | per fold: re-fitted m_cuts and m_space weights, cuts, audit |
| `out/` | | logs, tables, METHOD and OUT files (`METHOD_blind_*.json` = fuse's links reproduced blind; `METHOD_nestedcols_*.json` = analysis, not a proposed method) |

`cache/` (evidence tables, 455 MB) was deleted after the run; stage 3 rebuilds it. `out/n2_nested_cols.log`
lists 12 folds; SNGS-107 was run first on its own (its result is in `out/nested_cols/SNGS-107.json`).
