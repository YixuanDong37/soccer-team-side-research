# m_cuts: cutting pieces where the plugin's track moved onto another person

Agent `m_cuts` of the re-identification lab. Everything here lives in
`C:/Research/plugins/tools/reid_lab/work/m_cuts/`; nothing outside it was modified.
All numbers are on the 13 SN-GSR dev clips only.

## Result in one paragraph

A label-free detector finds places inside a gbf piece where the tracked person changes and
cuts the piece there. Headline, leave-one-clip-out (LOCO) held-out, cut where P(switch) >= 0.7:
221 cuts; of the 62 whose truth is firm, 59 sit on a real identity change and 3 do not (cut
precision 0.952, clip-bootstrap 95% 0.92-1.00); impure rows fall from 2.61% to 1.00% (lower in
every one of the 13 clips; a perfect cutter would reach 0.53%); the 4 "wrong" returns of gbf's
split all become "lost" (0 / 0 / 137). On top of the mutual baseline: wrong 6 -> 2, right stays
26, impure rows 3.03% -> 1.41%; when the mutual rule is re-run on the cut pieces: right 30,
wrong 2, lost 105, impure rows 1.58%. Cuts alone never make a return "right": they turn wrong
into lost and make the pieces pure enough for linking.

## Trade-off table (pooled over the 13 held-out clips; 137 scored returns)

Baselines from `reid_lab/score.py`, no cuts:

| method | right | wrong | lost | links right/wrong/unknown | impure rows |
|---|---|---|---|---|---|
| split (gbf now) | 0 | 4 | 133 | 0/0/0 | 2.61% |
| mutual | 26 | 6 | 105 | 60/7/2 | 3.03% |
| plugin | 30 | 14 | 93 | 33/10/50 | 3.08% |

LOCO held-out (the weights used on a clip were fitted on the other 12):

| cut where P >= | method | right | wrong | lost | links R/W/U | impure rows | cuts | T | F | t | f | U | cut precision T/(T+F) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | split + cuts | 0 | 1 | 136 | 0/0/0 | 1.32% | 179 | 47 | 1 | 77 | 5 | 49 | 0.979 |
| 0.9 | mutual + cuts | 26 | 3 | 108 | 61/6/2 | 1.73% | | | | | | | |
| 0.9 | mutual re-run + cuts | 30 | 3 | 104 | 78/8/2 | 2.00% | | | | | | | |
| **0.7** | **split + cuts** | **0** | **0** | **137** | 0/0/0 | **1.00%** | 221 | 59 | 3 | 96 | 9 | 54 | 0.952 |
| **0.7** | **mutual + cuts** | **26** | **2** | **109** | 61/6/2 | **1.41%** | | | | | | | |
| 0.7 | mutual re-run + cuts | 30 | 2 | 105 | 82/7/2 | 1.58% | | | | | | | |
| 0.7 | plugin + cuts | 30 | 9 | 98 | 33/9/51 | 1.45% | | | | | | | |
| 0.7 | cuts + cut links | 6 | 0 | 131 | 91/7/57 | 1.12% | | | | | | | |
| 0.7 | mutual re-run + cuts + cut links | 35 | 2 | 100 | 157/14/59 | 1.80% | | | | | | | |
| 0.5 | split + cuts | 0 | 0 | 137 | 0/0/0 | 0.80% | 255 | 63 | 6 | 105 | 20 | 61 | 0.913 |
| 0.5 | mutual + cuts | 26 | 2 | 109 | 61/6/2 | 1.21% | | | | | | | |
| 0.5 | mutual re-run + cuts | 30 | 2 | 105 | 84/6/2 | 1.23% | | | | | | | |
| 0.5 | mutual re-run + cuts + cut links | 35 | 2 | 100 | 162/13/67 | 1.46% | | | | | | | |
| 0.3 | split + cuts | 0 | 0 | 137 | 0/0/0 | 0.74% | 300 | 67 | 14 | 109 | 41 | 69 | 0.827 |

In-sample (weights fitted on all 13 clips; secondary) is almost identical, e.g. P >= 0.7:
219 cuts, T 59 / F 2, split + cuts 0/0/137 at 1.00%, mutual + cuts 26/2/109 at 1.41%, mutual
re-run 30/2/105 at 1.58%; cut links in-sample 76 right / 3 wrong against 91 / 7 held-out (the two
link thresholds are less stable than the cut weights). Full tables: `results.md`,
`results_cutlinks.md`, `results.json`.

How to read the cut columns (`truth.py`): T = the cut separates a true switch (at most 5 rows
left on the wrong side) or the majority annotated person of the 10 matched rows before differs
from the 10 after; F = same person on both sides (at least 3 matched rows each); t / f = the same
judged from only 1-2 matched rows on a side (weak; 121 of the 221 cuts split off a single row);
U = a side has no matched row at all. Cut precision uses T and F only; with the weak ones it is
(59+96)/(59+3+96+9) = 0.928 at P >= 0.7.
"mutual + cuts" = the baseline's 69 links moved onto the sub-pieces
(`features.remap_links(..., "appearance")`); "mutual re-run" = the same rule
(appearance_match.py's candidate pairs, mutual best, 0.05 margin) recomputed on the pieces as they
are after the cuts (`relink.py`, which reproduces appearance-match.json's candidate pairs exactly
on all 13 clips when there are no cuts); "plugin + cuts" = the plugin's re-joins attached to the
stretches next to each re-entry gap; "cut links" = see below.

## What the errors are (diagnosis, labels; `logs/`)

- 78 true in-piece switches in the 13 clips (runs of >= 5 matched rows of another annotated
  person). 67 of them happen across missing frames inside the picture: 47 across a gap of >= 10
  frames that split_tracks leaves alone because no box touches the image edge. The plugin
  re-activates a lost track on whoever is there. Only 11 are swaps with no missing frame.
- Besides these there are many stray tails and heads: 1-4 boxes after or before a gap, often a
  single box many seconds later. Of the 221 held-out cuts at P >= 0.7, 171 split off fewer than 5
  rows (121 a single box): 107 of those small parts are another person by the annotation (95
  judged from 1-2 matched rows, 12 firm), 10 the same person, 54 match no annotated person. They
  hold few rows but they are what makes returns "wrong" (SNGS-162 piece 53: one piece, two stray
  boxes, two wrong returns) and they move a piece's last frame and last position, which every
  linker reads.
- gap length alone is a poor cue: 599 internal gaps of >= 10 frames, about 9% are switches.
- 27 of the 78 switches are between players of different teams according to the annotation. In
  14 the embeddings agree that the kit changes (the team score flips sign; `diag_crossteam.py`); in
  13 it keeps its sign, and 12 of those have a before/after cosine of 0.84-0.99, i.e. the same kit
  on both sides (the 13th, cosine 0.76, is cut). None of the 47 same-team switches flips the team
  score, so the score is trustworthy. Those 12 are most likely the box-to-annotation matching
  flipping between two overlapping people (or an annotation slip), not the track changing person.
  The detector "misses" all 12, so 45/78 hit understates recall, and part of the 0.53% floor is
  measurement, not tracking. This is an inference from numbers; no frame was viewed.

## Method (label-free part: `features.py`)

Candidate boundaries of a piece: every gap of >= 2 missing frames ("gap", any side size, also a
single box) and, inside gap-free stretches, local minima below 0.90 of the before/after appearance
cosine ("app"). Per candidate:

- `cos`: cosine between the mean PRTReid embedding of the 20 clean rows before and the 20 after
  (clean = `data.clean_rows`; a side with fewer than 3 clean rows uses all its rows);
- `mm`: `cos` minus the best cosine to an alternative, taking the weaker of two alternatives:
  the best other piece that vanished in the 10 s before as the origin of the after side, and the
  best other piece that appears in the 10 s after as the continuation of the before side. A switch
  has both: the person taken over had a piece, and the person left behind gets a new one;
- `lgap` = log(1 + gap frames), `ld` = log(1 + metres on the pitch across the boundary).

A logistic regression on these four gives P(the two sides are different people). Per piece, greedy
binary segmentation: cut the most probable candidate if P >= threshold, recompute the windows of
the remaining candidates inside the two parts (a window never reaches across a cut), repeat.
Candidates that share both windows are one decision, placed at the largest gap; no second cut
within 10 rows of a cut.

Numbers fitted with labels: 5 (4 weights + bias), per LOCO fold on the firm T / F candidates of
the 12 fit clips. The threshold is not fitted: P >= 0.9 / 0.7 / 0.5 / 0.3 are operating points.
Frozen all-13 weights (standardised inputs): cos -1.36, mm -0.67, lgap +0.32, ld +0.30, bias
-3.34 (`final_params.json`). The feature set was chosen among 9 by held-out log-loss
(`logs/loco_models.txt`: cos only 0.168, cos+mm 0.159, cos+mm+lgap 0.153, chosen 0.152; adding
window size or splitting mm in two does not help).
Fixed settings (window 20 rows, gap >= 2, 10 s, 10 rows, ...) were set by judgement while looking
at these clips, not searched; `sensitivity.py` re-runs the whole LOCO with four of them moved
down and up (window 10 / 40 rows, gap >= 1 / 3, 5 s / 20 s, 5 / 20 rows): cut precision
0.946-0.967, impure rows 0.99-1.16% (`logs/sensitivity.txt`).

Cut links (`cutlinks.py`, secondary): for every cut, "best alternative origin -> after part" and
"before part -> best alternative continuation", kept when the cosine to the alternative is >= floor
and beats the cut's own cosine by >= margin, and the two parts are never seen in the same frame.
Two thresholds, chosen per fold from a 5 x 5 grid as the point with the most right links among
those >= 95% right on the fit clips (folds chose margin 0.10-0.20, floor 0.93; all-13: 0.20, 0.93).
Held-out link precision 91/(91+7) = 0.93, below the 0.95 it was fitted for.

## For the fusion step

```python
import sys; sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_cuts")
import features, relink, cutlinks
features.cuts(seq)                 # [[piece, frame], ...], frozen weights, P >= 0.7 (label-free)
features.cuts(seq, 0.5)            # another operating point
features.cuts_heldout(seq, 0.7)    # weights fitted WITHOUT this dev clip: use this inside LOCO
features.cut_table(seq, 0.7)       # the cuts with features and probability
features.subpieces(seq, cuts)      # {sub-piece id: piece, row range, first/last frame}
features.remap_links(seq, links, cuts, "appearance")   # piece links -> sub-piece links
relink.mutual_links(seq, cuts)     # the mutual rule on the pieces after the cuts
cutlinks.links_default(seq, 0.7, heldout=True)         # (cuts, cut links)
```

Sub-piece ids are score.py's: rows from the j-th cut of a piece (by frame, j from 0) get
`piece + 10**7 * (j + 1)`; rows before the first cut keep the piece id. A link made on whole
pieces must be re-pointed after cutting: "the earlier piece's last part" is wrong whenever that
part is a stray tail (links right/wrong/unknown 55/7/7 against 61/6/2 with the appearance choice).
`methods/*_cuts.csv` carry truth columns (`label`, `sw_id`): diagnosis only, never a method input.

## What failed or did not help

- Geometry of a takeover (the piece restarts exactly where another piece just ended, box IoU >=
  0.5, own box far away): 26 such events, 6 true switches, 11 same person (usually a fragment of
  the same person handing the boxes back), 9 unknown. Not usable as a rule; only the appearance
  comparison separates them.
- Impossible motion: the pitch jump exceeds gbf's bound (12 m/s plus 4 m position allowance) on 2
  of 78 switches; the image jump with camera motion removed has AUC 0.55. Only the soft `ld` term
  survived, with a small weight.
- Team flips: gbf's team is one value per piece, so there is no per-row team to flip; a change of
  kit shows up in `cos` (12 of the 14 real cross-team switches are cut).
- A threshold on the gap, or on the cosine alone, at equal precision finds far fewer switches
  (cos only, P >= 0.9: 24 switches hit against 37).
- First version: one pass with windows over the whole piece and a threshold picked for a target
  precision on the fit clips. Windows reached across the next switch and cut small clean stretches
  off (8 false cuts of 68 firm at the comparable point); the picked threshold jumped between
  folds. Recursive segmentation plus a fixed probability level fixed both (3 false of 62).
- Still missed at P >= 0.7: 33 of 78 switches, 634 rows on their smaller sides. 7 have no
  candidate (swap with no missing frame and no appearance dip), 16 have before/after cosine >=
  0.90 (same-kit look-alikes, or the matching artefacts above). The largest (SNGS-097 piece 8, 146
  rows) has P = 0.67 and is cut at P >= 0.5.
- The 2 returns still wrong on top of mutual are wrong links of the mutual rule itself; no cut can
  repair those.

## Caveats

- The weights are LOCO, but the design (candidate rules, features, recursion, the fixed settings)
  was developed on these same 13 clips with label-based diagnosis. The held-out numbers are
  therefore optimistic by an unknown amount; the 37 reserved clips are the clean check.
- 62 firm cuts: 3 false ones is a small count. 54 cuts (mostly single boxes) cannot be judged at
  all because the box matches no annotated person.
- impure_row_share can be bought by cutting everywhere; read it together with the cut precision
  and the number of cuts. A false cut loses continuity (a later link can restore it; the re-run
  mutual rule does re-join some), it never mixes two people.
- Clips differ a lot: 5 to 32 cuts per clip, impure rows before cutting 0.3% to 6.0%.
- SN-GSR at 720p with this plugin; the recall on swaps without missing frames is poor and such
  swaps may be more common on other footage.

## Files

| file | labels? | what |
|---|---|---|
| `features.py` | no | candidates, features, model application, segmentation, `cuts`, ids, `remap_links` |
| `relink.py` | no | the mutual rule on (sub-)pieces; `python -I relink.py` checks it against appearance-match.json |
| `cutlinks.py` | no | links a cut suggests |
| `final_params.json`, `loco_params.json`, `cutlink_params.json` | fitted | weights (all 13 / per held-out clip), cut-link thresholds |
| `truth.py` | yes | true switches and the T/F/t/f/U label of a candidate or cut (fit and scoring only) |
| `loco.py` | yes | LOCO comparison of 9 feature sets |
| `make_methods.py` | yes | final LOCO run, parameter files, `methods/*.json`, `results.json/.md` |
| `eval_cutlinks.py` | yes | LOCO for the cut links, `results_cutlinks.json/.md` |
| `report.py`, `sensitivity.py` | yes | per-clip table and bootstrap; fixed-setting sensitivity |
| `diag_*.py` | yes | diagnosis only (oracle ceiling, wrong returns, missed switches, switch types, cross-team check, cut-link grid) |
| `exploration/` | yes | early exploration scripts, kept for the record (see NOTE.txt; some no longer run) |
| `methods/` | | METHOD files for score.py: `{heldout,insample}_p{0.9,0.7,0.5,0.3}_{split,mutual,mutual_rerun,plugin}.json`, `..._{cutlinks,mutual_rerun_cutlinks}.json` for 0.9/0.7/0.5 |
| `logs/` | | outputs of the commands below |
| `report-m_cuts.html` | | the visual summary (Chinese) sent to the project lead |

## Commands (gbf's venv, always `-I`; run from this directory)

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
$PY -I relink.py                          > logs/relink_check.txt       # mutual rule reproduced, no labels
$PY -I loco.py                            > logs/loco_models.txt        # 9 feature sets, ~15 min
$PY -I make_methods.py                    > logs/make_methods.log       # final LOCO, params, methods/, results.*
$PY -I eval_cutlinks.py                   > logs/eval_cutlinks.log      # needs make_methods.py first
$PY -I report.py 0.7                      > logs/report_p0.7.txt
$PY -I sensitivity.py                     > logs/sensitivity.txt        # ~15 min
$PY -I C:/Research/plugins/tools/reid_lab/score.py methods/heldout_p0.7_split.json OUT.json
$PY -I C:/Research/plugins/tools/reid_lab/score.py methods/heldout_p0.7_mutual.json OUT.json
$PY -I C:/Research/plugins/tools/reid_lab/score.py methods/heldout_p0.7_mutual_rerun.json OUT.json
# diagnosis (labels, no method result)
$PY -I diag_oracle.py                     > logs/diag_oracle.txt
$PY -I diag_wrong.py                      > logs/diag_wrong.txt
$PY -I diag_final.py 0.7                  > logs/diag_final_p0.7.txt
$PY -I diag_missed.py 0.7                 > logs/diag_missed_p0.7.txt
$PY -I diag_switch_types.py 0.7           > logs/diag_switch_types_p0.7.txt
$PY -I diag_crossteam.py 0.7              > logs/diag_crossteam_p0.7.txt
$PY -I diag_cutlinks.py 0.7               > logs/diag_cutlinks_p0.7.txt
```

No bug was found in `reid_lab/score.py` or `data.py`. One thing to know when writing cuts:
score.py numbers the sub-pieces of a piece by the order of the cut frames in the list it is
given, so a duplicated [piece, frame] would shift the ids (`features.subpiece_ids` de-duplicates;
`features.cuts` never emits duplicates).
