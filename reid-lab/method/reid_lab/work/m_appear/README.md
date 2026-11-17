# m_appear: appearance-based re-linking of gbf pieces

Everything here works on the 13 SN-GSR dev clips only, through `reid_lab/data.py` (label-free) and
`reid_lab/score.py` (scoring). Nothing outside this directory was changed. No image or video was opened.

Python: `C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I` (called `PY` below),
run from this directory.

## The method (label-free code: `core.py`, `features.py`, `agglo.py`, `method.py`)

1. **Prototype** of a piece: mean of the unit PRTReid embeddings of its clean rows (as
   `analysis/appearance_match.py`); with fewer than 5 clean rows, of all its rows.
2. **Team centring + whitening** (`cos_w`): subtract the mean prototype row of the piece's team (label, or
   the nearer team mean when the label is missing), then whiten with the clip's within-piece covariance
   (shrinkage 0.3). Same-kit players share most of the raw vector; what is left tells them apart.
3. **Candidates**: every non-referee piece with at least one embedded row. Two groups may merge when they
   are of the same appearance team and share no frame. No gap limit, no verified-team requirement.
4. **Agglomeration** (`agglo.agglomerate`): repeatedly merge the most similar allowed pair that is
   mutual best, has cosine >= `t_abs`, and beats on both sides the best *rival* by `margin`. A rival of
   H is an allowed group that is seen at the same time as H (so certainly another person). Prototypes
   are recomputed from the pooled rows after every merge; merged groups never overlap in time.

Tuned inside leave-one-clip-out: `kind` (w or tc), `t_abs`, `margin` (3 parameters; grid 2 x 10 x 9).
Fixed by design after looking at all 13 clips: mutual-best requirement, centroid linkage, rival = shares
at least 1 frame. Never tuned: 5 clean rows, WCCN shrinkage 0.3 (`exp7_sensitivity.out`: 68-78 right,
6-7 wrong for shrinkage 0.03-3 and 1-20 clean rows).

## Results (137 scored returns; held-out = leave one clip out, pooled)

Full table: `out/tradeoff.md` (`PY tradeoff.py`). lam = price of one wrong identity (extra wrong return
or wrong link) in recovered returns, the only thing that differs between operating points.

| method | right | wrong | lost | links right / wrong / unknown | impure rows |
|---|---|---|---|---|---|
| gbf now (split) | 0 | 4 | 133 | 0 / 0 / 0 | 2.61 % |
| plugin joins | 30 | 14 | 93 | 33 / 10 / 50 | 3.08 % |
| mutual baseline | 26 | 6 | 105 | 60 / 7 / 2 | 3.03 % |
| **held-out, lam=2 (balanced)** `out/agglo-lam2.method.json` | 74 | 7 | 56 | 239 / 8 / 197 | 3.10 % |
| **held-out, lam=3 (cautious)** `out/agglo-lam3.method.json` | 67 | 6 | 64 | 226 / 3 / 187 | 2.86 % |
| held-out, lam=5 `out/agglo-lam5.method.json` | 67 | 6 | 64 | 214 / 3 / 180 | 2.86 % |
| held-out, lam=10 (strict) `out/agglo-lam10.method.json` | 60 | 6 | 71 | 201 / 3 / 165 | 2.83 % |
| held-out, lam=1 (not recommended) `out/agglo-lam1.method.json` | 87 | 16 | 34 | 281 / 16 / 248 | 3.75 % |
| held-out, margin only (kind w, t_abs 0.5 fixed in-sample), lam=2 | 73 | 6 | 58 | 236 / 2 / 192 | 2.80 % |
| held-out, margin only, lam=3 | 71 | 6 | 60 | 232 / 2 / 189 | 2.80 % |
| held-out, pairwise mutual best + rival margin, lam=2 | 58 | 9 | 70 | 205 / 11 / 117 | 3.08 % |
| in-sample, frozen balanced (w, 0.50, 0.25) `out/frozen-balanced.method.json` | 75 | 6 | 56 | 238 / 2 / 198 | 2.80 % |
| in-sample, frozen cautious (w, 0.50, 0.30) `out/frozen-cautious.method.json` | 71 | 6 | 60 | 220 / 1 / 174 | 2.76 % |

Of the 6 wrong at the cautious point, 4 are gbf-now's own (a plugin track that moved onto another
person), 1 is a correct link of person 17's pieces in SNGS-107 whose later piece first sat on person 18,
and 1 is the SNGS-162 pair listed below. Its 3 wrong links: that pair, SNGS-100 27 -> 63 (17 clean rows)
and SNGS-112 49 -> 100049 (3 clean rows). Cluster bootstrap over clips (`PY uncertainty.py ...`): right
share 0.49 (90 % interval 0.37-0.62), extra wrong share 0.015 (0-0.028) at lam=3.

Read with care:
- **The rival rule rests on one clip.** With "rival shares >= 5 frames" also offered to the selection
  (`out/loco_agglo_nested_covis.txt`), the fold that holds out SNGS-162 picks it and makes 7 more wrong
  returns there: held-out 70 / 13 / 54 at lam=3. On the other 12 clips the two rules hardly differ.
- **Below margin 0.2 precision does not transfer**: lam=1 picks margin 0.15 in-sample (89 / 7) and gives
  87 / 16 held-out.
- "unknown" links join pieces with fewer than 5 matched rows. With the majority person of however few
  matched rows (`diag_unknown_links.py`), lam=3: 140 agree, 5 disagree, 42 have a piece with no matched row.
- SNGS-162 (18 scored returns) gets 1 right at lam=3: nearly everybody leaves and returns there.

## Suspected annotation swaps (needs a human look; `PY diag_swaps.py`, `diag_swaps.out`)

SNGS-162, annotated people 12 and 14 (and, less completely, 9 and 18): the pieces before and after the
gap match crosswise. 4 -> 96: raw cosine 0.992, team-centred 0.931, with 425 and 156 clean rows, while the
annotated continuation 18 -> 96 has 0.124; 18 -> 106: 0.727 against 0.159. The pitch positions agree with
the crossed reading (|dy| 1.7 + 3.5 m against 7.8 + 9.5 m). Every appearance method that links these
gets one wrong return and one wrong link from this pair. Outside SNGS-162 the 192 most similar
best-earlier-candidate decisions (team-centred cosine) are all right (`explore7.out`).

## Commands

```
PY features.py                 # pair_features on all clips (smoke test, 30 s with tiers)
PY grid.py agglo               # 180 settings x 13 clips, label-free links + score.score_clip -> cache/agglo.json (about 4 min, 6 processes)
PY grid.py agglo_covis5 mutual # the two comparison families -> cache/*.json
PY loco.py agglo               # leave-one-clip-out selection -> out/agglo-lam{1,2,3,5,10}.method.json / .score.json
PY loco.py agglo --fix kind=w t_abs=0.5
PY loco.py mutual
PY loco.py agglo+agglo_covis5
PY method.py balanced out/frozen-balanced.method.json     # frozen parameters (in-sample on the dev clips)
PY C:/Research/plugins/tools/reid_lab/score.py out/agglo-lam3.method.json out/agglo-lam3.cli-score.json
PY tradeoff.py                 # out/tradeoff.md
PY uncertainty.py out/agglo-lam3.method.json
PY check_cache.py              # the code reproduces the cached links; cuts are accepted
```

For fusion: `from features import pair_features` (docstring lists the columns; `cuts=` and `tiers=True`
supported), `from method import links` (`links(seq, "balanced" | "cautious" | "strict", cuts=...)`).

## What was measured on the way

Diagnosis scripts use labels through `fitlabels.py` and never produce a method's links.

- `explore3.py` ... `explore7.py` (`explore7.out`): prototypes and similarities. Team centring is the
  gain (top-1 AP 0.927 -> 0.968; without SNGS-162 0.969 -> 0.988). No gain from: all rows, rows away from
  the edge or from the piece's ends, overlap <= 0.3, taller boxes, height or confidence weights, median,
  trimmed mean, 2-3 prototypes per piece, k-reciprocal re-ranking, CSLS, PCA, LDA.
- `exp3_calib.py`: a logistic score fitted inside LOCO on cosines, clean rows, gap, height ratio, impostor
  z-scores, same-plugin flag does not beat the plain team-centred cosine.
- `exp1_mutual.py`, `exp2_agglo.py`, `exp2b_gate.py`, `exp2c_covis.py`, `exp6_candidates.py`
  (`exp6_candidates.out`), in-sample: rival margin instead of plain runner-up margin; agglomeration against
  pairwise; verified team only 53 / 8; team label only 69 / 6; >= 5 clean rows only 67 / 6; need a rival
  59 / 6; no mutual best 77 / 9; rival >= 5 frames 89 / 13; raw cosine 69 / 9 or 41 / 5.
- `exp4_adjacent.py`: joining a piece to one starting right after at the same place (not appearance): 1-73
  links, 33-80 % right where judged; on top of the cautious point at best 2 more returns for 6 more wrong links.
- `exp5_plugin.py`: plugin joins with whitened cosine >= 0.3 added to the balanced point, in-sample:
  83 / 7, wrong links 2 -> 5. Left to fusion (`same_plugin` is a column).
- `oracle_bounds.py`: linking all same-person pieces gives 105 right, 7 wrong; inside
  appearance_match.py's candidate pairs only 51; with a 10 s gap limit 80.
- `diag_returns.py`, `diag_lost.py`, `diag_wrong.py`, `diag_wrong_agglo.py`, `diag_pair.py`,
  `diag_pair2.py`, `diag_covis.py`, `diag_covis_162.py`, `diag_162.py`, `diag_jersey.py`, `diag_simmat.py`,
  `explain_method.py`, `show_grid.py`: per-return and per-link views behind the statements above.
  `explore0.py` - `explore2.py`: first look at the data. `diag.py`, `diag2.py`: shared helpers.
