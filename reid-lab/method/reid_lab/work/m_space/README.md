# m_space: position, motion and camera-view evidence for re-linking pieces

Everything here works on the 13 dev clips only, reads the tracking outputs through
`reid_lab.data`, and writes only into this directory. Python: gbf's venv, always with `-I`.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/m_space
```

## What is delivered

| file | what | labels? |
|---|---|---|
| `geom.py` | cameras per frame, projection, signed distance of a standing person from the image border, start / end state of every piece (robust line fit over its first / last second) | no |
| `features.py` | `pair_features(seq) -> DataFrame` (columns `a`, `b`, ...; documented in the module docstring): one row per ordered pair of pieces with `a.last < b.first` | no |
| `linker.py` | the position (+ team / role) only linker: conditional-logit pair model with competition between candidates; `posteriors`, `links`, `peel_sequence`, `cut_sequence` apply it (label-free); `fit`, `loco_params`, `heldout_nll` fit / evaluate it | `fit` and `heldout_nll` only (through `fitdata.py`) |
| `posscore.py` | `position_scores(seq, exclude=(seq,))`: the fitted model's scores per pair for the fusion step (`pos_z`, `pos_p_in`, `pos_p_out`, `pos_sink`, ranks); weights fitted on the dev clips not in `exclude`, cached in `params_cache/` | fit clips only |
| `labels.py`, `fitdata.py` | piece identities and labelled pairs (through `reid_lab.score`); imported only by fitting and by `oracle_*.py` | yes |
| `run_position_linker.py` | the whole LOCO run: fits, trade-off table, METHOD / OUT files | fit + scoring |
| `METHOD_pos_strict.json`, `METHOD_pos_balanced.json` | final links, (emit, threshold) chosen per fold on the 12 fit clips | |
| `METHOD_pos_e<E>_t<NN>.json` | links at fixed settings (emit E in {1, 2}, threshold 0.60 / 0.50 / 0.40), LOCO weights | |
| `METHOD_extra_pos_strict_plus_plugin_rank2.json` | NOT position-only (see "Extra") | |
| `params_loco.json`, `params_all13.json` | fitted weights per held-out clip; weights fitted on all 13 (for clips outside the dev set) | |
| `tradeoff.md`, `tradeoff.json`, `tradeoff_emit.json` | pooled LOCO held-out counts over thresholds | |

## Commands (in this order)

```
$PY -I check_geometry.py          # label-free sanity checks: reprojection, projection vs gbf.geometry, the hidden-path programme
$PY -I features.py                # builds the pair features of the 13 clips (about 2 s per clip), prints counts
$PY -I run_position_linker.py     # LOCO fit + trade-off + METHOD_*.json + OUT_*.json   (about 2 minutes)
$PY -I posscore.py                # out-of-fold position scores per pair (fills params_cache/)
$PY -I tradeoff_emit.py           # which kinds of links to give out (emit 2 / 1 / 0)
$PY -I extra_plugin_veto.py       # EXTRA: position as a veto on the plugin's own re-joins
$PY -I sensitivity.py             # hand-set constants, one at a time (about 2 minutes per row, 14 rows)
$PY -I explore_models.py [variant ...]   # model variants: LOCO held-out log-likelihood and return counts
$PY -I C:/Research/plugins/tools/reid_lab/score.py METHOD_pos_strict.json OUT.json    # score any METHOD file
```

Diagnosis / oracle scripts (they read labels; their numbers are never method results):

```
$PY -I diag_path_g.py             # label-free: why gbf's path G made 0 stitches (refine.json stitch_audit counts)
$PY -I oracle_returns.py          # the 137 scored returns and the pieces around each gap
$PY -I oracle_pairs.py            # distance / speed of true partners (writes oracle_pairs.pkl, needed by the next two)
$PY -I oracle_aniso.py ; $PY -I oracle_flow.py ; $PY -I oracle_joint.py     # collective flow: does it shrink the residual?
$PY -I oracle_features.py ; $PY -I oracle_sigma.py                           # how the features separate true from false pairs
$PY -I oracle_fragments.py        # short-gap in-view fragments: position rules are only about 90 % precise
$PY -I oracle_census.py ; $PY -I oracle_noright.py                           # first appearances vs returns
$PY -I oracle_links.py 0.4 sinkhorn 1 ; $PY -I oracle_wrong.py adv_ew3 0.4   # which links go wrong
$PY -I oracle_returns_rank.py     # rank of the right earlier piece among the candidates of each returning piece
```

`explore0.py`, `explore1.py` are the first looks at the data.

## The features in one paragraph

For an earlier piece a and a later piece b: (1) the distance between where a ended and where b
started, after subtracting how far the visible players moved together in between (collective
flow; global median and a local kernel version that carries the point with the players around
it); (2) how long the place where a would be, had he moved with everyone else, was inside the
picture during the gap (and the same backwards for b): he would have been seen; (3) the least
time a speed-bounded path from a's end to b's start must spend inside the picture (min-plus
dynamic programme over a 1 m grid, 0.4 s steps), and whether b is within reach at all;
(4) border flags and sides, team / role agreement, a team census ("can a player of this team
still be unseen?"). Hand-set constants are listed at the top of `features.py` and `geom.py`;
none was tuned (`sensitivity.py`).

## The linker in one paragraph

A pair is admissible when b is within reach of a (8 m/s, 3 m slack), roles agree and teams are
not known to differ. Evidence z = w . x over (normalised residual, typical residual at this gap,
team unknown, border pattern, flow-carried in-picture seconds forwards and backwards). Each
later piece distributes one unit over its earlier candidates and "nobody came before" (its
weight depends on: starts at a border, time since clip start), each earlier piece likewise over
its later candidates and "nobody comes after"; the doubly normalised (Sinkhorn) assignment is the
score. The best pair is accepted while its score >= theta, its two slots leave the competition
and the rest is re-scored. Only links with at least one piece at an image border (emit 1) or
both (emit 2) are given out; in-view pairs compete but are never linked (position alone gets
about 1 in 10 of them wrong). Fitted per fold by maximum likelihood: 16 numbers (3 of sigma(dt),
7 pair weights, 3 + 3 context weights); the operating point (emit, theta) is chosen per fold on
the fit clips by a fixed rule.

## Headline numbers (pooled LOCO held-out, 137 scored returns; baseline split = 0 / 4 / 133)

| method | right | wrong | lost | links right / wrong / unknown | impure rows |
|---|---|---|---|---|---|
| split (gbf now) | 0 | 4 | 133 | 0 / 0 / 0 | 2.61 % |
| `pos_strict` | 41 | 4 | 92 | 57 / 1 / 46 | 2.63 % |
| `pos_balanced` | 55 | 10 | 72 | 66 / 11 / 72 | 3.44 % |
| `pos_e2_t50` (fixed) | 36 | 4 | 97 | 47 / 1 / 30 | 2.62 % |
| `pos_e1_t50` (fixed) | 41 | 4 | 92 | 57 / 1 / 46 | 2.63 % |
| `pos_e1_t40` (fixed) | 60 | 8 | 69 | 71 / 8 / 76 | 3.08 % |

In-sample (weights and operating point fitted on all 13 clips, secondary): strict 43 / 4 / 90,
balanced 56 / 6 / 75. The full threshold table is `tradeoff.md`.

## Extra (not position-only)

`extra_plugin_veto.py`: the plugin's own re-joins kept only where the position model ranks the
pair among the two best candidates on both sides, added to `pos_strict`: 49 / 4 / 84, links
63 / 2 / 59, impure 2.64 %. The rule was picked among seven after seeing their pooled numbers.
