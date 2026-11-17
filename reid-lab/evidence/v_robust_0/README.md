# v_robust_0: robustness check of the fused re-linker (work/fuse), verification round 0

Scope: the 13 SN-GSR dev clips only. Nothing outside this directory was written: fuse's modules are
imported read-only (no bytecode; table builders redirected to `cache/` here), `fuse/cache` and
`fuse/params_loco.json` are only read. No image or video was opened, no git, no network.
Python: gbf's venv, always `-I -B`.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_robust_0
$PY -I -B r1_reproduce.py            # re-score fuse's METHOD files, rebuild them from params_loco.json, baselines, cuts only   (2 min)
$PY -I -B r2_perclip_bootstrap.py    # per clip, concentration, clip bootstrap (2000 resamples, seed 0, paired)                 (seconds)
$PY -I -B r3_sensitivity.py 5        # +-10% on every threshold, all loose / all strict, 40 random joint draws                 (10-20 min, 5 workers, about 1.3 GB each)
$PY -I -B r3_sensitivity.py tables   # re-render out/r3_sensitivity.md from cache/r3_raw.pkl
$PY -I -B r4_link_table.py           # one row per link with the linker's evidence and the labels -> out/links_*.csv, out/returns_*.csv
$PY -I -B r5_wrong.py                # the wrong returns and wrong links one by one
$PY -I -B r6_gaps.py                 # outcomes and link precision by gap length (long gaps > 10 s)
$PY -I -B r7_team_role.py            # links across team / role labels (label-free part + annotated part)
$PY -I -B r8_impure.py               # impure rows per clip against split and against cuts only; mixed identities; completeness
$PY -I -B r9_guards.py               # simple guards: post-filters on the links and tiny pieces barred inside the linker
$PY -I -B r10_folds.py               # LOCO integrity: margins re-derived, lam sweep, parameter transfer, pair model refitted
$PY -I -B r11_suspects.py            # wrong links against the annotation's own continuation (exchange signature)
$PY -I -B r12_relaxed.py             # guards under a relaxed choice of rows A / B (not the project's rule)
$PY -I -B t1_probe_links.py ; $PY -I -B t2_probe_162.py ; $PY -I -B t3_probe_longgap.py     # small probes -> out/t*.txt
```

Order matters only in that r2, r5-r9, r11, r12 read what r1 and r4 wrote. r3 needs free memory:
a first attempt with 6 workers next to another agent's job exhausted the machine's commit limit
and other Python starts failed with exit code 127 until it finished (nothing was lost).

| file | labels? | what |
|---|---|---|
| `vr_common.py` | no | imports fuse's apply path read-only; cache redirection; pooling helpers |
| `vr_labels.py` | yes (diagnosis, scoring) | identities of pieces from `reid_lab.score`'s matching; annotated team / role / jersey per person; per-return details |
| `r1`, `r3`, `r9` | scoring only | score.py's rule on method outputs |
| `r2` | no (reads r1's scored outcomes) | statistics |
| `r4`-`r8`, `r10`-`r12`, `t*` | yes, diagnosis | never method results |
| `out/*.md`, `out/*.json`, `out/*.csv` | | every table quoted below |
| `cache/` | | tables for the shifted thresholds (p_cut 0.63 / 0.77, MAX_SPEED, START_RADIUS) and `r3_raw.pkl` |

No METHOD.json of my own: this directory tests fuse's files. The guard variants of r9 are
diagnosis, not a proposed method.

## What holds

- **Reproduction.** All seven METHOD files re-score to exactly the claimed numbers, and rebuilding
  them through fuse's label-free apply path from `params_loco.json` gives identical links and cuts
  for every clip (`out/r1_reproduce.md`). Baselines reproduce (split 0 / 4 / 133, plugin 30 / 14 / 93,
  mutual 26 / 6 / 105).
- **LOCO integrity.** The 39 margins re-derive from fuse's raw results with fuse's rule; the pair
  models of folds SNGS-162 and SNGS-107 refit to the stored weights exactly (difference 0) and their
  fit tables hold no row of the held-out clip; held-out is not worse than in-fit (parameter transfer,
  `out/r10_folds.md`: summed over clips the held-out fold differs from the mean of the 12 in-fit folds
  by -0.3 to +0.7 right returns at margins 0.2 - 0.8, -4.8 at 0.9). A grep of the modules on the
  apply path finds no annotation or score import.
- **The gain is not carried by one or two clips.** Balanced, recall and fixed margin 0.5 beat the
  plugin in right returns in 13 of 13 clips; the two best clips hold 27 - 28% of the right returns
  (and 21 - 23% of the scored returns); removing any two clips leaves the right share at 0.63 - 0.75
  (balanced), 0.75 - 0.90 (recall), 0.61 - 0.73 (margin 0.5) (`out/r2_perclip.md`).
- **Teams and roles.** No link joins two pieces with different team labels; no group shares a frame,
  holds a referee or has a second partner. The 3 links (recall) whose role labels differ are one
  annotated goalkeeper each time (the plugin's role label flickers). Of the links score.py judges
  wrong, none joins two annotated teams or roles.
- **Long gaps (> 10 s; 44 of 167 returns, 34 scored).** recall 25 right / 0 wrong / 9 lost, balanced
  and margin 0.5 17 / 0 / 17; plugin 0 / 3 / 31, mutual 0 / 1 / 33. The 40 links over more than 10 s
  (recall): 23 right / 0 wrong / 17 unjudged; by weak labels 34 agree / 3 disagree.
- **Thresholds other than the margin hardly matter at +-10%**: floor, t_abs, plugin weight, MAX_SPEED,
  START_RADIUS move a pooled result by 0 - 1 returns; t0 and the bonus weight by 0 - 9 right returns
  and at most 1 wrong; p_cut 0.63 / 0.77 by 0 - 2.

## Problems, by severity

1. **(major) The wrong count cannot be certified from 13 clips; it is one clip.** 4 of 4 (strict,
   margin 0.5) and 6 of 7 (balanced, recall) wrong returns are in SNGS-162. Clip bootstrap (2000
   resamples, seed 0): wrong share 2.5% - 97.5% interval 0 - 8.4% (strict, margin 0.5) and 0 - 13.3%
   (balanced, recall); P(wrong share <= 4/137) = 0.58 for strict, P(<= 7/137) = 0.58 for balanced and
   recall, P(<= 14/137) = 0.91 for recall. In the 36% of resamples without SNGS-162 the wrong share
   is 0 (strict) or at most 3.9%; with it the median is 3.2% / 5.8%. Against the plugin the wrong share
   is lower or equal in 99 - 100% of the resamples; against "gbf now" (split, 4 wrong of another kind)
   it is not lower: P(lower) = 0.46 strict, 0.25 balanced / recall.
2. **(major) The strict operating point is weak and fragile.** 33 right is not distinguishable from
   the plugin's 30 (5 clips better, 4 worse, 4 equal; bootstrap P(right share higher) = 0.64, against
   mutual 0.83) and it finds none of the 34 long-gap returns. Margin -10% / +10% moves it to 46 / 4 and
   23 / 4 (the 40 random +-10% draws: 16 - 50 right). The rule's penalty decides everything: lam 7 - 12
   gives 33 - 45 right at 4 wrong, lam 6 gives 59 / 6, lam 5 gives 85 / 6, so no member of the rule
   family reaches the fixed-margin row 91 / 4 that looks best after the fact; and that row sits on an
   edge: margin -10% or bonus -10% gives 5 wrong, 16 of the 40 random draws give 5 - 6.
3. **(major for a precision-first reading) "Fewer impure rows than gbf now" is the cuts, not the
   links.** Cuts alone: 1.00% impure rows and 0 wrong returns. The links add 314 (strict), 563
   (margin 0.5), 977 (balanced), 1001 (recall) impure rows on top, 72 - 99% of them in two clips
   (SNGS-162 +301 / +393 / +552 / +552; then SNGS-097 +172, SNGS-100 +116, SNGS-066 +83, SNGS-112 +70).
   Against split the impure rows are higher in 1 (strict), 2 (margin 0.5) and 3 (balanced, recall)
   clips; SNGS-162 goes from 3.76% (split) to 6.50% (recall). Identities holding two annotated
   people: 35 after the cuts, 40 / 44 / 49 / 49 with links (76 under split). In exchange the share of
   a player's boxes that sit in his main identity rises from 0.68 to 0.77 / 0.87 / 0.90 / 0.93.
4. **(minor) Tiny pieces act as bridges.** 2 of the 4 wrong returns of strict / margin 0.5 are one
   link through a 3-box piece with 1 clean box (SNGS-162 10000023 > 100115 > 116); at recall 5 of 7
   wrong returns run through pieces with fewer than 2 clean boxes. 54% of the links (recall) involve
   a piece under 5 boxes; by weak labels those are 91.6% right against 96.2% for the others. A guard
   removes them but is not free: "no bridge through a piece with < 2 clean boxes" as a post-filter
   gives 88 / 2 / 47 at recall (from 110 / 7 / 20) and 75 / 2 / 60 at margin 0.5 (from 91 / 4 / 42);
   barring such pieces inside the linker gives 93 / 4 / 40 and 82 / 2 / 53. The price is real, not an
   artefact of where score.py takes row B (r12). Chosen after seeing these clips: diagnosis only.
5. **(minor) The two wrong returns no guard removes** (SNGS-162 persons 9 and 12) rest on long pieces
   with strong evidence (cosine 0.85, 50 - 488 boxes). For 4 of the 5 SNGS-162 links judged wrong the
   annotation's own continuation has cosine 0.02 - 0.35 while the mirrored pair has 0.71 - 0.85: the
   signature of two annotated identities exchanged after the gap, or of the embedding failing the
   same way for both. Numbers cannot tell which; nobody has looked at the frames.
6. **(minor) Wrong links are not confined to SNGS-162.** 5 of the 10 links judged wrong at recall are
   in SNGS-097, SNGS-100 (2), SNGS-066, SNGS-112; SNGS-112 49 > 100049 survives up to margin 1.0. Only
   1 of them is counted as a wrong return because no scored return sits there.
7. **(minor) Wording.** "SNGS-162 alone: 7 right at every margin from 0.10 to 0.70" is 6 - 8 right
   (4 - 6 wrong is correct).
8. **(caveat) 30 s clips.** The longest annotated gap is 28.9 s and a clip has few candidates. For
   pairs over 10 s the position score is weak (median 0.15 - 0.18) but still adds a bonus of about 0.5
   (97% of those links have a bonus above 0.3), so long-gap joins are decided by appearance plus a
   nearly constant bonus. That worked here; a full match has many more candidates per return.

## Numbers behind the list

Tables: `out/r1_reproduce.md`, `out/r2_perclip.md`, `out/r3_sensitivity.md`, `out/r5_wrong.md`,
`out/r6_gaps.md`, `out/r7_team_role.md`, `out/r8_impure.md`, `out/r9_guards.md`, `out/r10_folds.md`,
`out/r11_suspects.md`, `out/r12_relaxed.md`; JSON next to each.

Clip bootstrap, paired differences of shares (2.5% / 50% / 97.5%):

| point | right share | wrong share | right minus plugin | wrong minus plugin | wrong minus split | wrong minus mutual |
|---|---|---|---|---|---|---|
| strict 33 / 4 / 100 | 0.177 / 0.240 / 0.313 | 0 / 0.028 / 0.084 | -0.067 / +0.019 / +0.104 | -0.126 / -0.071 / -0.030 | -0.032 / 0.000 / +0.035 | -0.034 / -0.014 / 0.000 |
| balanced 93 / 7 / 37 | 0.568 / 0.683 / 0.794 | 0 / 0.047 / 0.133 | +0.375 / +0.461 / +0.542 | -0.106 / -0.052 / -0.006 | -0.020 / +0.020 / +0.081 | -0.020 / +0.007 / +0.040 |
| recall 110 / 7 / 20 | 0.667 / 0.813 / 0.919 | 0 / 0.047 / 0.133 | +0.461 / +0.588 / +0.699 | -0.106 / -0.052 / -0.006 | -0.020 / +0.020 / +0.081 | -0.020 / +0.007 / +0.040 |
| margin 0.5 (not chosen by the rule) 91 / 4 / 42 | 0.552 / 0.669 / 0.778 | 0 / 0.028 / 0.084 | +0.357 / +0.447 / +0.532 | -0.126 / -0.071 / -0.030 | -0.032 / 0.000 / +0.035 | -0.034 / -0.014 / 0.000 |

Sensitivity, one threshold at a time by +-10% (range over all single changes):

| point | base | right | wrong | largest change |
|---|---|---|---|---|
| strict | 33 / 4 | 23 - 46 | 4 - 4 | margin -10%: +13 right |
| balanced | 93 / 7 | 87 - 96 | 6 - 7 | margin +10%: -6 right |
| recall | 110 / 7 | 109 - 111 | 7 - 8 | p_cut 0.63: +1 wrong |
| fixed margin 0.5 | 91 / 4 | 84 - 94 | 4 - 5 | margin +10%: -7 right; margin -10% or bonus -10%: +1 wrong |
| fixed margin 0.25 | 110 / 7 | 109 - 110 | 6 - 7 | |

All thresholds moved together: looser 52 / 4, 95 / 7, 112 / 7 (strict, balanced, recall), fixed 0.5
93 / 6; stricter 14 / 4, 82 / 7, 110 / 7, fixed 0.5 78 / 4.

## Caveats of this check

- The guards (r9), the relaxed rule (r12) and the exchange signature (r11) are diagnosis on the same
  13 clips with labels; none is a result. The relaxed rule is mine, not the project's.
- Weak labels (majority person of a piece with 1 - 4 matched boxes) are noisy: the 16 short-piece
  links at recall whose two majority persons are of different annotated teams all carry the same
  team label in tracks.parquet and rest on 1 - 5 matched boxes.
- +-10% was applied to fuse's thresholds, m_cuts' probability level and two of m_space's constants;
  m_appear's constants (shrinkage 0.3, 5 clean boxes, 5 co-visible frames) and the other m_space
  constants were not moved.
- I read `fuse/cache/*.pkl` (pickles written by the fuse agent) and imported `fuse_labels.py` once
  (r10, the refit check). I found no bug in `reid_lab/score.py` or `data.py`.
