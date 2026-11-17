# v_robust_1: robustness check of the fused re-linker round 1 (work/fuse_r1), verification round 1

Scope: the 13 SN-GSR dev clips only. Nothing outside this directory was written: fuse_r1's modules are
imported read-only (no bytecode; its cached table builders are redirected to `cache/` here), and
`fuse_r1/cache`, `fuse_r1/params_loco.json` and `fuse_r1/METHOD_*.json` are only read. No image or video
was opened, no git, no network, no package install. Python: gbf's venv, always `-I -B`.
Run on 2026-11-15, about 08:30 - 09:35 US Central time.

```
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_robust_1
$PY -I -B r1_reproduce.py            # re-score the 10 METHOD files, rebuild them from params_loco.json, baselines          (1 min)
$PY -I -B r2_perclip_bootstrap.py    # per clip, concentration, clip bootstrap (2000 resamples, seed 0, paired)            (seconds)
$PY -I -B r3_sensitivity.py 4        # +-10% on every threshold and constant, all loose / all strict, 60 random draws      (20 min, 4 workers, about 1 GB each)
$PY -I -B r3_sensitivity.py tables   # re-render out/r3_sensitivity.md from cache/r3_raw.pkl
$PY -I -B r4_link_table.py           # one row per link with the linker's evidence and the labels -> out/links_*.csv, out/returns_*.csv
$PY -I -B r5_wrong.py                # the wrong returns and wrong links one by one
$PY -I -B r6_gaps.py                 # outcomes and link precision by gap length (long gaps > 10 s)
$PY -I -B r7_team_role.py            # links across team / role labels (label-free part + annotated part)
$PY -I -B r8_impure.py               # impure rows per clip against split and against cuts only; mixed identities
$PY -I -B r9_guards.py               # simple guards: post-filters on the links and changes inside the linker             (15 min, run after r3)
$PY -I -B r10_rules.py               # rule B re-derived from fuse_r1's raw LOCO results; the rule's own numbers moved; parameter transfer
$PY -I -B r11_suspects.py            # wrong links against the annotation's own continuation (exchange signature)
$PY -I -B r12_mixed.py               # returns scored right inside an identity that is not clean; leaf links by weak labels
$PY -I -B t1_probe_split_wrong.py ; $PY -I -B t2_probe_bonus.py ; $PY -I -B t3_probe_roles.py
$PY -I -B t4_probe_clips.py ; $PY -I -B t5_probe_066.py ; $PY -I -B t6_probe_fixed.py     # small probes -> out/t*.txt (t0_smoke.py: one-clip test of r3)
sh run.sh <script.py> [args]         # the same call with a retry on exit code 127 (see below); log in out/<script>.log
```

Order: r2, r5-r9, r11, r12 and the probes read what r1 and r4 wrote; r9 must not run while r3 is building
tables into `cache/`. While another agent's 6-worker job was running the machine's commit limit was nearly
reached and the interpreter several times did not start (exit code 127, no output); the same command ran
normally when repeated. Every result below comes from a run that ended with exit code 0.

| file | labels? | what |
|---|---|---|
| `vr_common.py` | no | imports fuse_r1's apply path read-only; cache redirection; pooling helpers |
| `vr_labels.py` | yes (diagnosis, scoring) | identities of pieces from `reid_lab.score`'s matching; annotated team / role / jersey per person |
| `r1`, `r3`, `r9` | scoring only | score.py's rule on method outputs |
| `r2`, `r10`, `t6` | no (read scored outcomes) | statistics; r10 and t6 work on fuse_r1's `cache/loco_raw.pkl` |
| `r4`-`r8`, `r11`, `r12`, `t1`-`t3`, `t5` | yes, diagnosis | never method results |
| `t4` | no (+ r1's scored counts for the last column) | label-free numbers per clip |
| `out/*.md`, `out/*.json`, `out/*.csv`, `out/*.txt` | | every table quoted below |
| `cache/` | | tables for the shifted thresholds and `r3_raw.pkl` |

No METHOD.json of my own: this directory tests fuse_r1's files. The guards of r9 are diagnosis, not a
proposed method.

## Verdict

The numbers reproduce exactly and the gain in right returns is broad and stable. The claim that wrong
identities stay rare is not certified: it rests on one clip, on a rule for the operating points whose
structure (not its numbers) decides the strict point, and the linking step adds wrong merges that the
return count does not show. fuse_r1's report says most of this itself; nothing material was hidden.

## What holds

- **Reproduction.** All ten METHOD files re-score to the claimed numbers, and rebuilding them through
  fuse_r1's label-free apply path from `params_loco.json` gives identical links and cuts for every clip.
  Baselines reproduce (split 0 / 4 / 133, plugin 30 / 14 / 93, mutual 26 / 6 / 105). The 42 settings
  (3 points x 14 folds) re-derive from fuse_r1's raw LOCO results with its `rules.choose` and with my own
  re-implementation of rule B, and the fixed-setting table re-pools to fuse_r1's 32 rows
  (`out/r1_reproduce.md`, `out/r10_rules.md`, `out/t6_probe_fixed.txt`).
- **The gain is not carried by one or two clips.** Strict has more right returns than the plugin in 12 of
  13 clips (SNGS-106: 3 against 4), balanced in 12 (1 equal), recall in 12 (1 equal). The two best clips
  hold 31% / 28% / 27% of the right returns (they hold 20 - 21% of the scored returns). With any two clips
  removed the right share stays at 0.42 - 0.57 (strict), 0.56 - 0.71 (balanced), 0.71 - 0.85 (recall)
  (`out/r2_perclip.md`).
- **Clip bootstrap (2000 resamples, seed 0), right returns.** Right share, 2.5% / 50% / 97.5%: strict
  0.36 / 0.49 / 0.61, balanced 0.48 / 0.63 / 0.75, recall 0.65 / 0.78 / 0.89. Paired against plugin, split
  and mutual the right share is higher in 100% of the resamples at every point (strict minus plugin
  +0.15 / +0.27 / +0.39). Against the plugin the wrong share is lower in 100% / 100% / 98% of the
  resamples (strict / balanced / recall); against mutual in 89% / 66% / 28%; against split see problem 1.
- **Fitted numbers transfer.** At the frozen settings a clip scores the same under the numbers fitted
  without it as under the 12 folds that had it in their fit: held-out minus in-fit +0.7 / -1.2 / -0.3 right
  returns and 0.0 / 0.0 / -0.1 wrong (strict / balanced / recall, summed over the clips).
- **+-10% on thresholds and constants** (`out/r3_sensitivity.md`; nothing refitted or re-chosen). Strict:
  57 - 74 right and 3 wrong under every single change; balanced 84 - 86 / 5; recall 105 - 107 / 7 - 8 (the
  8th wrong return appears with the bonus or the position weight at -10%, in SNGS-067). I added what
  fuse_r1 had not moved: t_abs, the plugin weight alone, m_space's MAX_SPEED and START_RADIUS, m_appear's
  WCCN_LAMBDA and MIN_CLEAN: at most 2 right returns and no wrong return move (t_abs and the plugin
  weight change nothing at all). 40 random joint draws with another seed: strict
  54 - 73 right, 3 wrong in 39 draws and 4 in one; 20 more draws that also move t_abs, the solid boundary
  (4 / 5 / 6) and the cut level: 52 - 71 right, 3 wrong in 19, 4 in one. No draw exceeds a limit at any
  point. Frozen strict setting: 63 - 76 right and 2 - 3 wrong under single changes; over the 60 draws 56 - 75
  right, 2 wrong in 43 draws, 3 in 15, 4 in 2.
- **Teams and roles** (`out/r7_team_role.md`, `out/t3_probe_roles.txt`). In all nine METHOD files no link
  joins two pieces with different team labels, no group shares a frame and no referee is linked. Role
  labels differ in 2 links at recall (1 at the frozen balanced setting): both are one annotated goalkeeper
  whose role label flickers in the tracks. By the annotation, no link that score.py can judge joins two
  teams at strict or balanced; at recall one does (SNGS-066 18 > 37: the tracks call both pieces "left",
  the annotation puts 23 of piece 18's 49 matched boxes on a right-team player).
- **Long gaps (> 10 s; 44 of 167 returns, 34 scored)** (`out/r6_gaps.md`). Strict 11 right / 0 wrong / 23
  lost, balanced 16 / 0 / 18, recall 22 / 0 / 12 (plugin 0 / 3 / 31, mutual 0 / 1 / 33). Links over more than
  10 s: strict 14, all judged right; balanced 24 right / 0 wrong / 1 unjudged; recall 35 / 0 / 11 (by weak
  labels 41 agree / 3 disagree).

## Problems, by severity

1. **(major) The wrong count is one clip and cannot be certified.** 2 of 3 (strict), 4 of 5 (balanced),
   6 of 7 (recall) wrong returns are in SNGS-162. Clip bootstrap: wrong share 2.5% - 97.5% 0 - 5.1% (strict),
   0 - 9.0% (balanced), 0 - 13.3% (recall); P(wrong share within the point's limit) = 0.75, 0.75, 0.91.
   Against gbf now (split) the wrong share is never higher at strict, but it is higher in 53% of the
   resamples at balanced and 63% at recall. On SNGS-162 itself the method gives 3 right / 2 wrong (strict),
   4 / 4 (balanced), 8 / 6 (recall), and every margin up to 0.80 leaves at least one wrong return there
   (fuse_r1's fixed-setting table). The clip does not
   stand out without labels: fuse_r1's self-check puts it inside the range of the other clips (same-person
   median 0.84, different-people 99% quantile 0.55), and it is only the most fragmented clip by a small
   step (125 pieces, 32 cuts; next 123 and 30) (`out/t4_probe_clips.txt`).
2. **(major) The strict point exists because of the rule's structure.** Rule B's numbers are harmless
   (lam x 0.9 / x 1.1: 68 / 3 and 65 / 3; lam 5 to 20: 58 - 72 right, 2 - 4 wrong; cap 3 or 5: 66 / 2, 67 / 3). Its
   structure is not (`out/r10_rules.md`): with no fit clip left out of J, or when the clip left out may
   not be SNGS-162 (it is in 12 of 13 folds), strict is 26 / 2 / 109; when the error count w keeps only
   wrong returns the limits break (strict 76 / 7, balanced 96 / 9); when only the weak-label leaf term is
   dropped they break too: strict 70 / 5, because the one fold that holds SNGS-162 out then turns leaves
   on (SNGS-162 6 / 4 / 8), and balanced 96 / 8 with leaves on in every fold. So the strict limit holds
   through one term in one fold. The rule was fixed before fuse_r1's
   results but with round 0's held-out tables known. The fixed-margin rows are the stable description
   (leaves off: 0.50 81 / 4, 0.55 76 / 3, 0.60 70 / 2, 0.70 58 / 2), and they are not held-out for the choice.
3. **(major for a precision-first reading) Linking adds wrong merges that the return count hides.**
   Against cuts only (0 wrong, 1.00% impure rows, 35 identities holding two people) the links add 673 /
   1236 / 1293 impure rows and 7 / 13 / 14 mixed identities. SNGS-162 ends above gbf now at every point
   (4.31%, 8.37%, 8.57% against 3.76%), SNGS-112 too (0.64 - 0.95% against 0.47%), SNGS-097 at balanced and
   recall (3.5% against 3.3%). Returns scored right inside an identity that also holds >= 5 boxes of
   another person: 2 / 4 / 7, so wrong + mixed is 5 / 9 / 14 and "right and clean" 65 / 81 / 99
   (`out/r8_impure.md`, `out/r12_mixed.md`). Pooled, the impure share stays below gbf now (1.61 / 2.12 / 2.17%
   against 2.61%; lower in 100% / 83% / 81% of the resamples), and all of that margin comes from the cuts.
4. **(minor) The one wrong return outside SNGS-162 rests on a piece without a single clean box.**
   SNGS-066 person 21 is also wrong under gbf now, the plugin and mutual (there because the plugin's
   track held 14 boxes of other people first). m_cuts cuts that track (cuts only: lost); the part after
   the cut has 38 boxes, none of them clean, and the linker joins it (cosine 0.53, margins 0.58 / 0.65) to
   a 184-box piece of another annotated player. The annotation's own continuation has cosine -0.01 and
   -0.07 on the two sides, so either the embedding of the overlapped boxes or the annotation is off here
   (`out/t5_probe_066.txt`, `out/r11_suspects.md`). A piece without a clean box taking no part in the
   linker removes it at no cost in right returns: strict 67 / 2 / 68, balanced 85 / 4 / 48 (recall 98 / 6 / 33:
   -8 right). Diagnosis only: the guard was chosen after seeing these clips. At the frozen strict setting
   the link is already absent, but it comes back at margin -10% and with "solid from 6 boxes"
   (`out/r9_guards.md`).
5. **(minor) No simple guard removes the SNGS-162 wrong returns.** The two at the strict point rest on
   pieces of 50 - 488 boxes with cosine 0.85 and margins 0.76 - 0.90. Minimum clean boxes, minimum boxes, a
   higher appearance floor, a higher "nobody" level, a position-score bar and a gap limit all leave both
   in place (SNGS-162 3 / 2 / 13 or 2 / 2 / 14); "both margins >= 0.8" gives 3 / 1 / 14 and costs 25 right
   returns. For the three strict links judged wrong there that have an annotated alternative on both
   sides (4 > 96, 10000021 > 43, 10000030 > 93) the annotation's own continuation has cosine 0.02 - 0.35
   while the mirrored pair has 0.71 - 0.80, and for two of them the mirrored link is itself made at the
   looser points; at recall this holds for 5 of the clip's 10 links judged wrong (mirror 0.71 - 0.89).
   That is the signature of two annotated identities exchanged after the gap, or of the embedding
   failing in both directions. Numbers cannot tell which (`out/r11_suspects.md`).
6. **(minor) Wrong links are not confined to SNGS-162.** Links judged wrong outside it: 3 of 7 (strict:
   SNGS-100, SNGS-066, SNGS-112), 6 of 16 (balanced), 7 of 17 (recall). Only one of them is a wrong return
   because no scored return sits at the others.
7. **(minor) Leaves.** 20 of the 187 leaf links that weak labels can judge join two different people
   (10.7%; 11 of the 20 outside SNGS-162; 1-box leaves 9 of 102, 2-box 8 of 47). They add 2 wrong
   returns, both in SNGS-162, and at margin 0.20 raise the links score.py cannot judge from 36 to 311.
8. **(minor) The tier boundary is safe only from 4 boxes up.** Solid from 3 boxes: strict 69 / 5 (limit
   broken), balanced 88 / 6; from 2: 68 / 4, 91 / 6; from 4, 6, 10: 3 wrong; from 20: 61 / 3, recall 95 / 8.
9. **(minor) Balanced is a plateau, not a point.** The folds choose margins 0.15 - 0.50; with ties broken
   towards the smaller margin the all-13 choice moves from 0.40 to 0.20, with lam x 0.9 to 0.25. The
   pooled held-out result hardly moves (84 - 86 right, 5 wrong), the setting to freeze does.
10. **(caveat) The long-gap result belongs to 30 s clips.** At the strict point no link is made on
    appearance alone (the bar is 1.10): every one needs a position bonus, which for gaps over 10 s is
    0.44 - 0.60 from a position score of 0.11 - 0.44 among the few candidates of a short clip. With more
    candidates the bonus falls (score 0.01 gives 0.21) and such joins drop below the bar, so on long
    footage they would turn into lost returns, not wrong ones. Only 3 scored returns are longer than 20 s
    (strict 0 right, balanced and recall 1) (`out/t2_probe_bonus.txt`).
11. **(wording)** "3 wrong in all 40 draws" holds for fuse_r1's seed; with another seed one draw in 40
    has 4 (within the limit). "No group has mixed teams or a referee" is right; two groups at recall
    carry two role labels (the same goalkeeper).

## Not checked here

- fuse_r1's stress test with degraded embeddings (`stress.py` writes into fuse_r1's directory) and its
  self-check were read, not re-run.
- Whether PRTReid saw these clips in training, and the nesting of m_cuts' and m_space's weights, are the
  leak verifier's subject.
- +-10% was applied one constant at a time with the settings kept; re-choosing the settings after each
  change would need a full LOCO run per change.
- I read pickles written by the fuse_r1 agent (`fuse_r1/cache/*.pkl`). No bug found in
  `reid_lab/score.py` or `data.py`.
