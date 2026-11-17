# diag_failures: how to re-run

Diagnosis of the 167 annotated returns of the 13 dev clips (rule 5c: labels are used freely
here; nothing in this folder is a method, and no number is a method result).

Python: `C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I` (always with `-I`).
Run from this folder, in this order (`run_all.sh` does exactly this):

```
PY="C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I"
$PY pathg_audit.py             # re-runs gbf's stitch audit with every abstention listed -> out/pathg_pairs.json
$PY diagnose_returns.py        # one row per return + oracle ceilings -> out/returns.csv, out/oracles.json, out/METHOD_oracle_*.json
$PY mutual_links.py            # every link of the mutual rule, judged -> out/mutual_links.csv
$PY mutual_funnel.py           # where the mutual rule loses each scored return -> out/mutual_funnel.csv
$PY plugin_joins.py            # the plugin's 93 re-entry joins with label-free signals -> out/plugin_joins.csv
$PY undetected.py              # the 30 undetected returns, one row per missing side -> out/undetected.csv
$PY fragments.py               # fragment / foreign pieces under row A and row B -> out/fragments.csv
$PY switches.py                # returns needing a cut; every mixed piece -> out/switch_returns.csv, out/mixed_pieces.csv
$PY holes.py                   # holes >= 10 frames inside pieces: same person or not -> out/holes.csv
$PY swap_suspects.py           # pairs of returns whose identities look exchanged -> out/swap_suspects.csv
$PY id_split_suspects.py       # annotated ids that may be one player -> out/id_split_suspects.csv
$PY rank_ceiling.py            # how well descriptors rank the true earlier piece -> out/rank_ceiling.json
$PY geometry_rank.py           # the same by pitch position -> out/geometry_rank.json
$PY piece_stats.py             # piece counts -> out/piece_stats.json
$PY probe_hole_cuts.py         # label-free probe: cut at every hole >= H frames -> out/probe_hole_cuts.json, out/METHOD_probe_hole_cuts_H10.json
$PY probe_hole_cuts_oracle.py  # oracle links on top of those cuts -> out/probe_hole_cuts_oracle.json
$PY probe_plugin_filter.py     # in-sample probe: plugin joins kept by cosine -> out/probe_plugin_filter.json
$PY make_tables.py             # final CSV + tables -> returns.csv, out/tables.md, out/appendix.md
```

Scoring a METHOD file with the lab's CLI, for example the label-free hole-cut probe:

```
$PY C:/Research/plugins/tools/reid_lab/score.py out/METHOD_probe_hole_cuts_H10.json out/SCORE_probe_hole_cuts_H10.json
```

## Files

- `returns.csv`: one row per annotated return (167). Main columns: the person and the gap
  (`ref_id`, `gap_s`, `exit_side`, `reentry_side`, `ann_*` pitch positions, distance, speed),
  `status`, pieces A and B (`piece_a/b`, `plugin_a/b`, `a_/b_kind` own / fragment / foreign,
  rows, clean rows, team, verified team, role, purity), `out_plugin`, `out_split`, `out_mutual`
  and the oracle outcomes, `is_candidate` and `not_candidate_why`, `cos_strict`, `cos_any`,
  `mutual_status`, ranks and margins among the candidates, `pathG`, the system's positions
  (`sys_*`), the own pieces next to the gap (`anchor_a/b`, `anchors_*`, `imp_best_*`,
  `true_minus_best_impostor`), `ceiling_class`, the mutual funnel (`funnel`, `fn_*`),
  `label_suspect_exchange`, `appearance_contradicts_annotation`, `undetected_why`.
- `out/tables.md`: every summary table (T1-T17). `out/appendix.md`: a compact table of all 167
  returns and the returns that are wrong under the plugin.
- `out/METHOD_oracle_*.json`: the baselines and the oracles as METHOD files for score.py
  (oracles use labels: ceilings only). `out/METHOD_probe_hole_cuts_H10.json`: the label-free
  probe (cuts only).
- `common.py`: shared helpers (per-clip context with labels, piece table, candidate gate,
  mutual rule status).
- `peek*.py`, `explore*.py`: ad-hoc looks made on the way (they still run); the numbers quoted
  come from the scripts above.

## Rules kept

Only the 13 dev clips; nothing outside this folder was modified; no image or video was opened;
no network, GPU, tracker or batch script; `score.py` and `data.py` were not edited (no bug
found). The probes `probe_hole_cuts.py` and `probe_plugin_filter.py` read no label except
through `score.evaluate`; the second one's thresholds were read off the 13 clips (in-sample).
