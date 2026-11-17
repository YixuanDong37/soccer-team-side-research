# diag_cues: commands and files

Diagnosis of the label-free cues on the 13 dev clips (hard rule 5c: labels are used freely here,
so nothing in this folder is a method and there is no METHOD.json). Every script prints its
tables; the commands below keep them in `out/*.log` and the scripts also write `out/*.csv` /
`out/*.json`.

## Run

Git Bash, from this folder. Python is gbf's venv, always with `-I` (the scripts add their own
folder and `C:/Research/plugins/tools` to `sys.path`).

```bash
cd C:/Research/plugins/tools/reid_lab/work/diag_cues
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
$PY -I t0_sanity.py               > out/t0_sanity.log 2>&1              # ~30 s the first time (builds out/cache), then 5 s
$PY -I t1_team_role.py            > out/t1_team_role.log 2>&1           # (1) team, (6) role
$PY -I t2_position.py             > out/t2_position.log 2>&1            # (2) position, speed gate
$PY -I t3_camera.py               > out/t3_camera.log 2>&1              # (3) camera view, side and path cues
$PY -I t3b_calib_sanity.py        > out/t3b_calib_sanity.log 2>&1       # (3) implausible calibrations
$PY -I t4_jersey.py               > out/t4_jersey.log 2>&1              # (4) jersey
$PY -I t5a_appearance_rows.py     > out/t5a_appearance_rows.log 2>&1    # (5) appearance, single rows
$PY -I t5b_appearance_pieces.py   > out/t5b_appearance_pieces.log 2>&1  # (5) appearance, piece prototypes
$PY -I t5c_appearance_wccn.py     > out/t5c_appearance_wccn.log 2>&1    # (5) needs t5b's parquet
$PY -I t5d_bar_transfer.py        > out/t5d_bar_transfer.log 2>&1       # (5) needs t5c's parquet; LOCO check of one bar
$PY -I t5e_unlabelled_pieces.py   > out/t5e_unlabelled_pieces.log 2>&1  # (5) team-centring for pieces without a team label
$PY -I t7_candidates.py           > out/t7_candidates.log 2>&1          # the cues together (in-sample)
$PY -I t8_oracle.py               > out/t8_oracle.log 2>&1              # oracle ceilings via reid_lab.score
```

Whole run: about 3 minutes. Random sampling (t5a, t5b) is seeded. Delete `out/cache/` to rebuild
the matching.

## Files

| file | what it is |
|---|---|
| `common.py` | LABEL-BASED helpers: annotation with team / jersey / role, the scorer's matching (asserted equal to `reid_lab.score.clip`), piece identities, the returns with rows A and B, AUC and precision-at-recall |
| `camview.py` | label-free camera helpers from calib.parquet: per-frame projection (checked against `gbf.geometry.project`), visible pitch area, "is a person at this ground point in the picture" |
| `t0_sanity.py` | counts: 167 returns, 137 scored, pieces, matched rows |
| `t1_team_role.py` | team and role labels against the annotation: rows, pieces (edge pieces, re-entry pieces, return pieces), pairs |
| `t2_position.py` | coordinate convention; deviation of pitch_x / pitch_y by edge, distance, calib_status, height; exit-to-re-entry displacement against gap time; the gate d <= v T + e on true returns and on piece pairs |
| `t3_camera.py` | view per frame; exit side against re-entry side; camera-driven exits; exit point and straight path during the gap; side and path cues on piece pairs; funnel |
| `t3b_calib_sanity.py` | frames with an implausible focal length or camera centre |
| `t4_jersey.py` | tracks.jersey / jersey_conf against annotated numbers |
| `t5a_appearance_rows.py` | cosine of single embeddings by box height, clipping, overlap, time apart, confidence, clip |
| `t5b_appearance_pieces.py` | 24 prototype / score choices on piece pairs (AUC, precision at recall, rank-1); dependence on rows, height, edge share, overlap, time, purity; rows averaged; per clip; one bar for all clips; the scored returns |
| `t5c_appearance_wccn.py` | exploratory: within-piece whitening on top of team-centring |
| `t5d_bar_transfer.py` | leave-one-clip-out transfer of one cosine bar (pair level) |
| `t5e_unlabelled_pieces.py` | how to centre a piece that has no team label (clip mean, nearer team mean, partner's team) |
| `t7_candidates.py` | all earlier pieces as candidates of a later piece: what each gate keeps, top-1 by appearance, bar + margin grid |
| `t8_oracle.py` | ORACLE: ceilings of link-only and cut-and-link methods under `reid_lab.score.evaluate` |
| `out/*.log` | the printed tables of each script (the reference for every number quoted) |
| `out/*.csv`, `out/*.json`, `out/*.parquet` | the same tables and the pair tables behind them |
| `out/cache/` | matching and returns per clip (derived from the annotation; local only) |

Everything under `out/` is derived from SN-GSR annotations: numbers only, do not publish rows.
