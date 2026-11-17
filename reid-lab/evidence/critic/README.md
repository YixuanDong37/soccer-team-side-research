# critic: commands and file index

Everything here is diagnosis or oracle analysis on the 13 dev clips (rule 5c). Nothing in this
directory is a method and no METHOD.json is proposed. Scripts read the annotation unless marked
label-free. Nothing outside this directory was written; fuse_r1's cache and METHOD files were only
read (o4, o7, o7b and o7c raise an error if a table would be written).

Interpreter: `PY="C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I -B"`, run from
`C:/Research/plugins/tools/reid_lab/work/critic`. Each script prints its table and writes the same text
to `out/<script>.txt`.

| command | what it does | labels | time |
|---|---|---|---|
| `$PY o0_games.py > out/o0_games.txt` | match id (`info.game_id`) and annotated jersey coverage per dev clip | yes | 10 s |
| `$PY o1_jersey.py` | jersey oracle: annotated number as a perfect or partial reader, as veto / confirm / both, on gbf now, cuts only and the three fuse_r1 points; today's reads | yes | 3 min |
| `$PY o2_settled.py` | settled identity of all 167 annotated returns (2 s windows instead of one row) and whole-clip consistency per person | yes | 20 s |
| `$PY o2b_settled_wrong.py` | the settled-wrong returns next to the lab scorer's verdict | yes | 40 s |
| `$PY o3_games.py` | results per match; operating-point setting chosen without the held-out clip's match (reads fuse_r1/cache/loco_raw.pkl) | yes | 10 s |
| `$PY o4_nopos.py` | fuse_r1 with the position bonus scaled down or missing for a share of pieces | scoring only | 1 min |
| `$PY o5_roster.py` | how many team-mates are out of the picture when a player returns (annotation only) | yes | 10 s |
| `$PY o5b_closed_set.py` | identity-level closed-set ceiling: probe after the return against one prototype per team-mate | yes | 30 s |
| `$PY o6_height.py` | body height in metres as a cue between team-mates | verdict only | 20 s |
| `$PY o7_or_path.py` | strict links OR a high-lead appearance-only path | scoring only | 30 s |
| `$PY o7b_appearance_only.py` | ablation: same cuts and solid tier as fuse_r1, no position model, fixed leads | scoring only | 35 s |
| `$PY o7c_appearance_only_loco.py` | the same variant with the lead chosen per fold by fuse_r1's rule B (leave one clip out, and without the held-out clip's match) | choice + scoring | 6 min |
| `$PY o8_strata.py` | fuse_r1 outcomes by box height, length of absence, boxes after the return | yes | 30 s |

Helpers: `common.py` (annotation attributes, piece identities, scoring wrappers), `peek_labels.py`
(structure of one annotation file).

Notes
- `o1_jersey.py` readability models: `all`, `known5`, `h60x10` / `h80x10` / `h100x10` (>= 10 boxes at
  least that tall, 720p, away from the edge), `p50` / `p25` / `p10` (random share of the known pieces,
  20 draws), `now` (tracks.jersey, label-free). Whether the back is turned to the camera is not modelled.
- `o3_games.py` and `o7c` nest only the choice of the setting at match level; the weights of m_cuts,
  m_space and the pair model are still leave-one-clip-out. A full leave-one-match-out refit was not run.
- `o7b` / `o7c` use fuse_r1's linker2 with no bonus, no floor and no "nobody" rival, which is m_appear's
  rule on the cut, solid pieces.
- No image or video was opened; only the 13 dev clips and their listed files were read; no git, network
  or installs.
