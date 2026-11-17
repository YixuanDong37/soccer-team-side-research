# m_keep: keep the plugin's re-entry join, or keep it cut?

Run everything with gbf's Python, isolated mode, from this directory:

    PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
    cd C:/Research/plugins/tools/reid_lab/work/m_keep

## Re-run the results

    "$PY" -I loco.py > out_loco.txt            # LOCO fits + held-out METHOD files + trade-off tables (about 1 min)
    "$PY" -I bootstrap.py > out_bootstrap.txt  # clip bootstrap intervals of the held-out results
    "$PY" -I check_features.py                 # label-free self-check of features.py

    # official scorer on a METHOD file (same numbers as out/<name>.score.json written by loco.py)
    "$PY" -I C:/Research/plugins/tools/reid_lab/score.py out/keep_loco_lam1.json out/keep_loco_lam1.cli.json

    # fixed-parameter METHOD files (label-free; the parameters loco.py fits on all 13 clips)
    "$PY" -I make_method.py keep_balanced --t-keep 0.85 --no-appearance
    "$PY" -I make_method.py keep_strict   --t-keep 0.89 --n-min 15 --no-appearance
    "$PY" -I make_method.py comb_balanced --t-keep 0.85
    "$PY" -I make_method.py comb_strict   --t-keep 0.89 --n-min 15
    "$PY" -I C:/Research/plugins/tools/reid_lab/score.py out/comb_balanced.json out/comb_balanced.score.json

## Files

Label-free (never read an annotation, never import reid_lab.score):

- `features.py`: `pair_features(seq)` -> one row per plugin re-entry join gbf cut (a = previous piece of the
  plugin id, b = the re-entry piece) with the evidence listed in its docstring; `pair_features(seq, "all")` ->
  every (earlier, later) piece pair with a lighter feature set and the flag `plugin_join`;
  `describe_pair(seq, a, b)` for any pair.
- `method.py`: `keep_links(seq, t, rule="cos", n_min=1)` (the keep rule), `group_candidates`, `decide`,
  `appearance_links`, `combined`, `check_no_conflict`.
- `make_method.py`: METHOD.json from fixed parameters.
- `check_features.py`, `explore1.py`, `explore2.py`, `explore_impostors.py`: structure checks, no labels.

Label-based (scoring, LOCO fitting, diagnosis only):

- `loco.py`: leave-one-clip-out fits (objective J = right - lam * (wrong + links_wrong), lam in 1, 2, 3, 6),
  held-out METHOD files `out/<variant>_loco_lam<lam>.json`, the same fits on all 13 clips
  `out/<variant>_all13_lam<lam>.json` (in-sample), `out/loco_report.json`, trade-off tables in `out_loco.txt`.
  Variants: `keep` (a), `keep_long` (a, both pieces >= n_min boxes), `comb_lab` (b0), `comb_groups_default` (b1),
  `comb_groups_bar` (b3), `comb_long_bar` (b3 with keep_long), `comb_joint` (b2, four parameters).
- `bootstrap.py`: 13 clips resampled with replacement.
- `labels_for_fit.py`: per plugin join, the scorer's link label and the returns it turns right / wrong
  (diagnosis; writes `diag_table.csv`).
- `diag_*.py`: oracle / diagnosis runs; their numbers are not method results.
- `insample_*.py`: in-sample explorations (all 13 clips at fixed parameters; other keep scores, iterated
  merging, pairs without a runner-up, minimum piece length, what the keep step adds).

## The rules

(a) keep `[a, b]` when `cos(a, b) >= t_keep`. A piece's descriptor is the unit mean of the embeddings of its
clean rows when it has at least 5, else of all its rows with an embedding. Strict variant: also both pieces
have at least `n_min` boxes.

(b) the kept joins chain pieces into groups; then mutual best on group descriptors: earlier group X and later
group Y (players or goalkeepers, same majority team label, each at least 5 clean rows, Y starts after X ends
within 10 s, within reach at 12 m/s + 4 m), link X's last piece to Y's first piece when each is the other's
most similar candidate, the cosine is at least `bar`, and it beats the runner-up on both sides by `margin`.
Every link goes forward in time and every piece gets at most one earlier and one later partner
(`method.check_no_conflict` verifies each written file).
