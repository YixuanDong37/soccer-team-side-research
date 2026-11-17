#!/usr/bin/env bash
# Re-runs the whole diagnosis (Git Bash). DIAGNOSIS ONLY: labels are used freely (rule 5c).
set -euo pipefail
cd "$(dirname "$0")"
PY="C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I"
mkdir -p out
for s in pathg_audit diagnose_returns mutual_links mutual_funnel plugin_joins undetected fragments switches holes \
         swap_suspects id_split_suspects rank_ceiling geometry_rank piece_stats probe_hole_cuts probe_hole_cuts_oracle \
         probe_plugin_filter make_tables; do
  echo "== $s"
  $PY $s.py > out/log_$s.txt 2>&1
done
$PY C:/Research/plugins/tools/reid_lab/score.py out/METHOD_probe_hole_cuts_H10.json out/SCORE_probe_hole_cuts_H10.json > out/log_score_probe.txt 2>&1
echo "done: returns.csv, out/tables.md, out/appendix.md"
