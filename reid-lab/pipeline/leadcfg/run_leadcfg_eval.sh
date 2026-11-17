#!/usr/bin/env bash
# Score the 13 dev clips re-run with the lead's tracker settings (gsr-runs-leadcfg).
#
# This lab copy (tools_leadcfg/reid_lab) reads only gsr-runs-leadcfg; its data.py RUNS points
# there, and every path was rewritten to the copy.
#
#   T (transfer): the frozen fuse_r1 method with the parameters fitted on the EARLIER runs, as
#                 they are. Nothing is refitted on the new runs.
#   R (refit):    the same, pre-registered fitting procedure run again on the new runs:
#                 m_cuts make_methods, m_space run_position_linker + posscore, fuse_r1 prep,
#                 fit_loco and method. Old fits and caches are removed first.
#
# Usage: run_leadcfg_eval.sh [T|R|all]   (default all; R needs T's outputs saved first)
set -euo pipefail
WHAT="${1:-all}"
LAB="C:/Research/plugins/tools_leadcfg/reid_lab"
PY="C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe"
RUNS="C:/Research/gbf-data/gsr-runs-leadcfg"
LOGS="$LAB/eval-logs"
mkdir -p "$LOGS"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/run.log"; }
CLIPS="SNGS-107 SNGS-163 SNGS-106 SNGS-097 SNGS-067 SNGS-100 SNGS-066 SNGS-161 SNGS-169 SNGS-162 SNGS-151 SNGS-102 SNGS-112"
for c in $CLIPS; do [ -f "$RUNS/$c/.done-eval" ] || { log "missing run: $c"; exit 2; }; done

score_all() {   # score every METHOD_*.json of fuse_r1 into $1
  local dest="$1"; mkdir -p "$dest"
  for m in "$LAB"/work/fuse_r1/METHOD_*.json; do
    local name; name=$(basename "$m" .json); name=${name#METHOD_}
    "$PY" -I "$LAB/score.py" "$m" "$dest/OUT_$name.json" > "$dest/score_$name.log" 2>&1
    cp "$m" "$dest/"
  done
}

if [ "$WHAT" = all ] || [ "$WHAT" = T ]; then
  log "check: score.py against each clip's reentry-eval.json"
  (cd "$LAB" && "$PY" -I check_score.py) > "$LOGS/check_score.txt" 2>&1
  tail -2 "$LOGS/check_score.txt" | tee -a "$LOGS/run.log"
  log "baselines"
  "$PY" -I -X utf8 -c "
import json, sys
sys.path.insert(0, 'C:/Research/plugins/tools_leadcfg')
from reid_lab import score
out = {}
for name in ('plugin', 'split', 'mutual'):
    ev = score.evaluate(score.baseline_links(name))
    out[name] = ev
    p = ev['pooled']
    print(name, {k: p[k] for k in ('scored', 'right', 'wrong', 'lost', 'undetected', 'untracked', 'links_right', 'links_wrong', 'links_unknown', 'impure_row_share')})
json.dump(out, open(sys.argv[1], 'w', encoding='utf-8'), indent=1)
" "$LOGS/baselines.json" | tee -a "$LOGS/run.log"
  log "T: evidence tables with the earlier fits (prep.py)"
  (cd "$LAB/work/fuse_r1" && "$PY" -I -B prep.py 5) > "$LOGS/T_prep.log" 2>&1
  log "T: METHOD files with the earlier fits (method.py)"
  (cd "$LAB/work/fuse_r1" && "$PY" -I -B method.py) > "$LOGS/T_method.log" 2>&1
  score_all "$LAB/out_T"
  log "T done"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = R ]; then
  [ -d "$LAB/out_T" ] || { log "R needs out_T first"; exit 3; }
  log "R: remove the earlier fits and every cache"
  W="$LAB/work"
  rm -f "$W/m_cuts/final_params.json" "$W/m_cuts/loco_params.json" "$W/m_cuts/cutlink_params.json" "$W/m_cuts/tab_cache.pkl"
  rm -rf "$W/m_cuts/methods"; mkdir -p "$W/m_cuts/methods"
  rm -rf "$W/m_space/params_cache" "$W/m_space/params_all13.json" "$W/m_space/params_loco.json"
  rm -rf "$W/m_appear/cache"; mkdir -p "$W/m_appear/cache"
  rm -rf "$W/fuse_r1/cache" "$W/fuse_r1/params_loco.json" "$W/fuse_r1"/METHOD_*.json "$W/fuse_r1"/OUT_*.json
  log "R: m_cuts make_methods.py"
  (cd "$W/m_cuts" && "$PY" -I make_methods.py) > "$LOGS/R_m_cuts.log" 2>&1
  log "R: m_space run_position_linker.py"
  (cd "$W/m_space" && "$PY" -I run_position_linker.py) > "$LOGS/R_m_space_linker.log" 2>&1
  log "R: m_space posscore.py"
  (cd "$W/m_space" && "$PY" -I posscore.py) > "$LOGS/R_m_space_posscore.log" 2>&1
  log "R: fuse_r1 prep.py"
  (cd "$W/fuse_r1" && "$PY" -I -B prep.py 5) > "$LOGS/R_prep.log" 2>&1
  log "R: fuse_r1 fit_loco.py"
  (cd "$W/fuse_r1" && "$PY" -I -B fit_loco.py 5) > "$LOGS/R_fit_loco.log" 2>&1
  log "R: fuse_r1 method.py"
  (cd "$W/fuse_r1" && "$PY" -I -B method.py) > "$LOGS/R_method.log" 2>&1
  score_all "$LAB/out_R"
  log "R done"
fi
