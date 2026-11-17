#!/bin/sh
# v_robust_1 / run.sh -- run a script with gbf's venv python (-I -B); retry when the interpreter does not start
# (exit code 127 under memory pressure from other agents' jobs). Usage: sh run.sh script.py [args] ; log in out/<script>.log
PY=C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe
cd C:/Research/plugins/tools/reid_lab/work/v_robust_1
name=$(basename "$1" .py)
for i in 1 2 3 4 5 6 7 8; do
  "$PY" -I -B "$@" > "out/$name.log" 2>&1
  rc=$?
  if [ $rc -ne 127 ]; then echo "$name exit $rc (attempt $i)"; exit $rc; fi
  sleep 20
done
echo "$name exit 127 after 8 attempts"; exit 127
