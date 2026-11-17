#!/usr/bin/env bash
# Run run_gsr_clip.sh on every clip named in SELECTION_JSON's "selected" list (or on the clips
# given after it), one after another. Each clip gets at most CLIP_TIMEOUT; a clip that fails or
# times out is logged with its exit code and the batch goes on. One batch at a time (lock dir).
# Usage: run_gsr_batch.sh SELECTION_JSON [SNGS-nnn ...]
set -uo pipefail
SEL="$1"; shift
RUNS="C:/Research/gbf-data/gsr-runs"
LOG="$RUNS/batch.log"
LOCK="$RUNS/.batch-lock"
CLIP_TIMEOUT="75m"
PY="C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe"
mkdir -p "$RUNS"
mkdir "$LOCK" 2>/dev/null || { echo "another batch holds $LOCK; stop it or remove the folder"; exit 9; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
if [ "$#" -gt 0 ]; then CLIPS="$*"; else
  CLIPS=$("$PY" -I -c "import json,sys; print(' '.join(json.load(open(sys.argv[1], encoding='utf-8'))['selected']))" "$SEL")
fi
n=0; for c in $CLIPS; do n=$((n + 1)); done
i=0; failed=0
for SEQ in $CLIPS; do
  i=$((i + 1))
  echo "[$(date '+%F %T')] start $SEQ ($i/$n)" >> "$LOG"
  timeout --kill-after=60 "$CLIP_TIMEOUT" bash C:/Research/plugins/tools/gsr/run_gsr_clip.sh "$SEQ" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo "[$(date '+%F %T')] ok $SEQ ($i/$n)" | tee -a "$LOG"
  else
    failed=$((failed + 1))
    echo "[$(date '+%F %T')] FAILED $SEQ ($i/$n) exit $rc" | tee -a "$LOG"
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then   # timed out: stop what the clip left running
      powershell.exe -NoProfile -Command "Get-Process python,ffmpeg,ffprobe -ErrorAction SilentlyContinue | Where-Object { \$_.Path -like 'C:\\Research\\plugins\\sn-gamestate\\*' -or \$_.Path -like 'C:\\Research\\soccer-team-side-research\\gbf\\*' -or \$_.ProcessName -like 'ff*' } | Stop-Process -Force" >> "$LOG" 2>&1
    fi
  fi
done
echo "[$(date '+%F %T')] batch finished: $i clips, $failed failed" | tee -a "$LOG"
[ "$failed" = 0 ]
