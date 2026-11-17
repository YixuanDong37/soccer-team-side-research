#!/usr/bin/env bash
# Run run_gsr_clip_leadcfg.sh on the clips given, one after another, into gsr-runs-leadcfg.
#
# Each clip gets at most CLIP_TIMEOUT. A clip that fails or times out is logged with its exit
# code, and the batch goes on. One batch at a time (lock folder).
#
# On a timeout, only processes whose command line names this batch's output folder are
# stopped, so nothing else running on the machine is touched.
# Usage: run_gsr_batch_leadcfg.sh SNGS-nnn [SNGS-nnn ...]
set -uo pipefail
RUNS="C:/Research/gbf-data/gsr-runs-leadcfg"
LOG="$RUNS/batch.log"
LOCK="$RUNS/.batch-lock"
CLIP_TIMEOUT="75m"
mkdir -p "$RUNS"
mkdir "$LOCK" 2>/dev/null || { echo "another batch holds $LOCK; stop it or remove the folder"; exit 9; }
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
CLIPS="$*"
n=0; for c in $CLIPS; do n=$((n + 1)); done
i=0; failed=0
for SEQ in $CLIPS; do
  i=$((i + 1))
  echo "[$(date '+%F %T')] start $SEQ ($i/$n)" >> "$LOG"
  timeout --kill-after=60 "$CLIP_TIMEOUT" bash C:/Research/plugins/tools/gsr/run_gsr_clip_leadcfg.sh "$SEQ" >> "$LOG" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    echo "[$(date '+%F %T')] ok $SEQ ($i/$n)" | tee -a "$LOG"
  else
    failed=$((failed + 1))
    echo "[$(date '+%F %T')] FAILED $SEQ ($i/$n) exit $rc" | tee -a "$LOG"
    if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then   # timed out: stop what this clip left running
      powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*gsr-runs-leadcfg*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" >> "$LOG" 2>&1
    fi
  fi
done
echo "[$(date '+%F %T')] batch finished: $i clips, $failed failed" | tee -a "$LOG"
[ "$failed" = 0 ]
