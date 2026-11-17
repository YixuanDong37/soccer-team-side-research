#!/usr/bin/env bash
# One SN-GSR-2025 train clip through the GSR re-entry pipeline (run_gsr_clip.sh), with the
# tracker settings of the lead's 2026-11-15 runtime (track_clip_leadcfg.sh).
#
# Outputs go to gsr-runs-leadcfg/SEQ; the earlier runs in gsr-runs stay as they are.
#
# The 720p video and the ball file are copied from the earlier run of the same clip, so the
# input pixels are identical. The video is checked by sha256; the ball file is not read by
# gbf's import, shots and refine stages.
#
# Resumable like run_gsr_clip.sh: each finished step leaves OUT/.done-<step>.
# Usage: run_gsr_clip_leadcfg.sh SNGS-nnn
set -euo pipefail
export MSYS_NO_PATHCONV=1
SEQ="$1"
SRC="C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train/$SEQ"
OLD="C:/Research/gbf-data/gsr-runs/$SEQ"
OUT="C:/Research/gbf-data/gsr-runs-leadcfg/$SEQ"
OUT_DL="/Research/gbf-data/gsr-runs-leadcfg/$SEQ"   # drive-less: tracklab splits "vid://p:f" at the colon
TOOLS="C:/Research/plugins/tools"
SNGS="C:/Research/plugins/sn-gamestate"
GBF="C:/Research/soccer-team-side-research/gbf"
STEPS="video plugin ball export gbf embeddings appearance eval"
NOTE="Local rebuild of the project's plugin-sngs export (not the original) with the tracker settings of the lead's 2026-11-15 runtime: yolo11m at 1280 px with the BGR input fix, tracker camera compensation wired (ecc), NBJW keypoints moved to cell centres, use_prev_homography=true, no jersey OCR, role-only tracklet voting; tracklab frames read through tools/exact_reader."
mkdir -p "$OUT"
has() { [ -f "$OUT/.done-$1" ]; }
done_() { touch "$OUT/.done-$1"; }
start() {   # forget every later step before redoing this one
  local seen=0
  for s in $STEPS; do
    if [ "$seen" = 1 ]; then rm -f "$OUT/.done-$s"; fi
    if [ "$s" = "$1" ]; then seen=1; fi
  done
}
t0=$(date +%s)

if ! has video; then   # the same 720p file as the earlier run (H.264 CRF 18, 25 fps CFR, silent audio)
  start video
  [ -f "$OLD/video.mp4" ] || { echo "$SEQ: no earlier video at $OLD"; exit 3; }
  cp "$OLD/video.mp4" "$OUT/video.mp4"
  a=$(sha256sum "$OLD/video.mp4" | cut -d' ' -f1); b=$(sha256sum "$OUT/video.mp4" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "$SEQ: video copy differs"; exit 3; }
  done_ video
fi
if ! has plugin; then
  start plugin
  rm -f "$OUT/plugin/state.pklz"
  bash "$TOOLS/track_clip_leadcfg.sh" "$OUT_DL/video.mp4" "$OUT/plugin" dataset.eval_set=val > "$OUT/plugin.log" 2>&1
  [ -f "$OUT/plugin/state.pklz" ] || { echo "$SEQ: no state.pklz"; exit 4; }
  # the patches must have acted: detector frames in BGR, camera updates, centred keypoints
  "$GBF/.venv/Scripts/python.exe" -I -c "
import json, sys
lines = [l for l in open(sys.argv[1], encoding='utf-8', errors='replace') if l.startswith('LEADCFG_PATCHES ')]
c = json.loads(lines[-1].split(' ', 1)[1]) if lines else {}
ok = c.get('detector_imgsz') == 1280 and c.get('detector_frames_bgr', 0) >= 750 and c.get('camera_updates', 0) > 0 and c.get('keypoint_frames_centred', 0) >= 750
print('patch counts', c)
sys.exit(0 if ok else 1)" "$OUT/plugin.log" || { echo "$SEQ: lead-settings patches did not act (see plugin.log)"; exit 4; }
  done_ plugin
fi
if ! has ball; then   # gbf's command wants the ball file; stages import, shots and refine do not read it
  start ball
  rm -rf "$OUT/ball"
  cp -r "$OLD/ball" "$OUT/ball"
  done_ ball
fi
if ! has export; then
  start export
  rm -rf "$OUT/sngs-kp"
  (cd "$SNGS" && .venv/Scripts/python.exe -I "$TOOLS/export_sngs.py" --state "$OUT/plugin/state.pklz" \
     --video "$OUT/video.mp4" --out "$OUT/sngs-kp" --provenance "$TOOLS/gsr/provenance-gsr-leadcfg.json" \
     --keypoints-centred --note "$NOTE") > "$OUT/export.log" 2>&1
  done_ export
fi
if ! has gbf; then
  start gbf
  rm -rf "$OUT/gbf"
  # the run ends with exit 1 and the note that the next stage (kinematics) was not asked for;
  # refine is all we need, so success is the refine receipt saying "passed"
  (cd "$GBF" && PYTHONUTF8=1 .venv/Scripts/python.exe -m gbf highlights --video "$OUT/video.mp4" \
     --audio-source "$OUT/video.mp4" --sngs "$OUT/sngs-kp" --ball "$OUT/ball/ball_raw.jsonl" --out "$OUT/gbf" \
     --allow-license GPL-3.0-only,AGPL-3.0-only,GPL-2.0-only --pose-model none --stages import,shots,refine) \
     > "$OUT/gbf.log" 2>&1 || true
  "$GBF/.venv/Scripts/python.exe" -I -c "import json,sys; r=json.load(open(sys.argv[1],encoding='utf-8')); sys.exit(0 if r.get('status')=='passed' else 1)" \
    "$OUT/gbf/receipts/refine.json" || { echo "$SEQ: gbf refine did not pass"; exit 5; }
  done_ gbf
fi
if ! has embeddings; then
  start embeddings
  (cd "$SNGS" && .venv/Scripts/python.exe -I "$TOOLS/export_embeddings.py" "$OUT/plugin/state.pklz" "$OUT/reid-embeddings.npz") > "$OUT/embeddings.log" 2>&1
  done_ embeddings
fi
if ! has appearance; then
  start appearance
  rm -f "$OUT/appearance-match.json"
  "$GBF/.venv/Scripts/python.exe" -I "$TOOLS/analysis/appearance_match.py" "$OUT/gbf" "$OUT/reid-embeddings.npz" \
    "$OUT/gbf" "$OUT/appearance-match.json" > "$OUT/appearance.log" 2>&1 || true
  if [ ! -f "$OUT/appearance-match.json" ]; then   # the evaluation still scores plugin, split and path G
    echo '{"failed": true, "links": [], "chosen": null, "candidate_pairs": []}' > "$OUT/appearance-match.json"
    echo "$SEQ: appearance_match failed (see appearance.log); a stub without links was written"
  fi
  done_ appearance
fi
if ! has eval; then
  start eval
  "$GBF/.venv/Scripts/python.exe" -I "$TOOLS/analysis/gsr_reentry_eval.py" "$SRC/Labels-GameState.json" "$OUT/gbf" \
    "$OUT/appearance-match.json" "$OUT/reentry-eval.json" > "$OUT/eval.log" 2>&1
  done_ eval
fi
echo "$SEQ done in $(( $(date +%s) - t0 )) s"
