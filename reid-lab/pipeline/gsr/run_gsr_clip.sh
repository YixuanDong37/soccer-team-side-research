#!/usr/bin/env bash
# One SN-GSR-2025 train clip through the same pipeline as the private clip, then the re-entry
# check against its annotation. Resumable: each finished step leaves OUT/.done-<step>; running a
# step first removes the markers of every later step, so outputs of different runs never mix.
# Usage: run_gsr_clip.sh SNGS-nnn
set -euo pipefail
export MSYS_NO_PATHCONV=1
SEQ="$1"
SRC="C:/Research/gbf-data/external/sn-gsr-2025-ce84ee7/train/$SEQ"
OUT="C:/Research/gbf-data/gsr-runs/$SEQ"
OUT_DL="/Research/gbf-data/gsr-runs/$SEQ"   # drive-less: tracklab splits "vid://p:f" at the colon
TOOLS="C:/Research/plugins/tools"
SNGS="C:/Research/plugins/sn-gamestate"
GBF="C:/Research/soccer-team-side-research/gbf"
STEPS="video plugin ball export gbf embeddings appearance eval"
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

if ! has video; then   # 720p like the private clip, H.264 CRF 18, 25 fps CFR, silent audio track
  start video
  ffmpeg -nostdin -v error -y -framerate 25 -start_number 1 -i "$SRC/img1/%06d.jpg" \
    -f lavfi -i anullsrc=r=48000:cl=stereo -shortest \
    -vf "scale=1280:720:flags=lanczos" -c:v libx264 -crf 18 -pix_fmt yuvj420p -r 25 \
    -c:a aac -b:a 32k "$OUT/video.mp4"
  n=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$OUT/video.mp4")
  [ "$n" = "750" ] || { echo "$SEQ: video has $n frames, not 750"; exit 3; }
  done_ video
fi
if ! has plugin; then
  start plugin
  rm -f "$OUT/plugin/state.pklz"
  bash "$TOOLS/track_clip.sh" "$OUT_DL/video.mp4" "$OUT/plugin" dataset.eval_set=val > "$OUT/plugin.log" 2>&1
  [ -f "$OUT/plugin/state.pklz" ] || { echo "$SEQ: no state.pklz"; exit 4; }
  done_ plugin
fi
if ! has ball; then   # gbf's command wants the ball file; stages import, shots and refine do not read it
  start ball
  (cd "$SNGS" && .venv/Scripts/python.exe -I "$TOOLS/ball_detect.py" --video "$OUT/video.mp4" \
     --weights "$SNGS/pretrained_models/yolo/yolo11m.pt" --out "$OUT/ball") > "$OUT/ball.log" 2>&1
  done_ ball
fi
if ! has export; then
  start export
  rm -rf "$OUT/sngs-kp"
  (cd "$SNGS" && .venv/Scripts/python.exe -I "$TOOLS/export_sngs.py" --state "$OUT/plugin/state.pklz" \
     --video "$OUT/video.mp4" --out "$OUT/sngs-kp" --provenance "$TOOLS/gsr/provenance-gsr.json") > "$OUT/export.log" 2>&1
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
