#!/usr/bin/env bash
# Run sn-gamestate on one clip with the frame-exact reader, then export it for gbf.
# Usage: track_clip.sh CLIP_DRIVELESS RUN_DIR [extra Hydra overrides...]
#   CLIP_DRIVELESS: the clip's path without the drive letter (tracklab splits "vid://p:f" at
#   the colon), e.g. /data/clips/match-01/clip.mp4
#   RUN_DIR: Windows path for the run's outputs, e.g. D:/data/plugin-runs/x
set -euo pipefail
export MSYS_NO_PATHCONV=1   # keep Git Bash from rewriting drive-less /folder/... arguments
CLIP="$1"; RUN="$2"; shift 2
TOOLS="C:/Research/plugins/tools"
GBF="C:/Research/soccer-team-side-research/gbf"
mkdir -p "$RUN"
rm -rf "$RUN/hydra"   # prtreid refuses to start when its reid/0 folder from an earlier run exists

# gbf's own frame hashes for this clip (first, last and a middle frame)
FRAMES=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "C:$CLIP")
"$GBF/.venv/Scripts/python.exe" "$TOOLS/gbf_frame_hashes.py" "C:$CLIP" "$CLIP" "$RUN/gbf-frame-hashes.json" 0 1 2 $((FRAMES / 2)) $((FRAMES - 1))

cd C:/Research/plugins/sn-gamestate
.venv/Scripts/python.exe "$TOOLS/run_tracklab.py" --gbf-hashes "$RUN/gbf-frame-hashes.json" -- \
  -cn soccernet \
  dataset=video \
  "dataset.video_path=$CLIP" \
  modules/jersey_number_detect=easyocr \
  modules.calibration.image_width=1280 \
  modules.calibration.image_height=720 \
  modules.calibration.use_prev_homography=false \
  eval_tracking=false \
  visualization.save_videos=false \
  "state.save_file=$RUN/state.pklz" \
  "hydra.run.dir=$RUN/hydra" \
  model_dir=C:/Research/plugins/sn-gamestate/pretrained_models \
  "data_dir=$RUN/data" \
  num_cores=0 \
  "$@"
