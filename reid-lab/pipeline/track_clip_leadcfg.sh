#!/usr/bin/env bash
# Run sn-gamestate on one clip with the lead's 2026-11-15 runtime tracker settings (see
# run_tracklab_leadcfg.py for the patches and the differences that remain).
# Same arguments as track_clip.sh.
# Usage: track_clip_leadcfg.sh CLIP_DRIVELESS RUN_DIR [extra Hydra overrides...]
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
# The lead's runtime: no jersey OCR in the pipeline, role-only tracklet voting, tracker camera
# compensation on, NBJW with use_prev_homography (its default).
.venv/Scripts/python.exe "$TOOLS/run_tracklab_leadcfg.py" --gbf-hashes "$RUN/gbf-frame-hashes.json" -- \
  -cn soccernet \
  dataset=video \
  "dataset.video_path=$CLIP" \
  "pipeline=[bbox_detector,reid,track,pitch,calibration,tracklet_agg,team,team_side]" \
  "modules.tracklet_agg.cfg.attributes=[role]" \
  modules.track.cfg.ecc=True \
  modules.calibration.image_width=1280 \
  modules.calibration.image_height=720 \
  modules.calibration.use_prev_homography=true \
  eval_tracking=false \
  visualization.save_videos=false \
  "state.save_file=$RUN/state.pklz" \
  "hydra.run.dir=$RUN/hydra" \
  model_dir=C:/Research/plugins/sn-gamestate/pretrained_models \
  "data_dir=$RUN/data" \
  num_cores=0 \
  "$@"
