"""Run sn-gamestate (tracklab) on one video with the tracker settings of the lead's 2026-11-15
desktop runtime, re-implemented here.

The settings come from that runtime's predictor/adapter/plugin_launcher.py and runner.py
(GBF-Windows-x64-20261115). Nothing from that package is imported or run.

On top of tools/run_tracklab.py (frame-exact reader, checkpoint allowlist, frame check) this
script changes four things:

1. Person detector colours. tracklab hands YOLO an RGB frame, but ultralytics treats a numpy
   array as BGR, so the network saw red and blue swapped. Hand it the BGR frame so that the
   network sees RGB.
2. Person detector size. Run it at 1280 px; ultralytics' default is 640.
3. Tracker camera motion. BPBReIDStrongSORT compensates camera motion only in
   prepare_next_frame, which tracklab never calls, so modules.track.cfg.ecc had no effect.
   Call the tracker's camera_update before each frame's update instead.
4. Pitch keypoints. NBJW decodes each heatmap peak at its cell's top-left corner. Move each
   keypoint to the cell centre (one input pixel) and rebuild the lines from the moved points.

The Hydra overrides that go with these patches are in track_clip_leadcfg.sh: ecc=True, the
pipeline without jersey OCR, role-only tracklet voting, and use_prev_homography=true.

Remaining differences from the lead's runtime:
- PyTorch on the GPU instead of OpenVINO f32 on the CPU;
- 1280x720 H.264 input read frame-exactly, instead of 1920x1080 lossless PNG frames.
At 1280 px the detector sees the 720p frame at full size; the lead's 1080p frames are scaled
down to the same 1280 px width.

Usage (from the sn-gamestate folder, its .venv), the same as run_tracklab.py:
  python ../tools/run_tracklab_leadcfg.py --gbf-hashes hashes.json -- -cn soccernet dataset=video ...
"""

from __future__ import annotations

import atexit
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tracklab  # noqa: E402  frame-exact reader, checkpoint allowlist, frame check

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tracklab.utils.cv2 as tracklab_cv2  # noqa: E402

DETECTOR_IMGSZ = 1280
COUNTS = {"detector_frames_bgr": 0, "detector_imgsz": None, "camera_updates": 0,
          "camera_update_failures": 0, "keypoint_frames_centred": 0}


def seed() -> None:
    """The lead's launcher fixes the same seeds (team clustering and other random steps)."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def patch_detector() -> None:
    from tracklab.wrappers.bbox_detector.yolo_ultralytics_api import YOLOUltralytics

    def preprocess_in_bgr(self, image, detections, metadata):
        COUNTS["detector_frames_bgr"] += 1
        return {"image": np.ascontiguousarray(image[:, :, ::-1]), "shape": (image.shape[1], image.shape[0])}

    init = YOLOUltralytics.__init__

    def init_at_1280(self, cfg, device, batch_size, **kwargs):
        init(self, cfg, device, batch_size, **kwargs)
        self.model.overrides["imgsz"] = DETECTOR_IMGSZ
        COUNTS["detector_imgsz"] = DETECTOR_IMGSZ

    YOLOUltralytics.preprocess = preprocess_in_bgr
    YOLOUltralytics.__init__ = init_at_1280


def patch_tracker_camera_motion() -> None:
    from tracklab.wrappers.track.bpbreid_strong_sort_api import BPBReIDStrongSORT

    process = BPBReIDStrongSORT.process

    def process_with_camera_motion(self, batch, detections, metadatas):
        if self.cfg.ecc and len(detections) > 0:
            image = tracklab_cv2.cv2_load_image(metadatas["file_path"].values[0])
            if self.prev_frame is not None:
                matrix = self.model.tracker.camera_update(self.prev_frame, image)
                COUNTS["camera_updates" if matrix is not None else "camera_update_failures"] += 1
            self.prev_frame = image
        return process(self, batch, detections, metadatas)

    BPBReIDStrongSORT.process = process_with_camera_motion


def patch_pitch_keypoints() -> None:
    from sn_gamestate.calibration.nbjw_calib import NBJW_Calib_Keypoints, kp_to_line

    process = NBJW_Calib_Keypoints.process

    def process_with_centred_keypoints(self, batch, detections, metadatas):
        found, images = process(self, batch, detections, metadatas)
        width, height = batch.size()[-1], batch.size()[-2]
        for name in images.index:
            keypoints = images.at[name, "keypoints"]
            for point in keypoints.values():
                point["x"] = point["x"] + 1.0 / width
                point["y"] = point["y"] + 1.0 / height
            images.at[name, "lines"] = kp_to_line(keypoints)
            COUNTS["keypoint_frames_centred"] += 1
        return found, images

    NBJW_Calib_Keypoints.process = process_with_centred_keypoints


def report() -> None:
    print("LEADCFG_PATCHES " + json.dumps(COUNTS), flush=True)


if __name__ == "__main__":
    seed()
    patch_detector()
    patch_tracker_camera_motion()
    patch_pitch_keypoints()
    atexit.register(report)
    run_tracklab.main()
