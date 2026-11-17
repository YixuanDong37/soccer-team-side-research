"""Export the tracker's per-detection re-id embeddings so gbf pieces can be matched by appearance.

Run in the sn-gamestate venv: python -I export_embeddings.py STATE.pklz OUT.npz
OUT.npz: frame (int32), track (int64, -1 without a track), box (float64, n x 4: the same
[x1, y1, x2, y2] as detections.jsonl, from export_sngs.ltrb), emb (float32, n x 256: PRTReid's
embedding as stored, not normalised). OUT.json next to it records the source state's sha256
and the restriction tags: the embeddings come from PRTReid (hl3) on YOLO detections
(agpl_network_source) of a private clip (no_share).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "C:/Research/plugins/tools")
from export_sngs import load_state, ltrb  # noqa: E402

state, out = Path(sys.argv[1]), Path(sys.argv[2])
detections, _ = load_state(state)
frames, tracks, boxes, embs = [], [], [], []
for det in detections.itertuples(index=False):
    box = ltrb(det.bbox_ltwh)
    if box is None:
        continue
    track = det.track_id
    frames.append(int(det.image_id))
    tracks.append(int(track) if track == track else -1)  # NaN != NaN
    boxes.append(box)
    embs.append(np.asarray(det.embeddings, dtype=np.float32).reshape(-1))
np.savez_compressed(out, frame=np.array(frames, np.int32), track=np.array(tracks, np.int64),
                    box=np.array(boxes, np.float64), emb=np.vstack(embs))  # fmt: skip
digest = hashlib.sha256(state.read_bytes()).hexdigest()
out.with_suffix(".json").write_text(json.dumps({
    "source_state": str(state), "source_sha256": digest, "rows": len(frames),
    "dim": int(embs[0].shape[0]), "model": "PRTReid (prtreid-soccernet-baseline) via sn-gamestate",
    "restrictions": ["no_share", "hl3", "agpl_network_source"],
}, indent=2) + "\n", encoding="utf-8")  # fmt: skip
print(len(frames), "rows written to", out)
