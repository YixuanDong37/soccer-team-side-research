"""sha256 of the rgb24 bytes of a few canonical frames, decoded by gbf itself (run in gbf's
.venv). The tracklab wrapper compares its frames with these before a run.

Usage: python gbf_frame_hashes.py VIDEO DRIVELESS_PATH OUT.json [frame ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from gbf.ingest import iter_frames

video, driveless, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
wanted = sorted({int(v) for v in sys.argv[4:]} or {0, 1, 2, 833, 1664})
found: dict[str, str] = {}
for index, frame in iter_frames(video):
    if index in wanted:
        found[str(index)] = hashlib.sha256(frame.tobytes()).hexdigest()
    if index >= wanted[-1]:
        break
out.write_bytes(
    (json.dumps({"video": driveless, "frames": found}, indent=2) + "\n").encode("utf-8")
)
print(f"{len(found)} frame hashes written to {out}")
