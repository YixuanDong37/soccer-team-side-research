"""v_repro / hand_offset_check.py -- sanity check of the frame convention (image N.jpg = frame N - 1) and of
the x1.5 scale used by the scorer and by hand_check.py (LABELS; diagnosis only): matched pairs and their median
IoU when the annotation is shifted by -1 / 0 / +1 frames, and with scale 1.0 instead of 1.5.
    python -I -B hand_offset_check.py SNGS-162 SNGS-107
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hand_check as hc  # noqa: E402

for seq in sys.argv[1:]:
    people, W, H = hc.read_labels(seq)
    tracks = pd.read_parquet(hc.RUNS / seq / "gbf/refine/tracks.parquet").reset_index(drop=True)
    print(seq, "annotation", W, "x", H, "frames", int(people.frame.min()), "-", int(people.frame.max()),
          "| tracks frames", int(tracks.frame.min()), "-", int(tracks.frame.max()), "max x2", round(float(tracks.x2.max()), 1), "max y2", round(float(tracks.y2.max()), 1))
    for shift in (-1, 0, 1):
        m = hc.match_clip(people.assign(frame=people.frame + shift), tracks)
        print(f"  annotation shifted by {shift:+d}: matched pairs {len(m)}, median IoU {m.iou.median():.3f}, mean IoU {m.iou.mean():.3f}")
    hc.SCALE = 1.0
    m = hc.match_clip(people, tracks)
    print(f"  scale 1.0 instead of 1.5: matched pairs {len(m)}")
    hc.SCALE = 1.5
