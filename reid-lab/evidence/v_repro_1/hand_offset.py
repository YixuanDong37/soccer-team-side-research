"""v_repro_1 / hand_offset.py -- sanity check of the frame alignment the scorer assumes (image N.jpg =
frame N - 1) and of the x1.5 scale: matched pairs and mean IoU when the annotation is shifted by
-2 .. +2 frames, and with scale 1.0 instead of 1.5. LABELS (check of the scorer only).
Imports neither reid_lab nor gbf (uses hand_check.py's reader).

    python -I -B hand_offset.py SNGS-162 SNGS-107 SNGS-151
"""
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/v_repro_1")
import hand_check as h  # noqa: E402


def run(seq, shift, scale):
    lab, W, H = h.read_labels(seq)
    t = pd.read_parquet(h.RUNS / seq / "gbf/refine/tracks.parquet")
    ref = defaultdict(list)
    for r in lab:
        ref[r[0] + shift].append(r[3:7])
    n, s = 0, 0.0
    for f, g in t.groupby("frame"):
        if f not in ref:
            continue
        iou = h.iou_matrix(np.array(ref[f]), scale * g[["x1", "y1", "x2", "y2"]].to_numpy(float))
        rr, cc = linear_sum_assignment(np.where(iou >= 0.5, iou, 0.0), maximize=True)
        ok = iou[rr, cc] >= 0.5
        n += int(ok.sum())
        s += float(iou[rr, cc][ok].sum())
    return n, (s / n if n else float("nan"))


for seq in sys.argv[1:]:
    print(seq)
    for shift in (-2, -1, 0, 1, 2):
        n, m = run(seq, shift, 1.5)
        print(f"  annotation shifted by {shift:+d} frames, scale 1.5: {n} matched pairs, mean IoU {m:.3f}")
    n, m = run(seq, 0, 1.0)
    print(f"  shift 0, scale 1.0: {n} matched pairs")
