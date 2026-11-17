"""Holes inside gbf pieces (frames without a row between two rows of one piece) of >= 10 frames
that gbf did NOT split (no box at the image edge next to the hole): is it the same annotated
person before and after? DIAGNOSIS ONLY (rule 5c). -> out/holes.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I holes.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
rows = []
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    for pid, idx in c.piece_rows.items():
        t = c.t.iloc[idx].sort_values("frame")
        f = t.frame.to_numpy()
        ridx = t.index.to_numpy()
        refs = c.ref_of_predrow[ridx]
        for i in np.where(np.diff(f) - 1 >= 10)[0]:
            before, after = refs[: i + 1], refs[i + 1:]
            cb, ca = Counter(before[before >= 0].tolist()), Counter(after[after >= 0].tolist())
            # the person right before / right after the hole: majority of the 10 nearest matched rows
            nb = Counter(before[before >= 0][-10:].tolist()).most_common(1)
            na = Counter(after[after >= 0][:10].tolist()).most_common(1)
            if not nb or not na:
                lab = "unknown (no matched row on one side)"
            else:
                lab = "same person" if nb[0][0] == na[0][0] else "different people"
            gb, ga = ridx[: i + 1][c.clean[ridx[: i + 1]]], ridx[i + 1:][c.clean[ridx[i + 1:]]]
            cs = None
            if len(gb) >= 5 and len(ga) >= 5:
                da, db = c.emb[gb].mean(0), c.emb[ga].mean(0)
                cs = float(da @ db / np.linalg.norm(da) / np.linalg.norm(db))
            a, b = t.iloc[i], t.iloc[i + 1]
            d = sp = None
            if a.pitch_usable and b.pitch_usable and pd.notna(a.pitch_x) and pd.notna(b.pitch_x):
                d = float(np.hypot(b.pitch_x - a.pitch_x, b.pitch_y - a.pitch_y))
                sp = d / ((b.frame - a.frame) / 25)
            rows.append({"clip": seq, "piece": pid, "hole_frames": int(f[i + 1] - f[i] - 1), "last_frame": int(f[i]), "next_frame": int(f[i + 1]),
                         "label": lab, "rows_before": i + 1, "rows_after": len(f) - i - 1, "clean_before": len(gb), "clean_after": len(ga),
                         "cos_before_after": None if cs is None else round(cs, 4),
                         "centre_jump_px": round(float(np.hypot((a.x1 + a.x2 - b.x1 - b.x2) / 2, (a.y1 + a.y2 - b.y1 - b.y2) / 2)), 1),
                         "pitch_jump_m": None if d is None else round(d, 1), "speed_mps": None if sp is None else round(sp, 2)})  # fmt: skip
df = pd.DataFrame(rows)
df.to_csv(OUT / "holes.csv", index=False)
pd.set_option("display.width", 250)
print(len(df), df.label.value_counts().to_dict())
print(df.groupby("label")[["hole_frames", "cos_before_after", "centre_jump_px", "speed_mps", "rows_after"]].describe().T.to_string())
k = df[df.label != "unknown (no matched row on one side)"]
for th in (0.80, 0.85, 0.90, 0.93):
    print("cos <", th, k[k.cos_before_after < th].label.value_counts().to_dict(), "| cos >=", th, k[k.cos_before_after >= th].label.value_counts().to_dict(), "| no cos", k[k.cos_before_after.isna()].label.value_counts().to_dict())
for h in (10, 25, 50, 100):
    print("hole >=", h, k[k.hole_frames >= h].label.value_counts().to_dict())
