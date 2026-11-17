"""Returns whose row A or row B sits in a piece that is not the person's own piece (a fragment
with < 5 matched rows, or a piece that is mainly somebody else): what those pieces look like and
how they relate to the person's own piece on that side. DIAGNOSIS ONLY (rule 5c).
-> out/fragments.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I fragments.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, gsr  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
ret = pd.read_csv(OUT / "returns.csv")
sc = ret[ret.status == "scored"]
rows = []
for r in sc.itertuples():
    c = ctx(r.clip)
    t = c.t
    for tag in ("a", "b"):
        k = getattr(r, f"{tag}_kind")
        if k == "own":
            continue
        pid = int(getattr(r, f"piece_{tag}"))
        p = c.P[pid]
        anchor = getattr(r, f"anchor_{tag}")
        row = {"clip": r.clip, "ret": r.ret, "ref_id": r.ref_id, "side": tag.upper(), "kind": k, "piece": pid, "plugin": p["plugin"], "k": p["k"],
               "rows": p["rows"], "clean": p["n_clean"], "matched": p["matched"], "person_rows": int(getattr(r, f"{tag}_person_rows")),
               "majority_person": p["ident_any"], "majority_rows": p["ident_n"], "first": p["first"], "last": p["last"],
               "starts_at_edge": p["starts_at_edge"], "ends_at_edge": p["ends_at_edge"], "team": p["team"], "role": p["role"],
               "ceiling_class": r.ceiling_class, "own_piece": None if pd.isna(anchor) else int(anchor)}  # fmt: skip
        if not pd.isna(anchor) and int(anchor) != pid:
            q = c.P[int(anchor)]
            row["own_same_plugin"] = q["plugin"] == p["plugin"]
            fr = t.iloc[c.piece_rows[pid]].sort_values("frame")
            qr = t.iloc[c.piece_rows[int(anchor)]].sort_values("frame")
            if tag == "b":   # fragment first, then the own piece
                e, s = fr.iloc[-1], qr[qr.frame > fr.frame.max()].head(1)
                row["frames_between"] = int(q["first"] - p["last"] - 1)
            else:            # own piece first, then the fragment
                e, s = qr[qr.frame < fr.frame.min()].tail(1), fr.iloc[0]
                row["frames_between"] = int(p["first"] - q["last"] - 1)
            if len(pd.DataFrame(s).T if isinstance(s, pd.Series) else s) and len(pd.DataFrame(e).T if isinstance(e, pd.Series) else e):
                e = e.iloc[0] if isinstance(e, pd.DataFrame) else e
                s = s.iloc[0] if isinstance(s, pd.DataFrame) else s
                row["box_iou_across"] = round(float(gsr.box_iou(np.array([[e.x1, e.y1, e.x2, e.y2]]), np.array([[s.x1, s.y1, s.x2, s.y2]]))[0, 0]), 2)
                row["centre_shift_px"] = round(float(np.hypot((e.x1 + e.x2 - s.x1 - s.x2) / 2, (e.y1 + e.y2 - s.y1 - s.y2) / 2)), 1)
                row["frames_between_boxes"] = int(abs(s.frame - e.frame) - 1)
            row["overlap_in_time"] = bool(q["first"] <= p["last"] and p["first"] <= q["last"])
            if p["desc_any"] is not None and q["desc_any"] is not None:
                row["cos_any_to_own"] = round(float(p["desc_any"] @ q["desc_any"]), 4)
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "fragments.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print(df.to_string())
print(pd.crosstab([df.side, df.kind], df.ceiling_class).to_string())
fr = df[df.kind == "fragment"]
print("fragments:", len(fr), "rows<=4:", (fr.rows <= 4).sum(), "clean==0:", (fr.clean == 0).sum(), "k>0 (a re-entry piece of gbf's split):", (fr.k > 0).sum())
print("own piece on that side exists:", fr.own_piece.notna().sum(), "same plugin id:", fr.own_same_plugin.sum())
print(fr[["frames_between", "box_iou_across", "centre_shift_px", "cos_any_to_own"]].describe().to_string())
print("frames_between <= 5:", (fr.frames_between.abs() <= 5).sum(), "; <= 12:", (fr.frames_between.abs() <= 12).sum(), "; overlap in time:", fr.overlap_in_time.sum())
