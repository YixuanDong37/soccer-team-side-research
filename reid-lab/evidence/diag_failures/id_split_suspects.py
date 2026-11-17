"""Pairs of annotated ids of one clip that are never annotated on the same frame, have the same
team and role and no contradicting jersey, where the first id ends before the clip ends and the
second starts after the clip starts: they COULD be one real player annotated under two ids (a
return the annotation does not count). Listed with the appearance cosine between their own
pieces. No image was viewed: suspects for a human check only.
DIAGNOSIS ONLY (rule 5c). -> out/id_split_suspects.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I id_split_suspects.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ctx, data, sides_1080  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
rows = []
n_ids = 0
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    ids = [rid for rid, a in c.att.items() if a["role"] in ("player", "goalkeeper") and rid in c.people_by]
    n_ids += len(ids)
    last_frame = int(c.people.frame.max())
    own = {rid: sorted([p for p in c.P.values() if p["ident"] == rid], key=lambda p: p["first"]) for rid in ids}
    for i in ids:
        gi = c.people_by[i]
        if gi.frame.max() >= last_frame - 5:
            continue
        for j in ids:
            gj = c.people_by[j]
            if i == j or gj.frame.min() <= gi.frame.max() or gj.frame.min() < 5:
                continue
            ai, aj = c.att[i], c.att[j]
            if ai["team"] != aj["team"] or ai["role"] != aj["role"]:
                continue
            if ai["jersey"] is not None and aj["jersey"] is not None and ai["jersey"] != aj["jersey"]:
                continue
            a = [p for p in own[i] if p["desc_any"] is not None]
            b = [p for p in own[j] if p["desc_any"] is not None]
            if not a or not b:
                continue
            a, b = a[-1], b[0]
            s = float(a["desc_any"] @ b["desc_any"])
            # is the pair mutual best among own pieces of all same-team ids on the two sides?
            later = [float(a["desc_any"] @ q["desc_any"]) for k in ids for q in own[k] if k not in (i, j) and c.att[k]["team"] == ai["team"] and q["desc_any"] is not None and q["first"] > a["last"]]
            earlier = [float(b["desc_any"] @ q["desc_any"]) for k in ids for q in own[k] if k not in (i, j) and c.att[k]["team"] == ai["team"] and q["desc_any"] is not None and q["last"] < b["first"]]
            e, s2 = gi.iloc[-1], gj.iloc[0]
            gap = int(gj.frame.min() - gi.frame.max() - 1)
            d = float(np.hypot(s2.pitch_x - e.pitch_x, s2.pitch_y - e.pitch_y))
            rows.append({"clip": seq, "id_first": i, "id_second": j, "team": ai["team"], "role": ai["role"], "jersey_first": ai["jersey"], "jersey_second": aj["jersey"],
                         "first_ends": int(gi.frame.max()), "second_starts": int(gj.frame.min()), "gap_s": round(gap / 25, 2),
                         "end_side": sides_1080(e, c.ref), "start_side": sides_1080(s2, c.ref), "dist_m": round(d, 1), "speed_mps": round(d / max(gap / 25, 0.04), 2),
                         "piece_first": a["piece"], "piece_second": b["piece"], "cos": round(s, 4),
                         "best_other_later": round(max(later), 4) if later else None, "best_other_earlier": round(max(earlier), 4) if earlier else None,
                         "margin": round(s - max(later + earlier + [-1.0]), 4)})  # fmt: skip
df = pd.DataFrame(rows)
df.to_csv(OUT / "id_split_suspects.csv", index=False)
pd.set_option("display.width", 320); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print("annotated player/goalkeeper ids:", n_ids, "; possible pairs:", len(df))
strong = df[(df.cos >= 0.93) & (df.margin >= 0.03) & (df.end_side != "none") & (df.start_side != "none")]
print("strong suspects (cos >= 0.93, beats every other same-team person by >= 0.03, both at an image edge):", len(strong))
print(strong.to_string())
print(df[~df.index.isin(strong.index)].sort_values("margin", ascending=False).head(12).to_string())
