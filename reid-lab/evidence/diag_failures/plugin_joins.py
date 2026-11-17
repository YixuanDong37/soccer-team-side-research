"""The plugin's own re-entry joins (consecutive gbf pieces of one plugin id; the links of
score.baseline_links("plugin")), judged with labels, with label-free signals next to them.
DIAGNOSIS ONLY (rule 5c). -> out/plugin_joins.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I plugin_joins.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FPS, H720, W720, cos, ctx, data, reach_ok, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
rows = []
base = score.baseline_links("plugin")
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    P, t = c.P, c.t
    for a_id, b_id in base[seq]["links"]:
        a, b = P[a_id], P[b_id]
        if a["ident"] is not None and b["ident"] is not None:
            label = "right" if a["ident"] == b["ident"] else "wrong"
        else:
            label = "unknown"
        # a softer label for the unknown ones: majority person with any number of matched rows
        if a["ident_any"] is None or b["ident_any"] is None:
            soft = "one piece matches nobody"
        else:
            soft = "same person" if a["ident_any"] == b["ident_any"] else "different people"
        ra, rb = t.iloc[c.piece_rows[a_id]], t.iloc[c.piece_rows[b_id]]
        la, fb = ra.sort_values("frame").iloc[-1], rb.sort_values("frame").iloc[0]
        d = sp = None
        if a["last_pos"] is not None and b["first_pos"] is not None:
            d = float(np.hypot(b["first_pos"][1] - a["last_pos"][1], b["first_pos"][2] - a["last_pos"][2]))
            el = (b["first_pos"][0] - a["last_pos"][0]) / FPS
            sp = d / el if el > 0 else None
        # appearance: rank of B among every later piece A could be (same team label or none, no referee)
        s_any = cos(a, b, "desc_any")
        better = n_pool = None
        if s_any is not None:
            pool = [float(a["desc_any"] @ q["desc_any"]) for q in P.values()
                    if q["piece"] not in (a_id, b_id) and q["desc_any"] is not None and q["role"] != "referee"
                    and q["first"] > a["last"] and (a["team"] is None or q["team"] is None or q["team"] == a["team"])]
            pool_b = [float(b["desc_any"] @ q["desc_any"]) for q in P.values()
                      if q["piece"] not in (a_id, b_id) and q["desc_any"] is not None and q["role"] != "referee"
                      and q["last"] < b["first"] and (b["team"] is None or q["team"] is None or q["team"] == b["team"])]
            better, n_pool = sum(v > s_any for v in pool) + sum(v > s_any for v in pool_b), len(pool) + len(pool_b)
        # image-side geometry at the gap
        cxa, cxb = (la.x1 + la.x2) / 2, (fb.x1 + fb.x2) / 2
        rows.append({
            "clip": seq, "earlier": a_id, "later": b_id, "plugin": a["plugin"], "label": label, "soft_label": soft,
            "gap_frames": b["first"] - a["last"] - 1, "gap_s": round((b["first"] - a["last"] - 1) / FPS, 2),
            "a_rows": a["rows"], "b_rows": b["rows"], "a_clean": a["n_clean"], "b_clean": b["n_clean"],
            "a_matched": a["matched"], "b_matched": b["matched"],
            "a_person": a["ident_any"], "a_person_rows": a["ident_n"], "b_person": b["ident_any"], "b_person_rows": b["ident_n"],
            "cos_strict": None if cos(a, b) is None else round(cos(a, b), 4),
            "cos_any": None if s_any is None else round(s_any, 4), "desc_kinds": f"{a['desc_kind']}/{b['desc_kind']}",
            "pieces_scoring_higher": better, "pool": n_pool,
            "a_end_side": a["end_side"], "b_start_side": b["start_side"],
            "same_side": a["end_side"] == b["start_side"] and a["end_side"] != "none",
            "a_last_cx": round(float(cxa)), "b_first_cx": round(float(cxb)),
            "a_last_h": round(float(la.y2 - la.y1)), "b_first_h": round(float(fb.y2 - fb.y1)),
            "h_ratio": round(float((fb.y2 - fb.y1) / max(la.y2 - la.y1, 1e-6)), 2),
            "team_a": a["team"], "team_b": b["team"], "team_equal": a["team"] == b["team"],
            "role_a": a["role"], "role_b": b["role"],
            "dist_m": None if d is None else round(d, 1), "speed_mps": None if sp is None else round(sp, 2),
            "reach_ok": reach_ok(a, b),
            "b_last_conf_mean": round(float(rb.conf.mean()), 2),
        })  # fmt: skip
df = pd.DataFrame(rows)
df.to_csv(OUT / "plugin_joins.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
print(df.label.value_counts().to_dict()); print(pd.crosstab(df.label, df.soft_label))
num = ["gap_s", "a_rows", "b_rows", "a_clean", "b_clean", "cos_any", "cos_strict", "pieces_scoring_higher", "h_ratio", "dist_m", "speed_mps"]
print(df.groupby("label")[num].describe().T.to_string())
print(pd.crosstab(df.label, df.same_side)); print(pd.crosstab(df.label, [df.a_end_side, df.b_start_side]).T.to_string())
print(pd.crosstab(df.label, df.team_equal)); print(pd.crosstab(df.label, df.reach_ok)); print(pd.crosstab(df.label, df.desc_kinds))
cols = ["clip", "earlier", "later", "label", "soft_label", "gap_s", "a_rows", "b_rows", "a_clean", "b_clean", "a_person", "b_person", "a_person_rows", "b_person_rows", "cos_any", "desc_kinds", "pieces_scoring_higher", "pool", "a_end_side", "b_start_side", "h_ratio", "speed_mps", "team_equal"]
print(df[df.label == "wrong"][cols].to_string())
print(df[df.label == "right"][cols].to_string())
print(df[df.label == "unknown"][cols].to_string())
