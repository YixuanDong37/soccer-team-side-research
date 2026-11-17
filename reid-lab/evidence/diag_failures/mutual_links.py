"""The mutual rule (score.baseline_links("mutual")): every link it makes, judged with labels, and
for the wrong ones where the true partner was. DIAGNOSIS ONLY (rule 5c). -> out/mutual_links.csv

Run: C:/Research/soccer-team-side-research/gbf/.venv/Scripts/python.exe -I mutual_links.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FPS, ctx, data, eligible_why, mutual_status, pair_gate, score  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
rows = []
base = score.baseline_links("mutual")
for seq in data.DEV_CLIPS:
    c = ctx(seq)
    P = c.P
    for a_id, b_id in base[seq]["links"]:
        a, b = P[a_id], P[b_id]
        ms = mutual_status(c.cand, a_id, b_id)
        if a["ident"] is not None and b["ident"] is not None:
            label = "right" if a["ident"] == b["ident"] else "wrong"
        else:
            label = "unknown"
        d = None
        if a["last_pos"] is not None and b["first_pos"] is not None:
            d = float(np.hypot(b["first_pos"][1] - a["last_pos"][1], b["first_pos"][2] - a["last_pos"][2]))
        row = {
            "clip": seq, "earlier": a_id, "later": b_id, "label": label, "cos": round(c.cand[(a_id, b_id)], 4),
            "margin_a_side": round(ms["margin_a_side"], 4), "margin_b_side": round(ms["margin_b_side"], 4),
            "min_margin": round(min(ms["margin_a_side"], ms["margin_b_side"]), 4),
            "n_later_of_a": ms["n_later_of_a"], "n_earlier_of_b": ms["n_earlier_of_b"],
            "gap_s": round((b["first"] - a["last"]) / FPS, 2), "same_plugin": a["plugin"] == b["plugin"],
            "a_rows": a["rows"], "b_rows": b["rows"], "a_clean": a["n_clean"], "b_clean": b["n_clean"],
            "a_person": a["ident_any"], "a_person_rows": a["ident_n"], "a_purity": a["purity"],
            "b_person": b["ident_any"], "b_person_rows": b["ident_n"], "b_purity": b["purity"],
            "a_ann_team": c.att.get(a["ident_any"], {}).get("team"), "b_ann_team": c.att.get(b["ident_any"], {}).get("team"),
            "a_ann_role": c.att.get(a["ident_any"], {}).get("role"), "b_ann_role": c.att.get(b["ident_any"], {}).get("role"),
            "a_ann_jersey": c.att.get(a["ident_any"], {}).get("jersey"), "b_ann_jersey": c.att.get(b["ident_any"], {}).get("jersey"),
            "team": a["vteam"], "a_end_side": a["end_side"], "b_start_side": b["start_side"],
            "dist_m": None if d is None else round(d, 1),
            "speed_mps": None if d is None or b["first_pos"][0] <= a["last_pos"][0] else round(d / ((b["first_pos"][0] - a["last_pos"][0]) / FPS), 2),
        }  # fmt: skip
        if label != "right":
            # where is the true partner of each end?
            for tag, me, later in (("a", a, True), ("b", b, False)):
                rid = me["ident_any"]
                if rid is None:
                    row[f"{tag}_true_partner"] = "piece matches nobody annotated"
                    continue
                own = [q for q in P.values() if q["piece"] != me["piece"] and q["ident"] == rid
                       and ((q["first"] > me["last"]) if later else (q["last"] < me["first"]))]
                if not own:
                    seen = c.people_by[rid].frame
                    more = int((seen > me["last"]).sum()) if later else int((seen < me["first"]).sum())
                    row[f"{tag}_true_partner"] = ("no own piece " + ("after" if later else "before") + f" it (person annotated on {more} frames there)")
                    continue
                inc = [q for q in own if ((me["piece"], q["piece"]) if later else (q["piece"], me["piece"])) in c.cand]
                if inc:
                    best = max(inc, key=lambda q: c.cand[(me["piece"], q["piece"]) if later else (q["piece"], me["piece"])])
                    s = c.cand[(me["piece"], best["piece"]) if later else (best["piece"], me["piece"])]
                    row[f"{tag}_true_partner"] = f"candidate {best['piece']} cos {s:.4f} (lost by {row['cos'] - s:.4f})"
                else:
                    near = min(own, key=lambda q: abs(q["first"] - me["last"]) if later else abs(me["first"] - q["last"]))
                    gate = pair_gate(me, near) if later else pair_gate(near, me)
                    row[f"{tag}_true_partner"] = f"own piece {near['piece']} exists but is no candidate: {'; '.join(gate)}"
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "mutual_links.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 90)
print(df.label.value_counts().to_dict())
print(df.groupby("label")[["cos", "min_margin", "gap_s", "a_clean", "b_clean", "n_later_of_a", "n_earlier_of_b", "speed_mps"]].describe().T.to_string())
bad = df[df.label != "right"]
print(bad[["clip", "earlier", "later", "label", "cos", "min_margin", "gap_s", "same_plugin", "a_clean", "b_clean", "a_person", "a_person_rows", "a_purity", "b_person", "b_person_rows", "b_purity", "a_ann_team", "b_ann_team", "a_ann_role", "b_ann_role", "a_end_side", "b_start_side", "speed_mps"]].to_string())
for r in bad.itertuples():
    print(r.clip, r.earlier, r.later, "| A:", r.a_true_partner, "| B:", r.b_true_partner)
print("right links same_plugin:", df[df.label == "right"].same_plugin.value_counts().to_dict())
# in-sample (diagnosis only): what a higher margin or an absolute bar would do to these 69 links
for m in (0.05, 0.06, 0.08, 0.10, 0.12, 0.15):
    k = df[df.min_margin >= m]
    print("margin >=", m, k.label.value_counts().to_dict())
for t in (0.90, 0.93, 0.95, 0.96, 0.97):
    k = df[df.cos >= t]
    print("cos >=", t, k.label.value_counts().to_dict())
print("sides right:", pd.crosstab(df.label, [df.a_end_side != "none", df.b_start_side != "none"]).to_string())
