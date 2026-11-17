"""DIAGNOSIS (uses labels through score): how do keep links and the mutual baseline combine?"""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import numpy as np, pandas as pd
from reid_lab import data, score
import features, labels_for_fit as L

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 300)
mut = score.baseline_links("mutual")
t = L.table()
for thr in (0.80, 0.85, 0.90, 0.92):
    keep = {s: {"links": t[(t.seq == s) & (t.cos >= thr)][["a", "b"]].values.tolist()} for s in data.DEV_CLIPS}
    print("keep cos>=", thr, score.evaluate(keep)["pooled"])
    # union with mutual: drop mutual links that conflict with a keep link (same earlier or same later)
    comb = {}
    for s in data.DEV_CLIPS:
        kl = keep[s]["links"]
        used_a = {a for a, b in kl}; used_b = {b for a, b in kl}
        ml = [l for l in mut[s]["links"] if l[0] not in used_a and l[1] not in used_b and l not in kl]
        comb[s] = {"links": kl + ml}
    print("  + mutual:", score.evaluate(comb)["pooled"])
# what are the mutual links: same plugin id or not
rows = []
for s in data.DEV_CLIPS:
    P = features.piece_table(s)
    for a, b in mut[s]["links"]:
        r = score.score_clip(s, [[a, b]])
        base = score.score_clip(s, [])
        lab = "right" if r["links"]["right"] else ("wrong" if r["links"]["wrong"] else "unknown")
        dr = sum(1 for x, y in zip(base["per_return"], r["per_return"]) if y == "right" and x != "right")
        dw = sum(1 for x, y in zip(base["per_return"], r["per_return"]) if y == "wrong" and x != "wrong")
        f = features.describe_pair(s, a, b)
        rows.append({"seq": s, "a": a, "b": b, "same_plugin": P[a]["plugin"] == P[b]["plugin"], "link": lab, "dr": dr, "dw": dw,
                     **{k: f[k] for k in ("gap_s", "rows_a", "rows_b", "cos", "imp_pct", "margin_a", "margin_b", "rank_a", "rank_b", "dist_m", "speed_req", "side_a", "side_b")}})
m = pd.DataFrame(rows)
print(m.round(3).to_string())
print(m.groupby(["same_plugin", "link"])[["dr", "dw"]].agg(["sum", "count"]))
