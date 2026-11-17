import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
df = pd.read_csv("returns.csv")
d = df[(df["clip"] == "SNGS-162")][["ret","ref_id","ann_jersey","ann_team","exit_frame","reentry_frame","exit_side","reentry_side","ann_exit_x","ann_exit_y","ann_reentry_x","ann_reentry_y","ann_dist_m","ann_speed_mps","status","anchor_a","anchor_b","anchors_cos_any","true_minus_best_impostor","out_plugin","out_mutual","label_suspect_exchange"]]
print(d.to_string())
def chk(i, j):
    a, b = d[d.ret == i].iloc[0], d[d.ret == j].iloc[0]
    for (p, q, name) in ((a, a, "annotated i"), (b, b, "annotated j"), (a, b, "exchanged i->j"), (b, a, "exchanged j->i")):
        dist = np.hypot(q.ann_reentry_x - p.ann_exit_x, q.ann_reentry_y - p.ann_exit_y); dt = (q.reentry_frame - p.exit_frame) / 25
        print(f"  {name}: exit person {p.ref_id} ({p.ann_exit_x},{p.ann_exit_y}) @ {p.exit_frame} -> re-entry of {q.ref_id} ({q.ann_reentry_x},{q.ann_reentry_y}) @ {q.reentry_frame}: {dist:.1f} m in {dt:.1f} s = {dist/dt:.1f} m/s")
print("9 vs 18"); chk(6, 15)
print("12 vs 14"); chk(10, 12)
