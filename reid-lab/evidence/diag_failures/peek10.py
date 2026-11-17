import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
df = pd.read_csv("returns.csv")
print(df[df.ann_speed_mps > 7][["clip","ret","ref_id","gap_s","exit_side","reentry_side","ann_exit_x","ann_exit_y","ann_reentry_x","ann_reentry_y","ann_dist_m","ann_speed_mps","status","out_plugin","out_mutual","anchors_cos_any","true_minus_best_impostor"]].to_string())
print(df[df.exit_side != df.reentry_side][["clip","ret","ref_id","gap_s","exit_side","reentry_side","ann_dist_m","ann_speed_mps","status","out_plugin","out_mutual","anchors_cos_any","true_minus_best_impostor"]].to_string())
