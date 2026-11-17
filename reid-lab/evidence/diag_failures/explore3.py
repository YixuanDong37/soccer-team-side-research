import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 200)
df = pd.read_csv("out/plugin_joins.csv")
print(pd.crosstab(df.label, df.same_side)); print(pd.crosstab(df.label, df.team_equal)); print(pd.crosstab(df.label, df.reach_ok)); print(pd.crosstab(df.label, df.desc_kinds))
print(pd.crosstab(df.label, [df.a_end_side != "none", df.b_start_side != "none"]))
for col in ("cos_any", "gap_s", "pieces_scoring_higher", "h_ratio", "speed_mps", "b_rows", "a_rows"):
    print(col, {k: g[col].round(3).tolist() for k, g in df[df.label != "unknown"].groupby("label")})
cols = ["clip", "earlier", "later", "soft_label", "gap_s", "a_rows", "b_rows", "a_clean", "b_clean", "a_person", "b_person", "cos_any", "desc_kinds", "pieces_scoring_higher", "pool", "a_end_side", "b_start_side", "h_ratio", "speed_mps"]
print(df[df.label == "wrong"][cols].to_string())
u = df[df.label == "unknown"]
print(u.groupby("soft_label")[["a_rows", "b_rows", "cos_any", "gap_s"]].describe().T.to_string())
print("unknown: min(a_rows,b_rows) <= 4:", (u[["a_rows","b_rows"]].min(axis=1) <= 4).sum(), "of", len(u))
# simple in-sample separators (diagnosis)
k = df[df.label != "unknown"]
for th in (0.90, 0.91, 0.92, 0.93, 0.94, 0.95):
    kk = k[k.cos_any >= th]; print("cos_any >=", th, kk.label.value_counts().to_dict())
for n in (0, 1, 2, 3):
    kk = k[k.pieces_scoring_higher <= n]; print("pieces scoring higher <=", n, kk.label.value_counts().to_dict())
kk = k[(k.pieces_scoring_higher == 0) & (k.cos_any >= 0.92)]; print("both:", kk.label.value_counts().to_dict())
kk = df[(df.pieces_scoring_higher == 0) & (df.cos_any >= 0.92)]; print("both, all 93:", kk.label.value_counts().to_dict(), kk.soft_label.value_counts().to_dict())
