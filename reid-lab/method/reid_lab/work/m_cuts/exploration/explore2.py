"""DIAGNOSIS (labels via diag_switches.json): internal gaps of pieces vs true switches."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
SW = json.load(open("diag_switches.json"))
rows = []
for seq in data.DEV_CLIPS:
    t = data.tracks(seq)
    for pid, p in t[t.track_id >= 0].sort_values("frame").groupby("track_id"):
        f = p.frame.to_numpy()
        d = np.diff(f) - 1
        sws = [s for s in SW[seq] if s["piece"] == pid]
        for i in np.where(d >= 1)[0]:
            lab = any(s["last_a"] <= f[i] and f[i + 1] <= s["first_b"] for s in sws)
            rows.append({"seq": seq, "piece": pid, "f0": int(f[i]), "f1": int(f[i + 1]), "gap": int(d[i]), "switch": lab})
df = pd.DataFrame(rows)
bins = [1, 2, 3, 5, 10, 15, 25, 50, 100, 1000]
df["bin"] = pd.cut(df.gap, bins, right=False)
print(df.groupby("bin", observed=False).switch.agg(["size", "sum", "mean"]))
print(df[df.gap >= 10].groupby("seq").switch.agg(["size", "sum", "mean"]))
df.to_csv("diag_gaps.csv", index=False)
