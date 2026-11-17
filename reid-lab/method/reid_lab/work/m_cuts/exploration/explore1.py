"""DIAGNOSIS (labels via diag_switches.json): what do true switches look like in the track table?"""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
SW = json.load(open("diag_switches.json"))
rows = []
for seq in data.DEV_CLIPS:
    t = data.tracks(seq)
    # team / role constancy within a piece
    g = t[t.track_id >= 0].groupby("track_id")
    nteam = g.team.nunique(); nrole = g.role.nunique()
    dup = t[t.track_id >= 0].groupby(["track_id", "frame"]).size().max()
    print(seq, "pieces with >1 team:", int((nteam > 1).sum()), ">1 role:", int((nrole > 1).sum()), "max rows per (piece,frame):", dup,
          "team NaN rows:", int(t.team.isna().sum()), "jersey rows:", int(t.jersey.notna().sum()))
    for s in SW[seq]:
        p = t[t.track_id == s["piece"]].sort_values("frame")
        f = p.frame.to_numpy()
        seg = f[(f >= s["last_a"]) & (f <= s["first_b"])]
        d = np.diff(seg)
        maxgap = int(d.max() - 1) if len(d) else -1
        rows.append({"seq": seq, **s, "piece_first": int(f[0]), "piece_last": int(f[-1]), "max_gap_between": maxgap,
                     "n_rows_between": len(seg) - 2})
df = pd.DataFrame(rows)
print(df.to_string())
print(df.max_gap_between.describe())
print("gap>=2:", (df.max_gap_between >= 2).sum(), "gap>=5:", (df.max_gap_between >= 5).sum(), "gap>=10:", (df.max_gap_between >= 10).sum(), "gap==0:", (df.max_gap_between == 0).sum())
