import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data
import pandas as pd, numpy as np
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
seq = "SNGS-107"
t = data.tracks(seq)
for pid in (1, 37, 4, 100078):
    g = t[t.track_id == pid].sort_values("frame")
    print(pid, len(g))
    cols = ["frame","x1","y1","x2","y2","conf","team","pitch_usable","pitch_x","pitch_y","in_pitch","calib_status"]
    print(g[cols].tail(12).to_string()); print(g[cols].head(12).to_string())
# stats: usable frames, calib_status by clip
for s in data.DEV_CLIPS:
    c = data.calib(s); tt = data.tracks(s)
    print(s, "frames", len(c), "usable", int(c.pitch_usable.sum()), dict(c.calib_status.value_counts()), "rows", len(tt), "untracked", int((tt.track_id<0).sum()), "pitch nan", int(tt.pitch_x.isna().sum()), "not in_pitch", int((~tt.in_pitch).sum()), "roles", dict(tt.role.value_counts()), "team na", int(tt.team.isna().sum()))
