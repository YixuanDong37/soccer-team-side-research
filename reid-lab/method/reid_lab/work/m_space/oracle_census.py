"""ORACLE / DIAGNOSIS ONLY. Does a team census (how many players of the team were ever seen at
once before, how many are in the picture now) tell first appearances from returns?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import geom  # noqa: E402
import labels  # noqa: E402

pd.set_option("display.width", 300)
rows = []
for seq in data.DEV_CLIPS:
    e = geom.piece_ends(seq)
    weak = labels.identity_map(seq, 1)
    strong = labels.identity_map(seq, 5)
    t = data.tracks(seq)
    t = t[(t.track_id >= 0) & (t.role == "player") & t.team.notna()]
    n = geom.cameras(seq)["n"]
    cnt = {}
    for team in ("left", "right"):
        c = np.zeros(n, int)
        g = t[t.team == team].groupby("frame").track_id.nunique()
        c[g.index.to_numpy()] = g.to_numpy()
        cnt[team] = c
    for b in e[(e["first"] > 0) & (e.role == "player") & e.team.notna()].itertuples():
        ib = strong.get(int(b.piece))
        if ib is None:
            continue
        earlier = e[(e["last"] < b.first) & (e.piece.map(weak) == ib)]
        c = cnt[b.team]
        seen = int(np.sort(c[: b.first])[-5:].min()) if b.first >= 5 else int(c[: b.first].max())  # 5th largest: robust max
        vis = int(c[b.first]) - 1
        rows.append(dict(seq=seq, b=b.piece, first=b.first, edge=b.start_side != "", ret=len(earlier) > 0, seen=seen, vis=vis))
df = pd.DataFrame(rows)
df["rho"] = ((10 - df.seen).clip(lower=0) / (10 - df.vis).clip(lower=1)).clip(upper=1)
for edge in (True, False):
    d = df[df.edge == edge]
    print(f"starts at border={edge}: n={len(d)} returns={int(d.ret.sum())} first appearances={int((~d.ret).sum())}")
    print(d.groupby("ret")[["first", "seen", "vis", "rho"]].describe(percentiles=[.5]).round(2).T.to_string())
    d = d.assign(rb=pd.cut(d.rho, [-0.01, 0.1, 0.25, 0.5, 0.75, 1.0]), fb=pd.cut(d["first"], [0, 50, 150, 300, 750]))
    print(pd.crosstab(d.rb, d.ret))
    print(pd.crosstab(d.fb, d.ret))
