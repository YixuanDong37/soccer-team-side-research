"""ORACLE / DIAGNOSIS ONLY. Is across-pitch (y) displacement smaller than along-pitch (x)
for true partners? Does a collective shift (mean displacement of people tracked through the
gap) shrink the residual?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402

pd.set_option("display.width", 250)
p = pd.read_pickle("oracle_pairs.pkl")
p["dx"] = (p.b_start_x - p.a_end_x)
p["dy"] = (p.b_start_y - p.a_end_y)
s = p[p.same].sort_values(["seq", "b_piece", "a_last"])
imm = s.groupby(["seq", "b_piece"]).tail(1)
false = p[p.known & ~p.same & (p.a_team == p.b_team) & (p.a_role == p.b_role)]
for lo, hi in ((0, 25), (25, 75), (75, 150), (150, 300), (300, 800)):
    q = imm[(imm.gap >= lo) & (imm.gap < hi)]
    f = false[(false.gap >= lo) & (false.gap < hi)]
    print(f"gap [{lo},{hi}): true n={len(q)} |dx| med={q.dx.abs().median():.1f} p80={q.dx.abs().quantile(.8):.1f}  |dy| med={q.dy.abs().median():.1f} p80={q.dy.abs().quantile(.8):.1f}"
          f" || false n={len(f)} |dx| med={f.dx.abs().median():.1f} p20={f.dx.abs().quantile(.2):.1f}  |dy| med={f.dy.abs().median():.1f} p20={f.dy.abs().quantile(.2):.1f}")

# collective shift: per clip, per-frame mean position of people (players) tracked at both frames
print("\ncollective shift (mean displacement of pieces alive at both the end of a and the start of b)")
res = []
for seq in data.DEV_CLIPS:
    t = data.tracks(seq)
    t = t[(t.track_id >= 0) & t.pitch_usable & t.pitch_x.notna() & (t.role == "player")]
    pos = {}
    for f, g in t.groupby("frame"):
        pos[int(f)] = g.set_index("track_id")[["pitch_x", "pitch_y"]]
    q = imm[imm.seq == seq]
    for r in q.itertuples():
        A, B = pos.get(int(r.a_last)), pos.get(int(r.b_first))
        if A is None or B is None:
            res.append((r.gap, r.d, np.nan, 0)); continue
        common = A.index.intersection(B.index)
        common = [c for c in common if c not in (r.a_piece, r.b_piece)]
        if len(common) < 2:
            res.append((r.gap, r.d, np.nan, len(common))); continue
        sh = (B.loc[common].to_numpy() - A.loc[common].to_numpy()).mean(axis=0)
        resid = float(np.hypot(r.dx - sh[0], r.dy - sh[1]))
        res.append((r.gap, r.d, resid, len(common)))
res = pd.DataFrame(res, columns=["gap", "d", "resid", "n_common"])
for lo, hi in ((0, 25), (25, 75), (75, 150), (150, 300), (300, 800)):
    q = res[(res.gap >= lo) & (res.gap < hi)]
    k = q[q.resid.notna()]
    print(f"gap [{lo},{hi}): n={len(q)} with>=2 common={len(k)}  d med={k.d.median():.1f}  resid med={k.resid.median():.1f}  (d p80={k.d.quantile(.8):.1f} resid p80={k.resid.quantile(.8):.1f})")
