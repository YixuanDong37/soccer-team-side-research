"""ORACLE / DIAGNOSIS ONLY. Which links does the position linker get wrong, by kind of pair?
python -I oracle_links.py [theta] [mode] [peel]"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import labels  # noqa: E402
import linker  # noqa: E402

pd.set_option("display.width", 280)
pd.set_option("display.max_rows", 500)
theta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
mode = sys.argv[2] if len(sys.argv) > 2 else "sinkhorn"
peel = (sys.argv[3] == "1") if len(sys.argv) > 3 else True
kw = dict(resid="r", pair=("lu", "ls", "iv", "tu"), ctx=("one", "edge", "time"))
lp = linker.loco_params(**kw)
rows = []
for seq in data.DEV_CLIPS:
    ident = labels.identity_map(seq, 5)
    q = linker.posteriors(features.pair_features(seq), lp[seq], mode).set_index(["a", "b"])
    for a, b in linker.links(features.pair_features(seq), lp[seq], theta, mode, peel):
        r = q.loc[(a, b)]
        ia, ib = ident.get(a), ident.get(b)
        verdict = "unknown" if ia is None or ib is None else ("right" if ia == ib else "wrong")
        rows.append(dict(seq=seq, a=a, b=b, verdict=verdict, gap=int(r.gap), r=round(float(r.r), 1), inview=round(float(r.inview_s), 2),
                         a_edge=bool(r.a_end_edge), b_edge=bool(r.b_start_edge), a_rows=int(r.a_rows), b_rows=int(r.b_rows),
                         score0=round(float(r.score), 3), p_in=round(float(r.p_in), 3), p_out=round(float(r.p_out), 3), team=int(r.same_team), plugin=int(r.same_plugin)))
df = pd.DataFrame(rows)
df["kind"] = np.where(df.a_edge & df.b_edge, "edge-edge", np.where(df.a_edge | df.b_edge, "one-edge", "in-view"))
df["gapbin"] = pd.cut(df.gap, [-1, 9, 24, 74, 149, 299, 800], labels=["0-9", "10-24", "25-74", "75-149", "150-299", "300+"])
print(pd.crosstab([df.kind, df.gapbin], df.verdict))
print(pd.crosstab(df.kind, df.verdict))
print("small pieces (min rows < 5):")
print(pd.crosstab(np.minimum(df.a_rows, df.b_rows) < 5, df.verdict))
print(df[df.verdict == "wrong"].to_string())
