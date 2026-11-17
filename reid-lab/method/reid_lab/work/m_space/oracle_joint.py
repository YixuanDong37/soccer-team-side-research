"""ORACLE / DIAGNOSIS ONLY. Are the residuals of true partners that left and returned at about
the same time correlated (a common offset the visible players' flow does not explain)?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import fitdata  # noqa: E402

pd.set_option("display.width", 250)
p = fitdata.pooled(data.DEV_CLIPS)
T = p[p.imm & (p.gap >= 50)].reset_index(drop=True)
print("true pairs with gap >= 50:", len(T))
for win in (25, 50, 100):
    rows = []
    for i, r in T.iterrows():
        nb = T[(T.seq == r.seq) & (T.index != i) & ((T.a_last - r.a_last).abs() <= win) & ((T.b_first - r.b_first).abs() <= win)]
        if len(nb) == 0:
            continue
        for name, cx, cy in (("raw", "dx", "dy"), ("flow", "rx", "ry")):
            m = nb[[cx, cy]].to_numpy().mean(axis=0)
            rows.append((name, len(nb), float(np.hypot(r[cx], r[cy])), float(np.hypot(r[cx] - m[0], r[cy] - m[1]))))
    df = pd.DataFrame(rows, columns=["kind", "n_nb", "before", "after"])
    for name, g in df.groupby("kind"):
        print(f"window +-{win} frames, {name}: pairs with neighbours {len(g)}; |res| median before {g.before.median():.1f} after {g.after.median():.1f}; p80 before {g.before.quantile(.8):.1f} after {g.after.quantile(.8):.1f}")
# how residual components split along / across the pitch for returns
print("true gap>=50: |rx| med", T.rx.abs().median().round(1), "|ry| med", T.ry.abs().median().round(1), " raw |dx|", T.dx.abs().median().round(1), "|dy|", T.dy.abs().median().round(1))
