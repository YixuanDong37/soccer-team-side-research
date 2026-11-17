"""ORACLE / DIAGNOSIS ONLY. Short-gap pairs (fragments of one track): how precise would simple
position rules be, judged with weak identities (>= 1 matched row)?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import features  # noqa: E402
import labels  # noqa: E402
import linker  # noqa: E402

pd.set_option("display.width", 250)
rows = []
for seq in data.DEV_CLIPS:
    p = linker.admissible(features.pair_features(seq))
    weak = labels.identity_map(seq, 1)
    strong = labels.identity_map(seq, 5)
    p["wa"], p["wb"] = p.a.map(weak), p.b.map(weak)
    p["sa"], p["sb"] = p.a.map(strong), p.b.map(strong)
    p["seq"] = seq
    q = p[p.gap <= 25].copy()
    # mutual nearest by r_cv among pairs with gap <= 25
    q["rank_b"] = q.groupby("b").r_cv.rank(method="first")
    q["rank_a"] = q.groupby("a").r_cv.rank(method="first")
    q["second_b"] = q.groupby("b").r_cv.transform(lambda s: np.sort(s.to_numpy())[1] if len(s) > 1 else np.inf)
    q["second_a"] = q.groupby("a").r_cv.transform(lambda s: np.sort(s.to_numpy())[1] if len(s) > 1 else np.inf)
    rows.append(q)
q = pd.concat(rows, ignore_index=True)
q["kind"] = q.a_end_edge.astype(int) + q.b_start_edge.astype(int)
q["minrows"] = np.minimum(q.a_rows, q.b_rows)
q["judged"] = q.wa.notna() & q.wb.notna()
q["ok"] = q.judged & (q.wa == q.wb)
m = q[(q.rank_a == 1) & (q.rank_b == 1)]
print("mutual nearest short-gap pairs:", len(m), "judged", int(m.judged.sum()), "same", int(m.ok.sum()))
for rmax in (1.0, 1.5, 2.0, 3.0):
    for gmax in (5, 12, 25):
        for ratio in (1.0, 2.0, 3.0):
            s = m[(m.r_cv <= rmax) & (m.gap <= gmax) & (np.minimum(m.second_a, m.second_b) >= ratio * np.maximum(m.r_cv, 0.5))]
            j = s[s.judged]
            line = f"r_cv<={rmax} gap<={gmax} second>={ratio}x: n={len(s)} judged={len(j)} wrong={int((~j.ok).sum())}"
            for kind in (0, 1, 2):
                jj = j[j.kind == kind]
                line += f" | kind{kind}: {len(s[s.kind == kind])} n, {int((~jj.ok).sum())}/{len(jj)} wrong"
            sm = j[j.minrows <= 12]
            line += f" | minrows<=12: {int((~sm.ok).sum())}/{len(sm)} wrong"
            print(line)
