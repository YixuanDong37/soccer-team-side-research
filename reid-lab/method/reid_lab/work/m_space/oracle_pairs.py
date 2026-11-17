"""ORACLE / DIAGNOSIS ONLY (uses labels.py). How far do true partners move, and how well does
plain distance / required speed rank the true earlier piece among the candidates?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import geom  # noqa: E402
import labels  # noqa: E402

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 500)

allp = []
for seq in data.DEV_CLIPS:
    e = geom.piece_ends(seq)
    ident = labels.identity_map(seq, 5)
    weak = labels.identity_map(seq, 1)
    e = e.assign(ref=e.piece.map(ident), ref_weak=e.piece.map(weak))
    a = e.add_prefix("a_")
    b = e.add_prefix("b_")
    p = a.merge(b, how="cross")
    p = p[p.a_last < p.b_first].copy()
    p["seq"] = seq
    p["gap"] = p.b_first - p.a_last - 1
    p["dt"] = (p.b_first - p.a_last) / geom.FPS
    p["d"] = np.hypot(p.b_start_x - p.a_end_x, p.b_start_y - p.a_end_y)
    p["known"] = p.a_ref.notna() & p.b_ref.notna()
    p["same"] = p.known & (p.a_ref == p.b_ref)
    allp.append(p)
p = pd.concat(allp, ignore_index=True)
print("pairs", len(p), "known", int(p.known.sum()), "same", int(p.same.sum()))
# immediate predecessor: for each b, the same-person a with the largest a_last
s = p[p.same].sort_values(["seq", "b_piece", "a_last"])
imm = s.groupby(["seq", "b_piece"]).tail(1)
print("entries with a known earlier same-person piece:", len(imm))
imm = imm.assign(speed=imm.d / imm.dt)
print(imm[["gap", "dt", "d", "speed"]].describe(percentiles=[.1, .25, .5, .75, .9, .95]).to_string())
for lo, hi in ((0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800)):
    q = imm[(imm.gap >= lo) & (imm.gap < hi)]
    f = p[p.known & ~p.same & (p.gap >= lo) & (p.gap < hi)]
    ft = f[(f.a_team == f.b_team)]
    print(f"gap [{lo},{hi}): true n={len(q)} d med={q.d.median():.1f} p90={q.d.quantile(.9):.1f} max={q.d.max():.1f} | speed med={q.speed.median():.2f} p90={q.speed.quantile(.9):.2f} | "
          f"false n={len(f)} d med={f.d.median():.1f} p10={f.d.quantile(.1):.1f} | false same team n={len(ft)} d p10={ft.d.quantile(.1):.1f} med={ft.d.median():.1f}")
print("true pairs: same team", int((imm.a_team == imm.b_team).sum()), "either none", int((imm.a_team.isna() | imm.b_team.isna()).sum()), "differ", int(((imm.a_team != imm.b_team) & imm.a_team.notna() & imm.b_team.notna()).sum()))
print("true pairs: same role", int((imm.a_role == imm.b_role).sum()), "of", len(imm))
print("true pairs: same plugin", int((imm.a_plugin == imm.b_plugin).sum()))
print("true pairs end_side", imm.a_end_side.value_counts().to_dict(), "start_side", imm.b_start_side.value_counts().to_dict())
# rank of the true immediate predecessor among candidates by d and by d/dt
p["speed"] = p.d / p.dt
for name, col in (("d", "d"), ("speed", "speed")):
    ranks = []
    for (seq, bp), g in p.groupby(["seq", "b_piece"]):
        t = g[g.same]
        if not len(t):
            continue
        best_true = t[col].min()
        cand = g[(g.a_role == g.b_role)]
        ranks.append(int((cand[col] < best_true).sum()) + 1)
    r = np.array(ranks)
    print(name, "rank of best true among role-compatible candidates: n", len(r), "top1", int((r == 1).sum()), "top2", int((r <= 2).sum()), "top3", int((r <= 3).sum()), "median", np.median(r))
p.to_pickle("oracle_pairs.pkl")
