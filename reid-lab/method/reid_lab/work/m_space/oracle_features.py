"""ORACLE / DIAGNOSIS ONLY. How do the label-free features separate true partners (imm) from
pairs of different people?"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_space")
from reid_lab import data  # noqa: E402
import fitdata  # noqa: E402

pd.set_option("display.width", 250)
p = fitdata.pooled(data.DEV_CLIPS)
k = p[p.known]
T = k[k.imm]
F = k[~k.same]
print("pairs", len(p), "known", len(k), "true imm", len(T), "same not imm", int((k.same & ~k.imm).sum()), "different", len(F))
print("true: reach", int(T.reach.sum()), "/", len(T), " false: reach", int(F.reach.sum()), "/", len(F))
print("true unreachable:")
print(T[~T.reach][["seq", "a", "b", "gap", "d", "r", "speed_req", "a_rows", "b_rows", "a_end_side", "b_start_side"]].to_string())
print("\nsame_team: true", T.same_team.value_counts().to_dict(), " false", F.same_team.value_counts().to_dict())
print("same_role: true", T.same_role.value_counts().to_dict(), " false", F.same_role.value_counts().to_dict())
Tr, Fr = T[T.reach], F[F.reach & (F.same_role == 1) & (F.same_team != 0)]
print("\nafter reach + role + team filters: true", len(Tr), "false", len(Fr))
for lo, hi in ((0, 10), (10, 25), (25, 75), (75, 150), (150, 300), (300, 800)):
    t = Tr[(Tr.gap >= lo) & (Tr.gap < hi)]
    f = Fr[(Fr.gap >= lo) & (Fr.gap < hi)]
    print(f"gap [{lo},{hi}) true n={len(t)} false n={len(f)}")
    for col in ("d", "r", "r_cv", "inview_s", "line_inview"):
        qs = [.25, .5, .75, .9]
        print(f"   {col:12s} true q25/50/75/90: " + " ".join(f"{t[col].quantile(q):6.2f}" for q in qs) + "   false: " + " ".join(f"{f[col].quantile(q):6.2f}" for q in qs))
    print(f"   inview_s==0: true {float((t.inview_s == 0).mean()):.2f} false {float((f.inview_s == 0).mean()):.2f};  inview_s>1: true {float((t.inview_s > 1).mean()):.2f} false {float((f.inview_s > 1).mean()):.2f}")
    print(f"   edges a_end/b_start: true {float(t.a_end_edge.mean()):.2f}/{float(t.b_start_edge.mean()):.2f} false {float(f.a_end_edge.mean()):.2f}/{float(f.b_start_edge.mean()):.2f}")
