"""DIAGNOSIS: what are the highest-scoring false best-earlier-candidates (team-centred clean mean)?"""
import sys, os, time; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data
import numpy as np, pandas as pd
from core import ctx
from diag import unit
import fitlabels
rows = []
for seq in data.DEV_CLIPS:
    c = ctx(seq); P = c.P; ident = fitlabels.piece_identity(seq)
    X = np.where(c.has[:, None], c.E, 0.0); ok = c.has & (c.tid >= 0); out = X - X[ok & c.clean].mean(axis=0); team = c.t.team.to_numpy()
    for tm in ("left", "right"):
        m = ok & (team == tm)
        if (m & c.clean).sum() > 50: out[m] = X[m] - X[m & c.clean].mean(axis=0)
    X = unit(out); pr = {}
    for pid, r in c.rows_of.items():
        rr = r[c.clean[r]]
        if len(rr) < 5: rr = r[c.has[r]]
        if len(rr): pr[pid] = unit(X[rr].mean(axis=0))
    ids = [int(p) for p in P.piece if P.role[c.index[int(p)]] != "referee" and int(p) in ident]
    for b in ids:
        B = P.iloc[c.index[b]]; cand = []
        for a in ids:
            A = P.iloc[c.index[a]]
            if a == b or not (A["last"] < B["first"]): continue
            if A.team is not None and B.team is not None and A.team != B.team: continue
            cand.append((float(pr[a] @ pr[b]), a))
        if not cand: continue
        cand.sort(reverse=True); s, a = cand[0]; A = P.iloc[c.index[a]]
        g = ident[a][0] == ident[b][0]
        true_best = max([x for x in cand if ident[x[1]][0] == ident[b][0]], default=(None, None))
        rows.append(dict(seq=seq, a=a, b=b, s=round(s, 3), genuine=g, second=round(cand[1][0], 3) if len(cand) > 1 else None, true_best=None if true_best[0] is None else round(true_best[0], 3),
                         ida=ident[a], idb=ident[b], rows_a=A.rows, rows_b=B.rows, cl_a=A.n_clean, cl_b=B.n_clean, gap=B["first"] - A["last"], team_a=A.team, team_b=B.team, role_a=A.role, role_b=B.role, ncand=len(cand)))
d = pd.DataFrame(rows).sort_values("s", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(d[~d.genuine].head(40).to_string())
print("false top1 total", (~d.genuine).sum(), "of", len(d), "; of which a true partner existed", ((~d.genuine) & d.true_best.notna()).sum())
print(d[d.genuine].s.describe())
print("genuine top-1 by clean rows of min side:"); d["mincl"] = d[["cl_a", "cl_b"]].min(axis=1)
print(d.groupby(pd.cut(d.mincl, [-1, 0, 4, 9, 19, 49, 1000])).agg(n=("genuine", "size"), genuine=("genuine", "mean"), s=("s", "median")))
