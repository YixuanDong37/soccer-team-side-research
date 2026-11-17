"""ORACLE / DIAGNOSIS ONLY (reads labels through reid_lab.score). Not a method.
Per scored return: the pieces holding row A and row B, how they sit relative to the gap,
pitch distance between A's end and B's start, implied speed."""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
from reid_lab import data, score
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
rows = []
for seq in data.DEV_CLIPS:
    c = score.clip(seq)
    p = data.pieces(seq).set_index("piece")
    ident = score._identities(c.piece, c.refs)
    for r in c.returns:
        a, b = r["a"], r["b"]
        if a is None or b is None:
            continue
        pa, pb = int(c.piece[a]), int(c.piece[b])
        if pa < 0 or pb < 0:
            continue
        A, B = p.loc[pa], p.loc[pb]
        d = np.nan
        if A.end_xy is not None and B.start_xy is not None:
            d = float(np.hypot(A.end_xy[0] - B.start_xy[0], A.end_xy[1] - B.start_xy[1]))
        rows.append(dict(seq=seq, ref=r["ref_id"], last=r["last"], first=r["first"], gap=r["gap"], pa=pa, pb=pb,
                         a_last=int(A["last"]), b_first=int(B["first"]), a_tail=int(A["last"]) - r["last"], b_head=r["first"] - int(B["first"]),
                         same_plugin=int(A.plugin) == int(B.plugin), ida=ident.get(pa), idb=ident.get(pb),
                         a_rows=int(A.rows), b_rows=int(B.rows), team_a=A.team, team_b=B.team, role_a=A.role, role_b=B.role, d=d,
                         a_edge=bool(A.ends_at_edge), b_edge=bool(B.starts_at_edge)))
df = pd.DataFrame(rows)
df["pgap"] = df.b_first - df.a_last - 1
df["speed"] = df.d / (df.pgap.clip(lower=1) / 25)
df.to_csv("oracle_returns.csv", index=False)
print(df.to_string())
print(len(df))
print("gap quantiles (frames):", df.gap.quantile([.1,.25,.5,.75,.9,1]).to_dict())
print("B piece starts within 12 frames of return first:", (df.b_head.abs() <= 12).sum(), " A piece ends within 12 frames:", (df.a_tail.abs() <= 12).sum())
print("B's piece identity == return ref:", (df.idb == df.ref).sum(), "unknown:", df.idb.isna().sum(), " A's:", (df.ida == df.ref).sum(), "unknown:", df.ida.isna().sum())
print("same plugin:", df.same_plugin.sum())
print("team equal:", (df.team_a == df.team_b).sum(), "team na:", (df.team_a.isna() | df.team_b.isna()).sum())
print("role equal:", (df.role_a == df.role_b).sum())
