"""ORACLE DIAGNOSIS (reads labels through score internals; never a method result).

Upper bound of linking gbf pieces: link consecutive pieces with the same majority annotated
person.  Then: for every scored return, the piece of row B and the piece of row A, their sizes,
and whether keep / mutual link them.
"""
import sys, json
sys.path.insert(0, "C:/Research/plugins/tools")
sys.path.insert(0, "C:/Research/plugins/tools/reid_lab/work/m_keep")
import numpy as np, pandas as pd
from collections import Counter
from reid_lab import data, score
import features

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 80); pd.set_option("display.max_rows", 400)
orc = {}
rows = []
for s in data.DEV_CLIPS:
    c = score.clip(s)
    P = features.piece_table(s)
    ident = {}
    for tid in np.unique(c.piece[c.piece >= 0]):
        top, n = Counter(c.refs[c.piece == tid].tolist()).most_common(1)[0]
        ident[int(tid)] = (int(top), n)
    by = {}
    for tid, (ref, n) in ident.items():
        by.setdefault(ref, []).append(tid)
    links = []
    for ref, ids in by.items():
        ids = sorted(ids, key=lambda i: P[i]["first"])
        links += [[ids[i], ids[i + 1]] for i in range(len(ids) - 1)]
    orc[s] = {"links": links}
    for r in c.returns:
        if r["a"] is None or r["b"] is None:
            continue
        pa, pb = int(c.piece[r["a"]]), int(c.piece[r["b"]])
        # earlier pieces of the same person (majority) before B
        prev = [i for i in by.get(r["ref_id"], []) if P[i]["last"] < c.frames[r["b"]] and i != pb]
        A, B = P[pa], P[pb]
        rows.append({"seq": s, "ref": r["ref_id"], "gap": r["gap"], "pa": pa, "pb": pb, "same_plugin": A["plugin"] == B["plugin"], "same_piece": pa == pb,
                     "rows_a": A["n"], "rows_b": B["n"], "clean_a": A["n_clean"], "clean_b": B["n_clean"],
                     "ident_a": ident.get(pa, (None, 0))[0] == r["ref_id"], "ident_b": ident.get(pb, (None, 0))[0] == r["ref_id"],
                     "n_prev": len(prev), "best_prev_clean": max([P[i]["n_clean"] for i in prev], default=0),
                     "cos_ab": features.cos(A, B), "team_a": A["team"], "team_b": B["team"], "role_a": A["role"], "role_b": B["role"],
                     "b_first": B["first"], "b_frame": int(c.frames[r["b"]]), "a_last": A["last"], "a_frame": int(c.frames[r["a"]])})
print("oracle (same-majority-person chains):", score.evaluate(orc)["pooled"])
d = pd.DataFrame(rows)
d.to_csv("diag_returns.csv", index=False)
print(len(d), "scored returns")
print("same piece (no gbf split at this return):", int(d.same_piece.sum()))
print("same plugin id:", int(d.same_plugin.sum()))
print("B-piece majority is the returning person:", int(d.ident_b.sum()), " A-piece:", int(d.ident_a.sum()))
print("B piece clean rows >=5:", int((d.clean_b >= 5).sum()), " A piece clean>=5:", int((d.clean_a >= 5).sum()), " both:", int(((d.clean_a >= 5) & (d.clean_b >= 5)).sum()))
print("gap > 250 frames:", int((d.gap > 250).sum()))
print(d.groupby(["same_plugin", "same_piece"]).size())
print(d[["rows_b", "clean_b", "rows_a", "clean_a", "gap", "cos_ab"]].describe().round(2))
print(d.round(3).to_string())
