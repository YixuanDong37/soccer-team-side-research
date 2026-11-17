"""DIAGNOSIS (oracle, uses labels): what stands between each annotated return and a link."""
import sys, os; sys.path.insert(0, "C:/Research/plugins/tools"); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reid_lab import data, score
import fitlabels
import numpy as np, pandas as pd, json
rows = []
for seq in data.DEV_CLIPS:
    c = score.clip(seq); p = data.pieces(seq).set_index("piece"); t = data.tracks(seq)
    clean = data.clean_rows(seq)
    ncl = pd.Series(clean).groupby(t.track_id.to_numpy()).sum()
    te = {e["track_id"]: e for e in data.refine(seq)["teams"]["tracks"]}
    def vt(tid):
        e = te.get(tid)
        if not e or e["verdict"] != "consistent" or e.get("samples_against") or e.get("cleared"): return None
        return e["team"] if e.get("nearer_team") == e["team"] else None
    ident = fitlabels.piece_identity(seq)
    cand = {(int(a), int(b)) for a, b, s in data.appearance(seq)["candidate_pairs"]}
    for r in c.returns:
        if r["a"] is None or r["b"] is None: continue
        pa, pb = int(c.piece[r["a"]]), int(c.piece[r["b"]])
        A, B = p.loc[pa], p.loc[pb]
        rows.append(dict(seq=seq, ref=r["ref_id"], gap=r["gap"], pa=pa, pb=pb, same_piece=pa==pb, same_plugin=A.plugin==B.plugin,
                         rows_a=A.rows, rows_b=B.rows, cl_a=int(ncl.get(pa,0)), cl_b=int(ncl.get(pb,0)),
                         role_a=A.role, role_b=B.role, team_a=A.team, team_b=B.team, vt_a=vt(pa), vt_b=vt(pb),
                         id_a=ident.get(pa,(None,))[0], id_b=ident.get(pb,(None,))[0],
                         piece_gap=B["first"]-A["last"], in_cand=(pa,pb) in cand))
d = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_columns", 40)
print(d.to_string())
d.to_csv("diag_returns.csv", index=False)
print("same piece", d.same_piece.sum(), "same plugin", d.same_plugin.sum(), "in cand", d.in_cand.sum())
print("B identity == return ref", (d.id_b==d.ref).sum(), "A identity == ref", (d.id_a==d.ref).sum(), "both", ((d.id_a==d.ref)&(d.id_b==d.ref)).sum())
ok = (d.id_a==d.ref)&(d.id_b==d.ref)&(~d.same_piece)
print("linkable (both pieces purely this person, different pieces):", ok.sum())
print(" of those: cl_a>=5&cl_b>=5:", (ok&(d.cl_a>=5)&(d.cl_b>=5)).sum(), " piece_gap<=250:", (ok&(d.piece_gap<=250)).sum(), " both verified team:", (ok&d.vt_a.notna()&d.vt_b.notna()).sum(),
      " both player:", (ok&(d.role_a=="player")&(d.role_b=="player")).sum(), "piece_gap<=0:", (ok&(d.piece_gap<=0)).sum())
print(" all appearance_match conditions:", (ok&(d.cl_a>=5)&(d.cl_b>=5)&(d.piece_gap<=250)&d.vt_a.notna()&d.vt_b.notna()&(d.role_a=="player")&(d.role_b=="player")).sum())
print(" clean>=1 both:", (ok&(d.cl_a>=1)&(d.cl_b>=1)).sum(), "team label both and equal:", (ok&(d.team_a==d.team_b)&d.team_a.notna()).sum(), "team differ:", (ok&(d.team_a!=d.team_b)&d.team_a.notna()&d.team_b.notna()).sum(), "role differ:", (ok&(d.role_a!=d.role_b)).sum())
